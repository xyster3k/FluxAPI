"""
Flux Schnell Image Generation API Server
Production-ready with authentication, queuing, and flexible response formats

Setup:
1. Copy .env.example to .env and configure
2. pip install -r requirements.txt
3. python server.py
4. Server runs on http://localhost:7860
"""

import os
# Fix CUDA memory fragmentation - MUST be set before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import asyncio
import time
import random
import base64
import os
from pathlib import Path
from enum import Enum
from typing import Optional

from diffusers import FluxPipeline
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import io
from PIL import Image
import logging

from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Flux Schnell API", version="2.0.0")

# Global state
pipe = None
current_lora = None
current_gpu: int = config.GPU_ID
generation_queue: asyncio.Queue = None
queue_lock = asyncio.Lock()
model_on_gpu = False
last_generation_time = 0
unload_task = None
MODEL_IDLE_TIMEOUT = 30  # Seconds to keep model on GPU after last generation


class ResponseFormat(str, Enum):
    BINARY = "binary"
    BASE64 = "base64"
    FILE = "file"


class GenerationRequest(BaseModel):
    prompt: str
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    num_inference_steps: int = Field(default=4, ge=1, le=50)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=20.0)
    seed: Optional[int] = Field(default=None, description="Random seed. None = random")
    lora_path: Optional[str] = None
    lora_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    response_format: ResponseFormat = ResponseFormat.BINARY


class GenerationResponse(BaseModel):
    image: str  # Base64 encoded image
    seed: int
    generation_time_ms: int
    width: int
    height: int


class FileResponse(BaseModel):
    file_path: str
    seed: int
    generation_time_ms: int
    width: int
    height: int


class QueueStatus(BaseModel):
    queue_size: int
    max_queue_size: int
    is_generating: bool


# Authentication dependency
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key if authentication is enabled"""
    if not config.require_auth():
        return True

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header."
        )

    if x_api_key != config.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    return True


FLUX_MIN_VRAM_GB = 25  # Flux Schnell needs ~25GB VRAM


def find_best_gpu() -> int:
    """Find GPU with most free VRAM that can fit the model"""
    best_gpu = 0
    best_free = 0

    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        free_gb = free / 1024**3
        if free_gb > best_free:
            best_free = free_gb
            best_gpu = i

    return best_gpu


def load_pipeline(gpu_id: int = None):
    """Load Flux Schnell pipeline on specified GPU"""
    global pipe, current_gpu, current_lora, model_on_gpu

    if gpu_id is None:
        gpu_id = current_gpu

    logger.info(f"Loading Flux Schnell pipeline on GPU {gpu_id}...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"GPU count: {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    if gpu_id >= torch.cuda.device_count():
        raise RuntimeError(f"GPU {gpu_id} not found. Available: 0-{torch.cuda.device_count()-1}")

    # Check if selected GPU has enough VRAM
    free, total = torch.cuda.mem_get_info(gpu_id)
    free_gb = free / 1024**3
    total_gb = total / 1024**3

    if free_gb < FLUX_MIN_VRAM_GB:
        # Try to find a better GPU
        best_gpu = find_best_gpu()
        best_free, _ = torch.cuda.mem_get_info(best_gpu)
        best_free_gb = best_free / 1024**3

        if best_free_gb >= FLUX_MIN_VRAM_GB and best_gpu != gpu_id:
            logger.warning(f"GPU {gpu_id} has only {free_gb:.1f}GB free, need {FLUX_MIN_VRAM_GB}GB")
            logger.warning(f"Switching to GPU {best_gpu} with {best_free_gb:.1f}GB free")
            gpu_id = best_gpu
        else:
            logger.warning(f"GPU {gpu_id} has {free_gb:.1f}GB free, Flux needs ~{FLUX_MIN_VRAM_GB}GB!")
            logger.warning("This may fail. Consider using a GPU with more VRAM.")

    device = f"cuda:{gpu_id}"

    # Show memory status for all GPUs
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        logger.info(f"GPU {i} ({torch.cuda.get_device_name(i)}): {free/1024**3:.1f}/{total/1024**3:.1f} GB free")

    # Clear any existing allocations
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # Set default device BEFORE loading
    torch.cuda.set_device(gpu_id)

    logger.info(f"Selected GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")

    try:
        logger.info(f"Loading Flux Schnell model to CPU (will move to GPU on first request)...")

        # Load model to CPU first - will be moved to GPU on demand
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
        )

        # Keep model on CPU for now - will move to GPU when generation requested
        model_on_gpu = False
        current_gpu = gpu_id
        current_lora = None

        logger.info("Pipeline loaded to CPU RAM! Will move to GPU on first generation request.")

        return pipe

    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"GPU {gpu_id} out of memory! Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        raise


def move_model_to_gpu():
    """Move model from CPU to GPU for generation"""
    global pipe, model_on_gpu, current_gpu

    if model_on_gpu:
        return  # Already on GPU

    device = f"cuda:{current_gpu}"
    logger.info(f"Moving model to GPU {current_gpu}...")
    start_time = time.time()

    pipe.to(device)
    model_on_gpu = True

    move_time = time.time() - start_time
    free, total = torch.cuda.mem_get_info(current_gpu)
    logger.info(f"Model moved to GPU in {move_time:.1f}s. VRAM: {(total-free)/1024**3:.1f}/{total/1024**3:.1f} GB used")


def move_model_to_cpu():
    """Move model from GPU to CPU to free VRAM"""
    global pipe, model_on_gpu

    if not model_on_gpu:
        return  # Already on CPU

    logger.info("Moving model to CPU to free VRAM...")
    start_time = time.time()

    pipe.to("cpu")
    torch.cuda.empty_cache()
    model_on_gpu = False

    move_time = time.time() - start_time
    free, total = torch.cuda.mem_get_info(current_gpu)
    logger.info(f"Model moved to CPU in {move_time:.1f}s. VRAM: {(total-free)/1024**3:.1f}/{total/1024**3:.1f} GB used")


async def schedule_model_unload():
    """Schedule model unload after idle timeout"""
    global unload_task, last_generation_time

    # Cancel existing unload task if any
    if unload_task and not unload_task.done():
        unload_task.cancel()

    async def unload_after_timeout():
        await asyncio.sleep(MODEL_IDLE_TIMEOUT)
        # Check if no new generations happened during sleep
        if time.time() - last_generation_time >= MODEL_IDLE_TIMEOUT:
            if model_on_gpu and generation_queue.qsize() == 0:
                logger.info(f"Model idle for {MODEL_IDLE_TIMEOUT}s, moving to CPU...")
                # Run in executor since it's blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, move_model_to_cpu)

    unload_task = asyncio.create_task(unload_after_timeout())


def load_lora(lora_path: str, scale: float = 1.0):
    """Load LoRA weights into the pipeline"""
    global pipe, current_lora

    if current_lora == lora_path:
        logger.info(f"LoRA already loaded: {lora_path}")
        # Just update scale if needed
        pipe.set_adapters(["default"], adapter_weights=[scale])
        return

    try:
        logger.info(f"Loading LoRA from: {lora_path}")

        # Unload previous LoRA if any
        if current_lora is not None:
            pipe.unload_lora_weights()

        # Load new LoRA
        pipe.load_lora_weights(lora_path, adapter_name="default")
        pipe.set_adapters(["default"], adapter_weights=[scale])

        current_lora = lora_path
        logger.info(f"LoRA loaded successfully with scale {scale}")

    except Exception as e:
        logger.error(f"Failed to load LoRA: {e}")
        raise HTTPException(status_code=500, detail=f"LoRA loading failed: {str(e)}")


def _sync_generate(request: GenerationRequest) -> tuple:
    """
    Synchronous generation function for use in thread pool.
    Returns (image_bytes, seed, generation_time_ms)
    """
    global pipe, current_gpu, last_generation_time

    start_time = time.time()

    # Determine seed
    seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)

    # Move model to GPU if not already there
    move_model_to_gpu()

    # Load LoRA if specified
    if request.lora_path:
        load_lora(request.lora_path, request.lora_scale)

    logger.info(f"Generating on GPU {current_gpu}: '{request.prompt[:50]}...' seed={seed}")

    # Generator on GPU since model is now on GPU
    device = f"cuda:{current_gpu}"
    generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        prompt=request.prompt,
        width=request.width,
        height=request.height,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        generator=generator,
    )

    image = result.images[0]

    # Convert to PNG bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()

    generation_time_ms = int((time.time() - start_time) * 1000)
    logger.info(f"Generated in {generation_time_ms}ms, seed={seed}")

    return img_bytes, seed, generation_time_ms


@app.on_event("startup")
async def startup_event():
    """Initialize on server start"""
    global generation_queue

    # Initialize queue
    generation_queue = asyncio.Queue(maxsize=config.MAX_QUEUE_SIZE)

    # Create output directory for file responses
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load the model
    load_pipeline()


@app.get("/")
async def root():
    """Health check endpoint - no auth required"""
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpus.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 1),
                "active": i == current_gpu
            })

    return {
        "status": "online",
        "model": "FLUX.1-schnell",
        "version": "2.2.0",
        "cuda_available": torch.cuda.is_available(),
        "current_gpu": current_gpu,
        "model_on_gpu": model_on_gpu,
        "model_idle_timeout_sec": MODEL_IDLE_TIMEOUT,
        "gpus": gpus,
        "auth_required": config.require_auth(),
        "queue_size": generation_queue.qsize() if generation_queue else 0,
        "max_queue_size": config.MAX_QUEUE_SIZE,
    }


@app.get("/queue/status", response_model=QueueStatus)
async def queue_status():
    """Check current queue status - no auth required"""
    return QueueStatus(
        queue_size=generation_queue.qsize(),
        max_queue_size=config.MAX_QUEUE_SIZE,
        is_generating=queue_lock.locked()
    )


@app.post("/generate")
async def generate_image(
    request: GenerationRequest,
    _: bool = Depends(verify_api_key)
):
    """
    Generate image from prompt.

    Response format depends on request.response_format:
    - binary: Raw PNG bytes
    - base64: JSON with base64-encoded image + metadata
    - file: JSON with file path + metadata
    """
    global pipe

    if pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    # Check queue capacity
    if generation_queue.qsize() >= config.MAX_QUEUE_SIZE:
        raise HTTPException(
            status_code=503,
            detail=f"Queue full ({config.MAX_QUEUE_SIZE} requests). Try again later."
        )

    # Track this request in queue
    await generation_queue.put(1)

    try:
        # Wait for exclusive access to GPU
        async with queue_lock:
            # Run generation in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            img_bytes, seed, generation_time_ms = await loop.run_in_executor(
                None,
                lambda: _sync_generate(request)
            )

            # Update last generation time and schedule unload
            global last_generation_time
            last_generation_time = time.time()

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    finally:
        # Always remove from queue when done
        try:
            generation_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

        # Schedule model unload after idle timeout (only if queue is empty)
        if generation_queue.qsize() == 0:
            await schedule_model_unload()

    # Return based on requested format
    if request.response_format == ResponseFormat.BINARY:
        return Response(
            content=img_bytes,
            media_type="image/png",
            headers={
                "X-Seed": str(seed),
                "X-Generation-Time-Ms": str(generation_time_ms)
            }
        )

    elif request.response_format == ResponseFormat.BASE64:
        return JSONResponse(content={
            "image": base64.b64encode(img_bytes).decode("utf-8"),
            "seed": seed,
            "generation_time_ms": generation_time_ms,
            "width": request.width,
            "height": request.height
        })

    elif request.response_format == ResponseFormat.FILE:
        # Save to file
        filename = f"flux_{seed}_{int(time.time())}.png"
        file_path = Path(config.OUTPUT_DIR) / filename
        with open(file_path, "wb") as f:
            f.write(img_bytes)

        return JSONResponse(content={
            "file_path": str(file_path.absolute()),
            "seed": seed,
            "generation_time_ms": generation_time_ms,
            "width": request.width,
            "height": request.height
        })


@app.post("/unload-lora")
async def unload_lora(_: bool = Depends(verify_api_key)):
    """Unload currently loaded LoRA"""
    global pipe, current_lora

    if current_lora is None:
        return {"status": "no_lora_loaded"}

    try:
        pipe.unload_lora_weights()
        current_lora = None
        logger.info("LoRA unloaded")
        return {"status": "success"}

    except Exception as e:
        logger.error(f"Failed to unload LoRA: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class GPUSwitchRequest(BaseModel):
    gpu_id: int = Field(ge=0, description="GPU ID to switch to (0 or 1)")


@app.post("/switch-gpu")
async def switch_gpu(request: GPUSwitchRequest, _: bool = Depends(verify_api_key)):
    """
    Switch to a different GPU. This reloads the model on the new GPU.
    Note: This takes time as it needs to reload the entire model.
    """
    global current_gpu

    gpu_count = torch.cuda.device_count()
    if request.gpu_id >= gpu_count:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid GPU ID. Available GPUs: 0-{gpu_count - 1}"
        )

    if request.gpu_id == current_gpu:
        return {
            "status": "already_on_gpu",
            "gpu_id": current_gpu,
            "gpu_name": torch.cuda.get_device_name(current_gpu)
        }

    # Wait for any ongoing generation to complete
    async with queue_lock:
        logger.info(f"Switching from GPU {current_gpu} to GPU {request.gpu_id}...")

        try:
            # Clear current model from memory
            global pipe
            del pipe
            torch.cuda.empty_cache()

            # Load on new GPU
            load_pipeline(request.gpu_id)

            return {
                "status": "success",
                "gpu_id": current_gpu,
                "gpu_name": torch.cuda.get_device_name(current_gpu)
            }

        except Exception as e:
            logger.error(f"Failed to switch GPU: {e}")
            raise HTTPException(status_code=500, detail=f"GPU switch failed: {str(e)}")


if __name__ == "__main__":
    print(f"""
    ========================================
    Flux Schnell API Server v2.2.0
    ========================================
    Host: {config.HOST}
    Port: {config.PORT}
    Auth: {'Enabled' if config.require_auth() else 'Disabled (set FLUX_API_KEY in .env)'}
    Max Queue: {config.MAX_QUEUE_SIZE}
    GPU Idle Timeout: {MODEL_IDLE_TIMEOUT}s
    ========================================
    Model loads to CPU on startup.
    First request moves model to GPU.
    Model stays on GPU for {MODEL_IDLE_TIMEOUT}s after last request.
    ========================================
    """)

    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info"
    )
