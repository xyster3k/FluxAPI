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

app = FastAPI(title="Flux Schnell API", version="2.4.0")

# Global state
pipe = None
current_lora = None
current_gpu: int = config.GPU_ID
generation_queue: asyncio.Queue = None
queue_lock = asyncio.Lock()
model_on_gpu = False


class ResponseFormat(str, Enum):
    BINARY = "binary"
    BASE64 = "base64"
    FILE = "file"


class AspectRatio(str, Enum):
    """Common aspect ratios for image generation"""
    PORTRAIT_9_16 = "9:16"      # Vertical/portrait (default)
    PORTRAIT_2_3 = "2:3"        # Portrait
    PORTRAIT_3_4 = "3:4"        # Portrait
    SQUARE = "1:1"              # Square
    LANDSCAPE_4_3 = "4:3"       # Landscape
    LANDSCAPE_3_2 = "3:2"       # Landscape
    LANDSCAPE_16_9 = "16:9"     # Widescreen
    CUSTOM = "custom"           # Use width/height directly


# Aspect ratio multipliers (width_mult, height_mult)
ASPECT_RATIOS = {
    "9:16": (9, 16),
    "2:3": (2, 3),
    "3:4": (3, 4),
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:2": (3, 2),
    "16:9": (16, 9),
}


def calculate_dimensions(aspect_ratio: str, base_resolution: int = 576) -> tuple[int, int]:
    """
    Calculate width and height from aspect ratio.
    Uses base_resolution as the shorter dimension.
    Returns dimensions as multiples of 16 (Flux requirement).
    """
    if aspect_ratio not in ASPECT_RATIOS:
        return base_resolution, base_resolution

    w_mult, h_mult = ASPECT_RATIOS[aspect_ratio]

    # Determine which dimension is shorter
    if w_mult <= h_mult:
        # Portrait or square: width is shorter
        width = base_resolution
        height = int(base_resolution * h_mult / w_mult)
    else:
        # Landscape: height is shorter
        height = base_resolution
        width = int(base_resolution * w_mult / h_mult)

    # Round to nearest multiple of 16 (Flux requirement)
    width = ((width + 8) // 16) * 16
    height = ((height + 8) // 16) * 16

    return width, height


class GenerationRequest(BaseModel):
    prompt: str

    # Resolution settings
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.PORTRAIT_9_16,
        description="Aspect ratio. Use 'custom' to specify width/height directly."
    )
    base_resolution: int = Field(
        default=576, ge=256, le=2048,
        description="Base resolution (shorter side). Only used when aspect_ratio is not 'custom'."
    )
    width: Optional[int] = Field(
        default=None, ge=256, le=2048,
        description="Width in pixels (must be multiple of 16). Only used when aspect_ratio='custom'."
    )
    height: Optional[int] = Field(
        default=None, ge=256, le=2048,
        description="Height in pixels (must be multiple of 16). Only used when aspect_ratio='custom'."
    )

    # Generation settings
    num_inference_steps: int = Field(
        default=20, ge=1, le=50,
        description="Number of denoising steps."
    )
    guidance_scale: float = Field(
        default=0.0, ge=0.0, le=20.0,
        description="Classifier-free guidance scale. Note: Flux Schnell ignores this (use 0.0)."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility. None = random seed."
    )

    # LoRA settings
    lora_path: Optional[str] = Field(
        default=None,
        description="Path to LoRA file (.safetensors)"
    )
    lora_scale: float = Field(
        default=1.0, ge=0.0, le=2.0,
        description="LoRA influence strength (0.0 = no effect, 1.0 = full effect)"
    )

    response_format: ResponseFormat = ResponseFormat.BINARY

    def get_dimensions(self) -> tuple[int, int]:
        """Get final width and height based on aspect_ratio or custom dimensions."""
        if self.aspect_ratio == AspectRatio.CUSTOM:
            # Use custom dimensions, round to multiple of 16
            w = self.width or 576
            h = self.height or 1024
            w = ((w + 8) // 16) * 16
            h = ((h + 8) // 16) * 16
            return w, h
        else:
            return calculate_dimensions(self.aspect_ratio.value, self.base_resolution)


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
        logger.info(f"Loading Flux Schnell model with CPU offload for GPU {gpu_id}...")

        # Load model with CPU offload - keeps only active layers on GPU
        # This is required because Flux uses ~25GB and inference needs additional memory
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
        )

        # Enable CPU offload - moves layers to GPU only when needed
        pipe.enable_model_cpu_offload(gpu_id=gpu_id)

        # Model is ready (uses CPU offload, not fully on GPU)
        model_on_gpu = True  # Logically on GPU (offload handles the details)
        current_gpu = gpu_id
        current_lora = None

        free, total = torch.cuda.mem_get_info(gpu_id)
        logger.info(f"Pipeline ready with CPU offload! VRAM: {(total-free)/1024**3:.1f}/{total/1024**3:.1f} GB used")

        return pipe

    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"GPU {gpu_id} out of memory! Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        raise


def clear_gpu_cache():
    """Clear GPU cache after generation to free up memory"""
    torch.cuda.empty_cache()
    import gc
    gc.collect()


def resolve_lora_path(lora_path: str) -> str:
    """
    Resolve LoRA path - supports:
    1. Full absolute path: "E:/path/to/lora.safetensors"
    2. Just filename (if LORA_DIR configured): "pinterestshort1" or "pinterestshort1.safetensors"
    """
    # If it's already an absolute path that exists, use it
    if os.path.isabs(lora_path) and os.path.exists(lora_path):
        return lora_path

    # If LORA_DIR is configured, try to find the file there
    if config.LORA_DIR:
        lora_dir = Path(config.LORA_DIR)

        # Try exact filename
        full_path = lora_dir / lora_path
        if full_path.exists():
            return str(full_path)

        # Try adding .safetensors extension
        if not lora_path.endswith('.safetensors'):
            full_path = lora_dir / f"{lora_path}.safetensors"
            if full_path.exists():
                return str(full_path)

    # If nothing found, return original (will fail with clear error)
    return lora_path


def load_lora(lora_path: str, scale: float = 1.0):
    """Load LoRA weights into the pipeline"""
    global pipe, current_lora

    # Resolve the path (supports just filename if LORA_DIR configured)
    resolved_path = resolve_lora_path(lora_path)

    if current_lora == resolved_path:
        logger.info(f"LoRA already loaded: {resolved_path}")
        # Just update scale if needed
        pipe.set_adapters(["default"], adapter_weights=[scale])
        return

    try:
        logger.info(f"Loading LoRA from: {resolved_path}")

        # Unload previous LoRA if any
        if current_lora is not None:
            pipe.unload_lora_weights()

        # Load new LoRA
        pipe.load_lora_weights(resolved_path, adapter_name="default")
        pipe.set_adapters(["default"], adapter_weights=[scale])

        current_lora = resolved_path
        logger.info(f"LoRA loaded successfully with scale {scale}")

    except Exception as e:
        logger.error(f"Failed to load LoRA: {e}")
        raise HTTPException(status_code=500, detail=f"LoRA loading failed: {str(e)}")


def _sync_generate(request: GenerationRequest) -> tuple:
    """
    Synchronous generation function for use in thread pool.
    Returns (image_bytes, seed, generation_time_ms, width, height)
    """
    global pipe, current_gpu

    start_time = time.time()

    # Determine seed
    seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)

    # Calculate dimensions from aspect ratio
    width, height = request.get_dimensions()

    # Load LoRA if specified
    if request.lora_path:
        load_lora(request.lora_path, request.lora_scale)

    logger.info(f"Generating on GPU {current_gpu}: '{request.prompt[:50]}...' "
                f"seed={seed}, {width}x{height}, steps={request.num_inference_steps}")

    # Use CPU generator for reproducibility across different GPUs
    generator = torch.Generator(device="cpu").manual_seed(seed)

    result = pipe(
        prompt=request.prompt,
        width=width,
        height=height,
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
    logger.info(f"Generated in {generation_time_ms}ms, seed={seed}, {width}x{height}")

    return img_bytes, seed, generation_time_ms, width, height


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
        "version": "2.4.0",
        "memory_mode": "cpu_offload",
        "cuda_available": torch.cuda.is_available(),
        "current_gpu": current_gpu,
        "gpus": gpus,
        "auth_required": config.require_auth(),
        "queue_size": generation_queue.qsize() if generation_queue else 0,
        "max_queue_size": config.MAX_QUEUE_SIZE,
        "default_settings": {
            "aspect_ratio": "9:16",
            "base_resolution": 576,
            "num_inference_steps": 20,
            "guidance_scale": 0.75
        }
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
            img_bytes, seed, generation_time_ms, width, height = await loop.run_in_executor(
                None,
                lambda: _sync_generate(request)
            )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    finally:
        # Always remove from queue when done
        try:
            generation_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

        # Clear GPU cache after generation
        clear_gpu_cache()

    # Return based on requested format
    if request.response_format == ResponseFormat.BINARY:
        return Response(
            content=img_bytes,
            media_type="image/png",
            headers={
                "X-Seed": str(seed),
                "X-Generation-Time-Ms": str(generation_time_ms),
                "X-Width": str(width),
                "X-Height": str(height)
            }
        )

    elif request.response_format == ResponseFormat.BASE64:
        return JSONResponse(content={
            "image": base64.b64encode(img_bytes).decode("utf-8"),
            "seed": seed,
            "generation_time_ms": generation_time_ms,
            "width": width,
            "height": height,
            "aspect_ratio": request.aspect_ratio.value,
            "num_steps": request.num_inference_steps
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
            "width": width,
            "height": height,
            "aspect_ratio": request.aspect_ratio.value,
            "num_steps": request.num_inference_steps
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
    Flux Schnell API Server v2.4.0
    ========================================
    Host: {config.HOST}
    Port: {config.PORT}
    Auth: {'Enabled' if config.require_auth() else 'Disabled (set FLUX_API_KEY in .env)'}
    Max Queue: {config.MAX_QUEUE_SIZE}
    Memory Mode: CPU Offload
    ========================================
    Default Settings:
      Aspect Ratio: 9:16 (portrait)
      Base Resolution: 576px (shorter side)
      Inference Steps: 20
      LoRA Scale: 0.1
    ========================================
    Using CPU offload - model layers move to
    GPU only when needed during inference.
    ========================================
    """)

    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info"
    )
