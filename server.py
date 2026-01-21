"""
Flux Multi-Model Image Generation API Server v4.0.0
Production-ready with authentication, queuing, and flexible response formats

Supported Models:
- FLUX.2-klein-4B: Apache 2.0 (commercial), ~13GB VRAM, 4 steps
- FLUX.2-klein-9B: Non-commercial, ~29GB VRAM, 4 steps
- FLUX.2-dev: Non-commercial, ~24GB VRAM (FP8), 28 steps

Setup:
1. Copy .env.example to .env and configure
2. pip install git+https://github.com/huggingface/diffusers
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
from typing import Optional, Dict, Any

from diffusers import Flux2KleinPipeline, Flux2Pipeline
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import io
from PIL import Image
import logging

from config import config


# ============================================
# Model Configuration
# ============================================
AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "klein-4b": {
        "name": "FLUX.2-klein-4B",
        "repo": "black-forest-labs/FLUX.2-klein-4B",
        "pipeline": "Flux2KleinPipeline",
        "license": "Apache 2.0 (commercial OK)",
        "vram_gb": 13,
        "default_steps": 4,
        "default_guidance": 1.0,
        "description": "Fastest model, commercial use allowed"
    },
    "klein-9b": {
        "name": "FLUX.2-klein-9B",
        "repo": "black-forest-labs/FLUX.2-klein-9B",
        "pipeline": "Flux2KleinPipeline",
        "license": "Non-commercial",
        "vram_gb": 29,
        "default_steps": 4,
        "default_guidance": 1.0,
        "description": "Higher quality, requires RTX 5090 (32GB)"
    },
    "dev": {
        "name": "FLUX.2-dev",
        "repo": "black-forest-labs/FLUX.2-dev",
        "pipeline": "Flux2Pipeline",
        "license": "Non-commercial",
        "vram_gb": 24,
        "default_steps": 28,
        "default_guidance": 3.5,
        "description": "Highest quality, uses FP8 quantization"
    }
}

DEFAULT_MODEL = "klein-4b"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Flux Multi-Model API", version="4.0.0")

# CORS middleware - allow requests from any origin (for web client)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
pipe = None
current_lora = None
current_gpu: int = config.GPU_ID
current_model: str = DEFAULT_MODEL  # Track which model is loaded
generation_queue: asyncio.Queue = None
queue_lock = asyncio.Lock()
model_on_gpu = False


# ============================================
# Dynamic Runtime Settings (changeable via API)
# ============================================
class RuntimeSettings:
    """Server-side defaults that can be changed on the fly via API"""
    def __init__(self):
        self.num_inference_steps: int = 4  # Klein 4B distilled needs only 4 steps!
        self.guidance_scale: float = 1.0   # Klein uses 1.0 guidance
        self.aspect_ratio: str = "9:16"
        self.base_resolution: int = 576
        self.lora_path: Optional[str] = None  # Disabled by default
        self.lora_scale: float = 0.0  # Disabled by default
        self.response_format: str = "base64"
        self.max_sequence_length: int = 512  # T5 encoder max tokens

    def to_dict(self) -> dict:
        return {
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "aspect_ratio": self.aspect_ratio,
            "base_resolution": self.base_resolution,
            "lora_path": self.lora_path,
            "lora_scale": self.lora_scale,
            "response_format": self.response_format,
            "max_sequence_length": self.max_sequence_length
        }


# Global runtime settings instance
runtime_settings = RuntimeSettings()


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

    # Resolution settings - None means use runtime_settings default
    aspect_ratio: Optional[str] = Field(
        default=None,
        description="Aspect ratio (9:16, 1:1, 16:9, custom, etc.). None = use server default."
    )
    base_resolution: Optional[int] = Field(
        default=None, ge=256, le=2048,
        description="Base resolution (shorter side). None = use server default."
    )
    width: Optional[int] = Field(
        default=None, ge=256, le=2048,
        description="Width in pixels (must be multiple of 16). Only used when aspect_ratio='custom'."
    )
    height: Optional[int] = Field(
        default=None, ge=256, le=2048,
        description="Height in pixels (must be multiple of 16). Only used when aspect_ratio='custom'."
    )

    # Generation settings - None means use runtime_settings default
    num_inference_steps: Optional[int] = Field(
        default=None, ge=1, le=50,
        description="Number of denoising steps. None = use server default."
    )
    guidance_scale: Optional[float] = Field(
        default=None, ge=0.0, le=20.0,
        description="CFG scale (Klein 4B uses 1.0). None = use server default."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility. None = random seed."
    )

    # LoRA settings - None means use runtime_settings default
    lora_path: Optional[str] = Field(
        default=None,
        description="LoRA file (.safetensors). None = use server default. Empty string = no LoRA."
    )
    lora_scale: Optional[float] = Field(
        default=None, ge=0.0, le=2.0,
        description="LoRA strength. None = use server default."
    )

    response_format: Optional[str] = Field(
        default=None,
        description="Response format (binary, base64, file). None = use server default."
    )
    max_sequence_length: Optional[int] = Field(
        default=None, ge=77, le=512,
        description="Max tokens for T5 encoder (77-512). CLIP is always 77. None = use server default (512)."
    )

    def get_effective_values(self) -> dict:
        """Get effective values with runtime_settings defaults applied"""
        return {
            "aspect_ratio": self.aspect_ratio or runtime_settings.aspect_ratio,
            "base_resolution": self.base_resolution if self.base_resolution is not None else runtime_settings.base_resolution,
            "num_inference_steps": self.num_inference_steps if self.num_inference_steps is not None else runtime_settings.num_inference_steps,
            "guidance_scale": self.guidance_scale if self.guidance_scale is not None else runtime_settings.guidance_scale,
            "lora_path": self.lora_path if self.lora_path is not None else runtime_settings.lora_path,
            "lora_scale": self.lora_scale if self.lora_scale is not None else runtime_settings.lora_scale,
            "response_format": self.response_format or runtime_settings.response_format,
            "max_sequence_length": self.max_sequence_length if self.max_sequence_length is not None else runtime_settings.max_sequence_length,
        }

    def get_dimensions(self) -> tuple[int, int]:
        """Get final width and height based on aspect_ratio or custom dimensions."""
        effective = self.get_effective_values()
        aspect = effective["aspect_ratio"]
        base_res = effective["base_resolution"]

        if aspect == "custom":
            # Use custom dimensions, round to multiple of 16
            w = self.width or 576
            h = self.height or 1024
            w = ((w + 8) // 16) * 16
            h = ((h + 8) // 16) * 16
            return w, h
        else:
            return calculate_dimensions(aspect, base_res)


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


class SettingsUpdate(BaseModel):
    """Request model for updating runtime settings"""
    num_inference_steps: Optional[int] = Field(None, ge=1, le=50, description="Denoising steps (4 recommended for Klein 4B)")
    guidance_scale: Optional[float] = Field(None, ge=0.0, le=20.0, description="CFG scale (1.0 recommended for Klein 4B)")
    aspect_ratio: Optional[str] = Field(None, description="Default aspect ratio (9:16, 1:1, 16:9, etc.)")
    base_resolution: Optional[int] = Field(None, ge=256, le=2048, description="Base resolution (shorter side)")
    lora_path: Optional[str] = Field(None, description="Default LoRA (filename or path, use empty string to clear)")
    lora_scale: Optional[float] = Field(None, ge=0.0, le=2.0, description="Default LoRA strength")
    response_format: Optional[str] = Field(None, description="Default response format (binary, base64, file)")
    max_sequence_length: Optional[int] = Field(None, ge=77, le=512, description="T5 encoder max tokens (77-512, default 512)")


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


def unload_current_model():
    """Unload the current model from memory"""
    global pipe, current_lora, model_on_gpu

    if pipe is not None:
        del pipe
        pipe = None

    current_lora = None
    model_on_gpu = False

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    logger.info("Previous model unloaded from memory")


def load_pipeline(model_id: str = None, gpu_id: int = None):
    """Load specified Flux model pipeline on specified GPU"""
    global pipe, current_gpu, current_lora, model_on_gpu, current_model

    if model_id is None:
        model_id = current_model

    if model_id not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_id}. Available: {list(AVAILABLE_MODELS.keys())}")

    model_config = AVAILABLE_MODELS[model_id]

    if gpu_id is None:
        gpu_id = current_gpu

    logger.info(f"Loading {model_config['name']} on GPU {gpu_id}...")
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
    required_vram = model_config["vram_gb"]

    if total_gb < required_vram:
        raise RuntimeError(
            f"GPU {gpu_id} has {total_gb:.1f}GB total VRAM, but {model_config['name']} requires ~{required_vram}GB. "
            f"Try a smaller model or different GPU."
        )

    if free_gb < required_vram:
        logger.warning(f"GPU {gpu_id} has {free_gb:.1f}GB free (model needs ~{required_vram}GB)")
        logger.warning("Using CPU offload - this should still work but may be slower.")

    # Show memory status for all GPUs
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        logger.info(f"GPU {i} ({torch.cuda.get_device_name(i)}): {free/1024**3:.1f}/{total/1024**3:.1f} GB free")

    # Unload any existing model first
    unload_current_model()

    # Set default device BEFORE loading
    torch.cuda.set_device(gpu_id)

    logger.info(f"Selected GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")

    try:
        logger.info(f"Loading {model_config['name']} ({model_config['license']})...")
        logger.info(f"Description: {model_config['description']}")

        # Load the appropriate pipeline based on model type
        if model_config["pipeline"] == "Flux2KleinPipeline":
            pipe = Flux2KleinPipeline.from_pretrained(
                model_config["repo"],
                torch_dtype=torch.bfloat16,
            )
        elif model_config["pipeline"] == "Flux2Pipeline":
            # Flux 2 Dev - use FP8 for consumer GPUs
            pipe = Flux2Pipeline.from_pretrained(
                model_config["repo"],
                torch_dtype=torch.bfloat16,
            )
        else:
            raise ValueError(f"Unknown pipeline type: {model_config['pipeline']}")

        # Enable CPU offload for all models
        pipe.enable_model_cpu_offload(gpu_id=gpu_id)

        # Model is ready
        model_on_gpu = True
        current_gpu = gpu_id
        current_model = model_id
        current_lora = None

        # Update runtime settings to model defaults
        runtime_settings.num_inference_steps = model_config["default_steps"]
        runtime_settings.guidance_scale = model_config["default_guidance"]

        free, total = torch.cuda.mem_get_info(gpu_id)
        logger.info(f"Pipeline ready! VRAM: {(total-free)/1024**3:.1f}/{total/1024**3:.1f} GB used")
        logger.info(f"Defaults set: {model_config['default_steps']} steps, {model_config['default_guidance']} guidance")

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
    """
    Load LoRA weights into the pipeline.

    WARNING: Flux Schnell LoRAs are NOT compatible with Klein 4B!
    You need to train new LoRAs specifically for Klein 4B architecture.
    """
    global pipe, current_lora

    # Resolve the path (supports just filename if LORA_DIR configured)
    resolved_path = resolve_lora_path(lora_path)

    if current_lora == resolved_path:
        logger.info(f"LoRA already loaded: {resolved_path}")
        # Just update scale if needed
        pipe.set_adapters(["default"], adapter_weights=[scale])
        return

    try:
        logger.warning("NOTE: Old Schnell LoRAs are NOT compatible with Klein 4B!")
        logger.warning("You need LoRAs trained specifically for FLUX.2-klein-4B architecture.")
        logger.info(f"Attempting to load LoRA from: {resolved_path}")

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
        raise HTTPException(status_code=500, detail=f"LoRA loading failed (note: Schnell LoRAs are NOT compatible with Klein 4B): {str(e)}")


def _sync_generate(request: GenerationRequest) -> tuple:
    """
    Synchronous generation function for use in thread pool.
    Returns (image_bytes, seed, generation_time_ms, width, height, effective_values)
    """
    global pipe, current_gpu

    start_time = time.time()

    # Get effective values (request values with runtime_settings defaults)
    effective = request.get_effective_values()

    # Determine seed
    seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)

    # Calculate dimensions from aspect ratio
    width, height = request.get_dimensions()

    # Load LoRA if specified (empty string means explicitly no LoRA)
    lora_path = effective["lora_path"]
    lora_scale = effective["lora_scale"]

    if lora_path:  # Non-empty string means load LoRA
        load_lora(lora_path, lora_scale)

    steps = effective["num_inference_steps"]
    guidance = effective["guidance_scale"]
    max_seq_len = effective["max_sequence_length"]

    logger.info(f"Generating on GPU {current_gpu}: '{request.prompt[:50]}...' "
                f"seed={seed}, {width}x{height}, steps={steps}, max_tokens={max_seq_len}")

    # Use CPU generator for reproducibility across different GPUs
    generator = torch.Generator(device="cpu").manual_seed(seed)

    result = pipe(
        prompt=request.prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        max_sequence_length=max_seq_len,  # T5 encoder token limit (CLIP is always 77)
    )

    image = result.images[0]

    # Convert to PNG bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()

    generation_time_ms = int((time.time() - start_time) * 1000)
    logger.info(f"Generated in {generation_time_ms}ms, seed={seed}, {width}x{height}")

    return img_bytes, seed, generation_time_ms, width, height, effective


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
            free, total = torch.cuda.mem_get_info(i)
            gpus.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "vram_total_gb": round(total / 1024**3, 1),
                "vram_free_gb": round(free / 1024**3, 1),
                "active": i == current_gpu
            })

    model_config = AVAILABLE_MODELS.get(current_model, {})

    return {
        "status": "online",
        "version": "4.0.0",
        "current_model": {
            "id": current_model,
            "name": model_config.get("name", "Unknown"),
            "license": model_config.get("license", "Unknown"),
            "description": model_config.get("description", "")
        },
        "memory_mode": "cpu_offload",
        "cuda_available": torch.cuda.is_available(),
        "current_gpu": current_gpu,
        "gpus": gpus,
        "auth_required": config.require_auth(),
        "queue_size": generation_queue.qsize() if generation_queue else 0,
        "max_queue_size": config.MAX_QUEUE_SIZE,
        "runtime_settings": runtime_settings.to_dict(),
        "endpoints": {
            "models": "GET /models - list available models",
            "switch_model": "POST /switch-model - switch to different model",
            "settings": "GET /settings - view, POST /settings - update defaults",
            "generate": "POST /generate - generate image",
            "queue": "GET /queue/status - check queue"
        }
    }


@app.get("/models")
async def list_models():
    """List all available models and their specifications"""
    models = []
    for model_id, config in AVAILABLE_MODELS.items():
        models.append({
            "id": model_id,
            "name": config["name"],
            "license": config["license"],
            "vram_required_gb": config["vram_gb"],
            "default_steps": config["default_steps"],
            "default_guidance": config["default_guidance"],
            "description": config["description"],
            "is_current": model_id == current_model
        })

    return {
        "models": models,
        "current_model": current_model,
        "note": "Use POST /switch-model to change models. Model switching takes time as the entire model must be reloaded."
    }


class ModelSwitchRequest(BaseModel):
    model_id: str = Field(description="Model ID to switch to (klein-4b, klein-9b, or dev)")
    gpu_id: Optional[int] = Field(None, ge=0, description="GPU ID to use (optional, uses current GPU if not specified)")


@app.post("/switch-model")
async def switch_model(request: ModelSwitchRequest, _: bool = Depends(verify_api_key)):
    """
    Switch to a different model. This unloads the current model and loads the new one.
    Note: This takes significant time as the entire model must be reloaded (~30-60 seconds).

    Available models:
    - klein-4b: Fastest, commercial use OK (~13GB VRAM)
    - klein-9b: Higher quality, non-commercial (~29GB VRAM, needs RTX 5090)
    - dev: Highest quality, non-commercial (~24GB VRAM with FP8)
    """
    if request.model_id not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {request.model_id}. Available: {list(AVAILABLE_MODELS.keys())}"
        )

    model_config = AVAILABLE_MODELS[request.model_id]
    target_gpu = request.gpu_id if request.gpu_id is not None else current_gpu

    # Check if GPU has enough VRAM
    if torch.cuda.is_available() and target_gpu < torch.cuda.device_count():
        _, total = torch.cuda.mem_get_info(target_gpu)
        total_gb = total / 1024**3
        if total_gb < model_config["vram_gb"]:
            raise HTTPException(
                status_code=400,
                detail=f"GPU {target_gpu} has {total_gb:.1f}GB VRAM, but {model_config['name']} requires ~{model_config['vram_gb']}GB"
            )

    if request.model_id == current_model and (request.gpu_id is None or request.gpu_id == current_gpu):
        return {
            "status": "already_loaded",
            "model_id": current_model,
            "model_name": model_config["name"],
            "gpu_id": current_gpu
        }

    # Wait for any ongoing generation to complete
    async with queue_lock:
        logger.info(f"Switching from {current_model} to {request.model_id}...")

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: load_pipeline(request.model_id, target_gpu)
            )

            return {
                "status": "success",
                "model_id": current_model,
                "model_name": AVAILABLE_MODELS[current_model]["name"],
                "license": AVAILABLE_MODELS[current_model]["license"],
                "gpu_id": current_gpu,
                "default_steps": AVAILABLE_MODELS[current_model]["default_steps"],
                "default_guidance": AVAILABLE_MODELS[current_model]["default_guidance"],
                "message": f"Model switched successfully. Runtime settings updated to model defaults."
            }

        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            raise HTTPException(status_code=500, detail=f"Model switch failed: {str(e)}")


@app.get("/queue/status", response_model=QueueStatus)
async def queue_status():
    """Check current queue status - no auth required"""
    return QueueStatus(
        queue_size=generation_queue.qsize(),
        max_queue_size=config.MAX_QUEUE_SIZE,
        is_generating=queue_lock.locked()
    )


# ============================================
# Settings Management Endpoints
# ============================================
@app.get("/settings")
async def get_settings():
    """
    Get current runtime settings (defaults used when not specified in request).
    No authentication required - useful for checking current configuration.
    """
    return {
        "settings": runtime_settings.to_dict(),
        "description": "These are the default values used when not specified in generation request"
    }


@app.post("/settings")
async def update_settings(
    update: SettingsUpdate,
    _: bool = Depends(verify_api_key)
):
    """
    Update runtime settings on the fly. Only provided fields are updated.

    Examples:
    - Change steps: {"num_inference_steps": 15}
    - Change LoRA: {"lora_path": "mymodel", "lora_scale": 0.7}
    - Clear LoRA: {"lora_path": ""}
    - Multiple: {"num_inference_steps": 25, "aspect_ratio": "1:1"}
    """
    changes = {}

    if update.num_inference_steps is not None:
        runtime_settings.num_inference_steps = update.num_inference_steps
        changes["num_inference_steps"] = update.num_inference_steps

    if update.guidance_scale is not None:
        runtime_settings.guidance_scale = update.guidance_scale
        changes["guidance_scale"] = update.guidance_scale

    if update.aspect_ratio is not None:
        # Validate aspect ratio
        valid_ratios = list(ASPECT_RATIOS.keys()) + ["custom"]
        if update.aspect_ratio not in valid_ratios:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid aspect_ratio. Valid options: {valid_ratios}"
            )
        runtime_settings.aspect_ratio = update.aspect_ratio
        changes["aspect_ratio"] = update.aspect_ratio

    if update.base_resolution is not None:
        runtime_settings.base_resolution = update.base_resolution
        changes["base_resolution"] = update.base_resolution

    if update.lora_path is not None:
        # Empty string means clear LoRA, otherwise set it
        runtime_settings.lora_path = update.lora_path if update.lora_path else None
        changes["lora_path"] = runtime_settings.lora_path

    if update.lora_scale is not None:
        runtime_settings.lora_scale = update.lora_scale
        changes["lora_scale"] = update.lora_scale

    if update.response_format is not None:
        valid_formats = ["binary", "base64", "file"]
        if update.response_format not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid response_format. Valid options: {valid_formats}"
            )
        runtime_settings.response_format = update.response_format
        changes["response_format"] = update.response_format

    if update.max_sequence_length is not None:
        runtime_settings.max_sequence_length = update.max_sequence_length
        changes["max_sequence_length"] = update.max_sequence_length

    logger.info(f"Settings updated: {changes}")

    return {
        "status": "updated",
        "changes": changes,
        "current_settings": runtime_settings.to_dict()
    }


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
            img_bytes, seed, generation_time_ms, width, height, effective = await loop.run_in_executor(
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

    # Get effective response format
    resp_format = effective["response_format"]

    # Return based on requested format
    if resp_format == "binary":
        return Response(
            content=img_bytes,
            media_type="image/png",
            headers={
                "X-Seed": str(seed),
                "X-Generation-Time-Ms": str(generation_time_ms),
                "X-Width": str(width),
                "X-Height": str(height),
                "X-Model": current_model,
                "X-Model-Name": AVAILABLE_MODELS[current_model]["name"]
            }
        )

    elif resp_format == "base64":
        return JSONResponse(content={
            "image": base64.b64encode(img_bytes).decode("utf-8"),
            "seed": seed,
            "generation_time_ms": generation_time_ms,
            "width": width,
            "height": height,
            "model": current_model,
            "model_name": AVAILABLE_MODELS[current_model]["name"],
            "aspect_ratio": effective["aspect_ratio"],
            "num_steps": effective["num_inference_steps"],
            "guidance_scale": effective["guidance_scale"],
            "lora_path": effective["lora_path"],
            "lora_scale": effective["lora_scale"]
        })

    elif resp_format == "file":
        # Save to file
        filename = f"flux_{current_model}_{seed}_{int(time.time())}.png"
        file_path = Path(config.OUTPUT_DIR) / filename
        with open(file_path, "wb") as f:
            f.write(img_bytes)

        return JSONResponse(content={
            "file_path": str(file_path.absolute()),
            "seed": seed,
            "generation_time_ms": generation_time_ms,
            "width": width,
            "height": height,
            "model": current_model,
            "model_name": AVAILABLE_MODELS[current_model]["name"],
            "aspect_ratio": effective["aspect_ratio"],
            "num_steps": effective["num_inference_steps"],
            "guidance_scale": effective["guidance_scale"],
            "lora_path": effective["lora_path"],
            "lora_scale": effective["lora_scale"]
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
    ==========================================
       Flux Multi-Model API Server v4.0.0
    ==========================================
    Available Models:
    - klein-4b: Apache 2.0, ~13GB VRAM, 4 steps
    - klein-9b: Non-commercial, ~29GB VRAM, 4 steps
    - dev: Non-commercial, ~24GB VRAM, 28 steps
    ==========================================

    Server: http://{config.HOST}:{config.PORT}
    Auth: {'Enabled' if config.require_auth() else 'Disabled'}
    GPU: {config.GPU_ID}
    Default Model: {DEFAULT_MODEL}

    API Endpoints:
    - GET  /models        - List available models
    - POST /switch-model  - Switch to different model
    - POST /generate      - Generate image
    - GET  /settings      - View settings
    - POST /settings      - Update settings
    ==========================================
    """)

    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info"
    )
