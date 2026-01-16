"""
Flux Schnell Image Generation API Server
For use with Telegram bots - runs on your RTX 5090

Setup:
1. pip install -r requirements.txt
2. python server.py
3. Server runs on http://localhost:7860
"""

import torch
from diffusers import FluxPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from peft import PeftModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn
import io
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Flux Schnell API", version="1.0.0")

# Global pipeline - loaded once at startup
pipe = None
current_lora = None

class GenerationRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 4  # Flux Schnell is fast - 4 steps is enough
    guidance_scale: float = 0.0  # Schnell doesn't use guidance
    lora_path: str = None  # Optional LORA path
    lora_scale: float = 1.0  # LORA strength


def load_pipeline():
    """Load Flux Schnell pipeline - called once at startup"""
    global pipe

    logger.info("Loading Flux Schnell pipeline...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    try:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            device_map="balanced"  # Automatically distribute across GPU/CPU
        )

        # Enable memory optimizations
        pipe.enable_model_cpu_offload()  # Offload models to CPU when not in use

        logger.info("✅ Pipeline loaded successfully!")
        return pipe

    except Exception as e:
        logger.error(f"❌ Failed to load pipeline: {e}")
        raise


def load_lora(lora_path: str, scale: float = 1.0):
    """Load LORA weights into the pipeline"""
    global pipe, current_lora

    if current_lora == lora_path:
        logger.info(f"LORA already loaded: {lora_path}")
        return

    try:
        logger.info(f"Loading LORA from: {lora_path}")

        # Unload previous LORA if any
        if current_lora is not None:
            pipe.unload_lora_weights()

        # Load new LORA
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=scale)

        current_lora = lora_path
        logger.info(f"✅ LORA loaded successfully with scale {scale}")

    except Exception as e:
        logger.error(f"❌ Failed to load LORA: {e}")
        raise HTTPException(status_code=500, detail=f"LORA loading failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Load pipeline when server starts"""
    load_pipeline()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "model": "FLUX.1-schnell",
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }


@app.post("/generate")
async def generate_image(request: GenerationRequest):
    """
    Generate image from prompt

    Returns: PNG image as binary response
    """
    global pipe

    if pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    try:
        # Load LORA if specified
        if request.lora_path:
            load_lora(request.lora_path, request.lora_scale)

        logger.info(f"Generating image: {request.prompt[:50]}...")
        logger.info(f"Size: {request.width}x{request.height}, Steps: {request.num_inference_steps}")

        # Generate image
        result = pipe(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=torch.Generator().manual_seed(42)  # Fixed seed for consistency
        )

        image = result.images[0]

        # Convert to PNG bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        logger.info("✅ Image generated successfully")

        return Response(content=img_byte_arr.getvalue(), media_type="image/png")

    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/unload-lora")
async def unload_lora():
    """Unload currently loaded LORA"""
    global pipe, current_lora

    if current_lora is None:
        return {"status": "no_lora_loaded"}

    try:
        pipe.unload_lora_weights()
        current_lora = None
        logger.info("✅ LORA unloaded")
        return {"status": "success"}

    except Exception as e:
        logger.error(f"❌ Failed to unload LORA: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all interfaces
        port=7860,
        log_level="info"
    )
