# Flux 2 Multi-Model API Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![GitHub](https://img.shields.io/badge/GitHub-xyster3k%2FFluxAPI-black.svg)](https://github.com/xyster3k/FluxAPI)

Local REST API server for **FLUX.2** image generation with support for multiple models. Designed for integration with Telegram bots, web apps, and other applications.

## Web Client

Need a simple UI to generate images? Check out [FluxAPIWebClient](https://github.com/xyster3k/FluxAPIWebClient) - a ready-to-use web client for this API.

## Available Models

| Model | Parameters | VRAM | Steps | License | Description |
|-------|------------|------|-------|---------|-------------|
| **klein-4b** | 4B | ~13GB | 4 | Apache 2.0 | Fastest, commercial use allowed |
| **klein-9b** | 9B + 8B encoder | ~29GB | 4 | Non-commercial | Higher quality, requires RTX 5090 |
| **dev** | 12B + 24B encoder | ~24GB | 28 | Non-commercial | Highest quality, uses FP8 |

> **Warning:** The **klein-9b** and **dev** models are licensed for **non-commercial use only**. They are included for demonstration and personal use purposes only. For commercial applications, use **klein-4b** (Apache 2.0 license).

## Requirements

- Windows 10/11
- Python 3.11 ([Download](https://www.python.org/downloads/release/python-3119/))
- NVIDIA GPU with **13GB+ VRAM** (RTX 3090, RTX 4070+, RTX 5090)
- NVIDIA drivers installed
- ~50GB disk space for all model files (cached in `%USERPROFILE%\.cache\huggingface`)

## Quick Start (Recommended)

### 1. Clone the repository

```bash
git clone https://github.com/xyster3k/FluxAPI.git
cd FluxAPI
```

### 2. Run setup

```bash
setup.bat
```

This automatically:
- Detects your GPU (Blackwell vs Ampere/Ada)
- Installs correct PyTorch version with CUDA
- Installs all dependencies
- Installs diffusers from git (required for Flux 2)
- Creates `.env` from template

### 3. Configure your API key

Edit `.env` file:
```env
FLUX_API_KEY=your-secret-api-key-here
```

Generate a secure key:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 4. Start the server

```bash
start.bat
```

**First startup will automatically download the default model (~10GB).** This happens once and models are cached for future use.

---

## Manual Installation

If `setup.bat` doesn't work for your setup:

### 1. Install Python 3.11

Download from [python.org](https://www.python.org/downloads/release/python-3119/). Check "Add Python to PATH" during installation.

### 2. Install PyTorch with CUDA

For **RTX 5090** (Blackwell):
```bash
py -3.11 -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

For **RTX 3090/4090** (Ampere/Ada):
```bash
py -3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install dependencies

```bash
py -3.11 -m pip install git+https://github.com/huggingface/diffusers
py -3.11 -m pip install -r requirements.txt
```

### 4. Configure environment

Copy `.env.example` to `.env` and edit:

```env
# API Key for authentication (generate a secure key)
FLUX_API_KEY=your-secret-api-key-here

# Server settings
FLUX_PORT=7860
FLUX_HOST=0.0.0.0

# GPU selection (0=primary GPU)
FLUX_GPU_ID=0

# Maximum requests in queue
FLUX_MAX_QUEUE=10

# Output directory for saved images
FLUX_OUTPUT_DIR=./generated

# LoRA directory (optional - allows using just filename)
FLUX_LORA_DIR=
```

### 5. Start the server

```bash
py -3.11 server.py
```

---

## Model Downloads

Models are **automatically downloaded** from Hugging Face on first use:

| Model | Size | Download Time* |
|-------|------|---------------|
| klein-4b | ~10GB | ~5 min |
| klein-9b | ~20GB | ~10 min |
| dev | ~25GB | ~12 min |

*Times based on 100 Mbps connection

Models are cached in `%USERPROFILE%\.cache\huggingface\hub` and won't be re-downloaded.

To pre-download all models, start the server and switch to each model:
```powershell
# After server starts, call switch-model for each
Invoke-RestMethod -Uri "http://localhost:7860/switch-model" -Method POST -Headers @{"X-API-Key"="your-key"; "Content-Type"="application/json"} -Body '{"model_id":"klein-9b"}'
Invoke-RestMethod -Uri "http://localhost:7860/switch-model" -Method POST -Headers @{"X-API-Key"="your-key"; "Content-Type"="application/json"} -Body '{"model_id":"dev"}'
```

---

## API Endpoints

### Health Check
```
GET http://localhost:7860/
```
Returns server status, GPU info, current model, and default settings. No authentication required.

### List Available Models
```
GET http://localhost:7860/models
```
Returns list of available models with their specifications. No authentication required.

Response:
```json
{
  "models": {
    "klein-4b": {
      "name": "FLUX.2-klein-4B",
      "license": "Apache 2.0 (commercial OK)",
      "vram_gb": 13,
      "default_steps": 4,
      "default_guidance": 1.0,
      "description": "Fastest model, commercial use allowed"
    },
    "klein-9b": {...},
    "dev": {...}
  },
  "current_model": "klein-4b"
}
```

### Switch Model
```
POST http://localhost:7860/switch-model
Header: X-API-Key: your-api-key
Content-Type: application/json
Body: {"model_id": "dev"}
```
Unloads current model and loads the specified model. Returns new model info and updated defaults.

### Generate Image
```
POST http://localhost:7860/generate
Header: X-API-Key: your-api-key
Content-Type: application/json
```

### Queue Status
```
GET http://localhost:7860/queue/status
```
Returns current queue size and current model. No authentication required.

### Unload LoRA
```
POST http://localhost:7860/unload-lora
Header: X-API-Key: your-api-key
```

### Switch GPU
```
POST http://localhost:7860/switch-gpu
Header: X-API-Key: your-api-key
Content-Type: application/json
Body: {"gpu_id": 1}
```

---

## Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description of the image |
| `aspect_ratio` | string | `"9:16"` | Aspect ratio: `9:16`, `2:3`, `3:4`, `1:1`, `4:3`, `3:2`, `16:9`, `custom` |
| `base_resolution` | int | `576` | Base resolution (shorter side, 256-2048) |
| `width` | int | null | Custom width (only when aspect_ratio=`custom`) |
| `height` | int | null | Custom height (only when aspect_ratio=`custom`) |
| `num_inference_steps` | int | model default | Denoising steps (Klein: 4, Dev: 28) |
| `guidance_scale` | float | model default | CFG scale (Klein: 1.0, Dev: 3.5) |
| `seed` | int | null | Random seed (null = random) |
| `lora_path` | string | null | LoRA filename or full path (**must be Klein 4B LoRAs!**) |
| `lora_scale` | float | `1.0` | LoRA strength (0.0-2.0) |
| `response_format` | string | `"base64"` | Response type: `binary`, `base64`, `file` |

> **LoRA Warning:** Flux Schnell LoRAs are NOT compatible with Klein 4B. You must train new LoRAs specifically for the Klein 4B architecture.

---

## Response Formats

### Binary (default)
Returns raw PNG bytes. Metadata in headers:
- `X-Seed`: Generation seed
- `X-Generation-Time-Ms`: Time in milliseconds
- `X-Width`: Image width
- `X-Height`: Image height

### Base64
Returns JSON:
```json
{
  "image": "base64-encoded-png-data",
  "seed": 1234567890,
  "generation_time_ms": 15000,
  "width": 576,
  "height": 1024,
  "aspect_ratio": "9:16",
  "num_steps": 4,
  "model": "klein-4b",
  "model_name": "FLUX.2-klein-4B"
}
```

### File
Saves to disk and returns JSON:
```json
{
  "file_path": "D:\\FluxAPI\\generated\\flux_1234567890_1705500000.png",
  "seed": 1234567890,
  "generation_time_ms": 15000,
  "width": 576,
  "height": 1024,
  "aspect_ratio": "9:16",
  "num_steps": 4,
  "model": "klein-4b",
  "model_name": "FLUX.2-klein-4B"
}
```

---

## Python Integration (for Telegram Bot)

### Basic Example

```python
import requests
import base64
from io import BytesIO

API_URL = "http://localhost:7860/generate"
API_KEY = "your-api-key-here"

def generate_image(prompt: str, lora: str = None, lora_scale: float = 0.8) -> tuple[bytes, int]:
    """
    Generate an image and return (image_bytes, seed)
    """
    payload = {
        "prompt": prompt,
        "response_format": "base64"
    }

    if lora:
        payload["lora_path"] = lora
        payload["lora_scale"] = lora_scale

    response = requests.post(
        API_URL,
        json=payload,
        headers={
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        },
        timeout=120  # Generation can take up to 2 minutes
    )

    response.raise_for_status()
    data = response.json()

    image_bytes = base64.b64decode(data["image"])
    seed = data["seed"]

    return image_bytes, seed


# Usage
image_bytes, seed = generate_image(
    prompt="a beautiful sunset over mountains",
    lora="pinterestshort1",
    lora_scale=0.8
)
print(f"Generated with seed: {seed}")
```

### Telegram Bot Example

```python
import requests
import base64
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

API_URL = "http://localhost:7860/generate"
API_KEY = "your-api-key-here"
BOT_TOKEN = "your-telegram-bot-token"

async def generate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /generate command"""
    if not context.args:
        await update.message.reply_text("Usage: /generate <prompt>")
        return

    prompt = " ".join(context.args)
    await update.message.reply_text(f"Generating: {prompt}...")

    try:
        response = requests.post(
            API_URL,
            json={
                "prompt": prompt,
                "lora_path": "pinterestshort1",
                "lora_scale": 0.8,
                "response_format": "base64"
            },
            headers={
                "X-API-Key": API_KEY,
                "Content-Type": "application/json"
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()

        image_bytes = base64.b64decode(data["image"])

        await update.message.reply_photo(
            photo=image_bytes,
            caption=f"Model: {data['model_name']}\nSeed: {data['seed']}\nSize: {data['width']}x{data['height']}\nTime: {data['generation_time_ms']}ms"
        )

    except requests.exceptions.RequestException as e:
        await update.message.reply_text(f"Error: {str(e)}")

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("generate", generate_command))
    app.run_polling()

if __name__ == "__main__":
    main()
```

### With Seed Reproducibility

```python
def generate_with_variations(prompt: str, base_seed: int = None):
    """Generate image and allow variations with same seed"""

    # First generation (random seed)
    response = requests.post(API_URL, json={
        "prompt": prompt,
        "seed": base_seed,  # None = random
        "response_format": "base64"
    }, headers={"X-API-Key": API_KEY, "Content-Type": "application/json"})

    data = response.json()
    seed = data["seed"]

    print(f"Original seed: {seed}")

    # Reproduce exact same image
    response2 = requests.post(API_URL, json={
        "prompt": prompt,
        "seed": seed,  # Same seed = same image
        "response_format": "base64"
    }, headers={"X-API-Key": API_KEY, "Content-Type": "application/json"})

    # Images will be identical
    return data, seed
```

### Async Version (for aiogram/telegram bots)

```python
import aiohttp
import base64

async def generate_image_async(prompt: str, lora: str = None) -> tuple[bytes, dict]:
    """Async image generation"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "prompt": prompt,
            "response_format": "base64"
        }
        if lora:
            payload["lora_path"] = lora
            payload["lora_scale"] = 0.8

        async with session.post(
            API_URL,
            json=payload,
            headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            response.raise_for_status()
            data = await response.json()
            image_bytes = base64.b64decode(data["image"])
            return image_bytes, data
```

---

## Model Switching Examples

### List Available Models (PowerShell)
```powershell
Invoke-RestMethod -Uri "http://localhost:7860/models"
```

### Switch to Dev Model (PowerShell)
```powershell
$body = @{ model_id = "dev" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:7860/switch-model" -Method POST -Headers @{"X-API-Key"="your-api-key"; "Content-Type"="application/json"} -Body $body
```

### Switch Model (Python)
```python
import requests

API_URL = "http://localhost:7860"
API_KEY = "your-api-key-here"

def switch_model(model_id: str) -> dict:
    """Switch to a different model (klein-4b, klein-9b, or dev)"""
    response = requests.post(
        f"{API_URL}/switch-model",
        json={"model_id": model_id},
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
        timeout=120  # Model loading takes time
    )
    response.raise_for_status()
    return response.json()

# Switch to highest quality model
result = switch_model("dev")
print(f"Switched to: {result['model_name']}")
print(f"Default steps: {result['default_steps']}")
print(f"Default guidance: {result['default_guidance']}")
```

### Get Available Models (Python)
```python
def list_models() -> dict:
    """Get list of available models and current model"""
    response = requests.get(f"{API_URL}/models")
    return response.json()

models = list_models()
print(f"Current model: {models['current_model']}")
for model_id, info in models['models'].items():
    print(f"  {model_id}: {info['description']} ({info['vram_gb']}GB VRAM)")
```

---

## PowerShell Testing

### Basic generation:
```powershell
$body = @{
    prompt = "a beautiful woman in a coffee shop"
    lora_path = "pinterestshort1"
    lora_scale = 0.8
    response_format = "file"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:7860/generate" -Method POST -Headers @{"X-API-Key"="your-api-key"; "Content-Type"="application/json"} -Body $body
```

### With all parameters:
```powershell
$body = @{
    prompt = "a beautiful sunset over mountains"
    aspect_ratio = "16:9"
    base_resolution = 720
    num_inference_steps = 20
    seed = 12345
    lora_path = "pinterestshort1"
    lora_scale = 0.8
    response_format = "file"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:7860/generate" -Method POST -Headers @{"X-API-Key"="your-api-key"; "Content-Type"="application/json"} -Body $body
```

---

## Troubleshooting

### CUDA Out of Memory
- Klein 4B: ~13GB VRAM
- Klein 9B: ~29GB VRAM (requires RTX 5090 32GB)
- Dev: ~24GB VRAM with FP8 quantization
- The server uses CPU offload to manage VRAM
- Close other GPU applications if issues persist
- If switching models fails, try restarting the server

### Model Switching Takes Time
- First model load downloads the model (~10-15GB each)
- Switching models unloads previous model and loads new one
- This can take 30-60 seconds depending on model size
- Check `/queue/status` to see current model

### Slow First Generation
- First request after startup or model switch warms up the model
- Klein models: subsequent requests should be **sub-second** with 4 steps
- Dev model: expect 20-30 seconds with 28 steps

### LoRA Not Working
- **Schnell LoRAs are NOT compatible with Flux 2 models!**
- You must train new LoRAs specifically for each architecture
- Use `FLUX.2-klein-base-4B` as base model for Klein LoRAs
- Check `FLUX_LORA_DIR` in `.env` is correct

### Connection Refused
- Ensure server is running (`start.bat`)
- Check firewall settings
- Verify port 7860 is not in use

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Attribution required:** When using this software, please include attribution to the original author.

### Model Licenses
- **FLUX.2-klein-4B**: Apache 2.0 (commercial use allowed)
- **FLUX.2-klein-9B**: Non-commercial research only
- **FLUX.2-dev**: Non-commercial research only
- **LoRAs**: Check individual LoRA licenses

---

## Migration from Schnell

If you were using Flux Schnell before:

1. **LoRAs are NOT compatible** - You need to retrain them for Flux 2 models
2. **Install diffusers from git** (required for Flux2KleinPipeline):
   ```bash
   py -3.11 -m pip install git+https://github.com/huggingface/diffusers
   ```
3. **Default model is Klein 4B** - Fastest option with commercial license
4. **Switch models via API** - Use `/switch-model` endpoint to change models
5. **Each model has different defaults**:
   - Klein 4B/9B: 4 steps, guidance 1.0
   - Dev: 28 steps, guidance 3.5

## Model Recommendations

| Use Case | Recommended Model |
|----------|-------------------|
| Commercial production | **klein-4b** (Apache 2.0) |
| Fast prototyping | **klein-4b** (sub-second) |
| Higher quality (personal) | **klein-9b** or **dev** |
| Best quality (personal) | **dev** |
| Limited VRAM (13-24GB) | **klein-4b** |
| RTX 5090 (32GB) | Any model |

---

## Project Structure

```
FluxAPI/
├── server.py          # Main API server
├── config.py          # Configuration management
├── setup.bat          # One-click setup script
├── start.bat          # Server launcher
├── requirements.txt   # Python dependencies
├── .env.example       # Environment template
├── .env               # Your configuration (not in repo)
├── .gitignore         # Git ignore rules
├── LICENSE            # MIT License
└── README.md          # This file
```

---

## Credits

- **FLUX.2 Models**: [Black Forest Labs](https://blackforestlabs.ai/)
- **Diffusers Library**: [Hugging Face](https://huggingface.co/docs/diffusers)

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Author

Created by [Xyster3k](https://github.com/xyster3k)
