# Telegram Bot Integration Guide

How to integrate the Flux API with your Telegram bot.

## Prerequisites

- Flux API server running on `http://localhost:7860`
- Telegram Bot Token (from [@BotFather](https://t.me/BotFather))
- Python 3.11+

## Installation

```bash
pip install python-telegram-bot aiohttp
```

Or for synchronous version:
```bash
pip install pyTelegramBotAPI requests
```

---

## Option 1: python-telegram-bot (Async, Recommended)

```python
import base64
import aiohttp
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Configuration
BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
FLUX_API_URL = "http://localhost:7860/generate"
FLUX_API_KEY = "your-flux-api-key"

# Default LoRA settings
DEFAULT_LORA = "pinterestshort1"
DEFAULT_LORA_SCALE = 0.8


async def generate_image(prompt: str, seed: int = None, lora: str = None, lora_scale: float = 0.8) -> dict:
    """Call Flux API to generate image"""
    payload = {
        "prompt": prompt,
        "response_format": "base64"
    }

    if seed is not None:
        payload["seed"] = seed

    if lora:
        payload["lora_path"] = lora
        payload["lora_scale"] = lora_scale

    async with aiohttp.ClientSession() as session:
        async with session.post(
            FLUX_API_URL,
            json=payload,
            headers={
                "X-API-Key": FLUX_API_KEY,
                "Content-Type": "application/json"
            },
            timeout=aiohttp.ClientTimeout(total=180)
        ) as response:
            if response.status != 200:
                error = await response.text()
                raise Exception(f"API Error {response.status}: {error}")

            return await response.json()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    await update.message.reply_text(
        "Welcome to Flux Image Generator!\n\n"
        "Commands:\n"
        "/generate <prompt> - Generate an image\n"
        "/regenerate - Regenerate with same seed\n"
        "/help - Show help"
    )


async def generate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /generate command"""
    if not context.args:
        await update.message.reply_text("Usage: /generate <your prompt here>")
        return

    prompt = " ".join(context.args)
    status_msg = await update.message.reply_text(f"Generating: {prompt}...")

    try:
        result = await generate_image(
            prompt=prompt,
            lora=DEFAULT_LORA,
            lora_scale=DEFAULT_LORA_SCALE
        )

        image_bytes = base64.b64decode(result["image"])

        # Store seed for regeneration
        context.user_data["last_seed"] = result["seed"]
        context.user_data["last_prompt"] = prompt

        await update.message.reply_photo(
            photo=image_bytes,
            caption=(
                f"Seed: `{result['seed']}`\n"
                f"Size: {result['width']}x{result['height']}\n"
                f"Time: {result['generation_time_ms']}ms"
            ),
            parse_mode="Markdown"
        )

        await status_msg.delete()

    except Exception as e:
        await status_msg.edit_text(f"Error: {str(e)}")


async def regenerate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Regenerate last image with same seed"""
    seed = context.user_data.get("last_seed")
    prompt = context.user_data.get("last_prompt")

    if not seed or not prompt:
        await update.message.reply_text("No previous generation to regenerate. Use /generate first.")
        return

    status_msg = await update.message.reply_text(f"Regenerating with seed {seed}...")

    try:
        result = await generate_image(
            prompt=prompt,
            seed=seed,
            lora=DEFAULT_LORA,
            lora_scale=DEFAULT_LORA_SCALE
        )

        image_bytes = base64.b64decode(result["image"])

        await update.message.reply_photo(
            photo=image_bytes,
            caption=f"Regenerated with seed: `{result['seed']}`",
            parse_mode="Markdown"
        )

        await status_msg.delete()

    except Exception as e:
        await status_msg.edit_text(f"Error: {str(e)}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    await update.message.reply_text(
        "*Flux Image Generator*\n\n"
        "*Commands:*\n"
        "`/generate <prompt>` - Generate image from text\n"
        "`/regenerate` - Regenerate last image (same seed)\n\n"
        "*Examples:*\n"
        "`/generate a beautiful sunset over mountains`\n"
        "`/generate portrait of a woman, natural lighting`\n\n"
        "*Settings:*\n"
        f"LoRA: {DEFAULT_LORA}\n"
        f"LoRA Scale: {DEFAULT_LORA_SCALE}\n"
        "Resolution: 576x1024 (9:16)\n"
        "Steps: 20",
        parse_mode="Markdown"
    )


def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("generate", generate))
    app.add_handler(CommandHandler("regenerate", regenerate))
    app.add_handler(CommandHandler("help", help_command))

    print("Bot started!")
    app.run_polling()


if __name__ == "__main__":
    main()
```

---

## Option 2: pyTelegramBotAPI (Sync, Simpler)

```python
import base64
import requests
import telebot

# Configuration
BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
FLUX_API_URL = "http://localhost:7860/generate"
FLUX_API_KEY = "your-flux-api-key"

bot = telebot.TeleBot(BOT_TOKEN)

# Store last generation for regeneration
user_data = {}


def generate_image(prompt: str, seed: int = None, lora: str = "pinterestshort1") -> dict:
    """Call Flux API"""
    payload = {
        "prompt": prompt,
        "lora_path": lora,
        "lora_scale": 0.8,
        "response_format": "base64"
    }

    if seed is not None:
        payload["seed"] = seed

    response = requests.post(
        FLUX_API_URL,
        json=payload,
        headers={
            "X-API-Key": FLUX_API_KEY,
            "Content-Type": "application/json"
        },
        timeout=180
    )

    response.raise_for_status()
    return response.json()


@bot.message_handler(commands=["start"])
def start(message):
    bot.reply_to(message,
        "Welcome! Use /generate <prompt> to create images.\n"
        "Example: /generate a beautiful sunset"
    )


@bot.message_handler(commands=["generate"])
def generate(message):
    prompt = message.text.replace("/generate", "").strip()

    if not prompt:
        bot.reply_to(message, "Usage: /generate <your prompt>")
        return

    status = bot.reply_to(message, f"Generating: {prompt}...")

    try:
        result = generate_image(prompt)

        image_bytes = base64.b64decode(result["image"])

        # Save for regeneration
        user_data[message.from_user.id] = {
            "seed": result["seed"],
            "prompt": prompt
        }

        bot.send_photo(
            message.chat.id,
            photo=image_bytes,
            caption=f"Seed: {result['seed']}\nTime: {result['generation_time_ms']}ms"
        )

        bot.delete_message(message.chat.id, status.message_id)

    except Exception as e:
        bot.edit_message_text(f"Error: {e}", message.chat.id, status.message_id)


@bot.message_handler(commands=["regenerate"])
def regenerate(message):
    data = user_data.get(message.from_user.id)

    if not data:
        bot.reply_to(message, "No previous generation. Use /generate first.")
        return

    status = bot.reply_to(message, f"Regenerating with seed {data['seed']}...")

    try:
        result = generate_image(data["prompt"], seed=data["seed"])
        image_bytes = base64.b64decode(result["image"])

        bot.send_photo(
            message.chat.id,
            photo=image_bytes,
            caption=f"Regenerated with seed: {result['seed']}"
        )

        bot.delete_message(message.chat.id, status.message_id)

    except Exception as e:
        bot.edit_message_text(f"Error: {e}", message.chat.id, status.message_id)


print("Bot started!")
bot.polling()
```

---

## Advanced: Full-Featured Bot

```python
import base64
import aiohttp
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
FLUX_API_URL = "http://localhost:7860/generate"
FLUX_API_KEY = "your-flux-api-key"


async def generate_image(
    prompt: str,
    seed: int = None,
    lora: str = "pinterestshort1",
    lora_scale: float = 0.8,
    aspect_ratio: str = "9:16",
    steps: int = 20
) -> dict:
    """Generate image with all options"""
    payload = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "num_inference_steps": steps,
        "response_format": "base64"
    }

    if seed is not None:
        payload["seed"] = seed

    if lora:
        payload["lora_path"] = lora
        payload["lora_scale"] = lora_scale

    async with aiohttp.ClientSession() as session:
        async with session.post(
            FLUX_API_URL,
            json=payload,
            headers={"X-API-Key": FLUX_API_KEY, "Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=180)
        ) as response:
            if response.status != 200:
                raise Exception(f"API Error: {await response.text()}")
            return await response.json()


async def generate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate with aspect ratio buttons"""
    if not context.args:
        await update.message.reply_text("Usage: /generate <prompt>")
        return

    prompt = " ".join(context.args)
    context.user_data["pending_prompt"] = prompt

    keyboard = [
        [
            InlineKeyboardButton("9:16 Portrait", callback_data="ar_9:16"),
            InlineKeyboardButton("1:1 Square", callback_data="ar_1:1"),
        ],
        [
            InlineKeyboardButton("16:9 Landscape", callback_data="ar_16:9"),
            InlineKeyboardButton("3:4 Portrait", callback_data="ar_3:4"),
        ]
    ]

    await update.message.reply_text(
        f"Prompt: {prompt}\n\nSelect aspect ratio:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def aspect_ratio_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle aspect ratio selection"""
    query = update.callback_query
    await query.answer()

    aspect_ratio = query.data.replace("ar_", "")
    prompt = context.user_data.get("pending_prompt", "")

    if not prompt:
        await query.edit_message_text("Session expired. Use /generate again.")
        return

    await query.edit_message_text(f"Generating ({aspect_ratio}):\n{prompt}")

    try:
        result = await generate_image(prompt, aspect_ratio=aspect_ratio)
        image_bytes = base64.b64decode(result["image"])

        context.user_data["last_seed"] = result["seed"]
        context.user_data["last_prompt"] = prompt
        context.user_data["last_aspect"] = aspect_ratio

        keyboard = [[
            InlineKeyboardButton("Regenerate", callback_data="regen"),
            InlineKeyboardButton("Vary Seed", callback_data="vary"),
        ]]

        await query.message.reply_photo(
            photo=image_bytes,
            caption=f"Seed: `{result['seed']}`\n{result['width']}x{result['height']}",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

        await query.delete_message()

    except Exception as e:
        await query.edit_message_text(f"Error: {e}")


async def action_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regenerate/vary buttons"""
    query = update.callback_query
    await query.answer()

    action = query.data
    prompt = context.user_data.get("last_prompt")
    seed = context.user_data.get("last_seed")
    aspect = context.user_data.get("last_aspect", "9:16")

    if not prompt:
        await query.message.reply_text("No previous generation.")
        return

    if action == "vary":
        seed = None  # New random seed

    status = await query.message.reply_text(
        f"{'Regenerating' if action == 'regen' else 'Creating variation'}..."
    )

    try:
        result = await generate_image(prompt, seed=seed, aspect_ratio=aspect)
        image_bytes = base64.b64decode(result["image"])

        context.user_data["last_seed"] = result["seed"]

        keyboard = [[
            InlineKeyboardButton("Regenerate", callback_data="regen"),
            InlineKeyboardButton("Vary Seed", callback_data="vary"),
        ]]

        await query.message.reply_photo(
            photo=image_bytes,
            caption=f"Seed: `{result['seed']}`",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

        await status.delete()

    except Exception as e:
        await status.edit_text(f"Error: {e}")


def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("generate", generate))
    app.add_handler(CallbackQueryHandler(aspect_ratio_callback, pattern="^ar_"))
    app.add_handler(CallbackQueryHandler(action_callback, pattern="^(regen|vary)$"))

    print("Bot started!")
    app.run_polling()


if __name__ == "__main__":
    main()
```

---

## Configuration Tips

### 1. Store config in environment variables

```python
import os

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
FLUX_API_KEY = os.getenv("FLUX_API_KEY")
FLUX_API_URL = os.getenv("FLUX_API_URL", "http://localhost:7860/generate")
```

### 2. Error handling with retry

```python
import asyncio

async def generate_with_retry(prompt: str, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            return await generate_image(prompt)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### 3. Queue status check

```python
async def check_queue() -> int:
    """Check API queue before generating"""
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:7860/queue/status") as resp:
            data = await resp.json()
            return data["queue_size"]

# Usage
queue_size = await check_queue()
if queue_size >= 5:
    await update.message.reply_text(f"Server busy ({queue_size} in queue). Please wait...")
```

---

## Running the Bot

1. Create bot with [@BotFather](https://t.me/BotFather)
2. Copy the token
3. Update `BOT_TOKEN` in your script
4. Update `FLUX_API_KEY` with your API key
5. Ensure Flux API server is running
6. Run the bot:

```bash
py -3.11 telegram_bot.py
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Timeout errors | Increase timeout (generation takes 30-120s) |
| Connection refused | Check Flux API is running on port 7860 |
| 401 Unauthorized | Verify FLUX_API_KEY matches server .env |
| 503 Queue full | Wait and retry, or increase FLUX_MAX_QUEUE |
