"""
Configuration management for Flux API Server
Loads settings from environment variables
"""

import os
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


class Config:
    # API Authentication
    API_KEY: str = os.getenv("FLUX_API_KEY", "")

    # Server settings
    PORT: int = int(os.getenv("FLUX_PORT", "7860"))
    HOST: str = os.getenv("FLUX_HOST", "0.0.0.0")

    # GPU settings
    GPU_ID: int = int(os.getenv("FLUX_GPU_ID", "0"))  # 0=RTX 5090, 1=RTX 3090

    # Queue settings
    MAX_QUEUE_SIZE: int = int(os.getenv("FLUX_MAX_QUEUE", "10"))

    # Generation defaults
    DEFAULT_WIDTH: int = 1024
    DEFAULT_HEIGHT: int = 1024
    DEFAULT_STEPS: int = 4

    # Output directory for file response format
    OUTPUT_DIR: str = os.getenv("FLUX_OUTPUT_DIR", "./generated")

    @classmethod
    def require_auth(cls) -> bool:
        """Returns True if API key is configured"""
        return bool(cls.API_KEY)


config = Config()
