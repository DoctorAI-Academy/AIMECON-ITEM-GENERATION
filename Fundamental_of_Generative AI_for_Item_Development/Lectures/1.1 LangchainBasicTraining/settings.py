from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    openai_api_key: str
    anthropic_api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    
    class Config:
        env_file = ".env"

settings = Settings()