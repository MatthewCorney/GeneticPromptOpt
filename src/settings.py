from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class Settings(BaseSettings):
    openai_api_key: str = ''
    debug: bool = False  # Default to False if not set
    model: str = "gpt-4o-mini"
    embedding_model : str = "text-embedding-3-small"

    class Config:
        # Tell pydantic-settings to look for the .env file
        env_file = "../.env"
        env_file_encoding = "utf-8"


settings = Settings()
client = OpenAI(api_key=settings.openai_api_key)


