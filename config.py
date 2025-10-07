import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
    RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST", "api-nba-v1.p.rapidapi.com")
    DEFAULT_SEASON = os.getenv("DEFAULT_SEASON", "2025")
    CACHE_TYPE = os.getenv("CACHE_TYPE", "SimpleCache")
    CACHE_REDIS_URL = os.getenv("CACHE_REDIS_URL", "")
    JSON_SORT_KEYS = False
