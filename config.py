import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta


load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    EXA_API_KEY: str = os.getenv("EXA_API_KEY", "")
    
    # FastAPI Settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # CrewAI Settings - DISABLE BUILT-IN MEMORY
    CREW_MEMORY: bool = False  # Changed from True to False
    CREW_VERBOSE: bool = True  # Changed from True to False to reduce output
    CREWAI_TRACING_ENABLED: bool = False
    
    # LLM Settings - Use correct Gemini model format
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_TEMPERATURE: float = 0.7
    
    # EXA Settings
    EXA_CONTENT: bool = True
    EXA_SUMMARY: bool = True
    EXA_TYPE: str = "auto"
    
    # Session Settings
    SESSION_TIMEOUT: int = 3600
    MAX_CONTEXT_MESSAGES: int = 10
    
    # Custom Memory Settings
    MEMORY_DB_PATH: str = os.getenv("MEMORY_DB_PATH", "travel_memory.db")
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

# Initialize settings
settings = Settings()

# Setup logger AFTER settings are initialized
from logger import setup_logger
logger = setup_logger(__name__, settings.LOG_LEVEL)

logger.info("Environment variables loaded")

# Validate required API keys
def validate_api_keys():
    """Validate that required API keys are present"""
    missing_configs = []
    
    if not settings.GEMINI_API_KEY:
        missing_configs.append("GEMINI_API_KEY")
        
    if not settings.EXA_API_KEY:
        missing_configs.append("EXA_API_KEY")
    
    if missing_configs:
        error_msg = f"Missing required environment variables: {', '.join(missing_configs)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("All required API keys validated successfully")

# Run validation on import
try:
    validate_api_keys()
    # Set Gemini API key and disable tracing
    os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY
    os.environ["CREWAI_TRACING_ENABLED"] = "false"
    
    # CRITICAL: Prevent OpenAI embedding function from being loaded
    os.environ["CREWAI_STORAGE_DIR"] = ""
    
    logger.info("GEMINI_API_KEY set in environment")
    logger.info("CREWAI_TRACING_ENABLED set to false")
    logger.info("Custom SQLite memory will be used")
except ValueError as e:
    logger.warning(f"Configuration validation failed: {e}")
    logger.warning("Application may not function correctly without proper configuration")
    
def get_date_context():
    """Generate current date context for agents"""
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    day_after = today + timedelta(days=2)
    next_week = today + timedelta(days=7)

    return f"""
CURRENT DATE CONTEXT (USE THIS FOR ALL DATE INTERPRETATIONS):
- TODAY: {today.strftime('%A, %d %B %Y')} (use format: {today.strftime('%Y-%m-%d')})
- TOMORROW: {tomorrow.strftime('%A, %d %B %Y')} (use format: {tomorrow.strftime('%Y-%m-%d')})
- DAY AFTER TOMORROW: {day_after.strftime('%A, %d %B %Y')} (use format: {day_after.strftime('%Y-%m-%d')})
- NEXT WEEK: {next_week.strftime('%A, %d %B %Y')} (use format: {next_week.strftime('%Y-%m-%d')})
- CURRENT TIME: {today.strftime('%H:%M')} IST

"""