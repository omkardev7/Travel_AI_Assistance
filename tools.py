from crewai_tools import EXASearchTool
from config import settings
from logger import setup_logger

logger = setup_logger(__name__)

def get_exa_tool() -> EXASearchTool:

    logger.info("Initializing EXA Search Tool")
    
    try:
        tool = EXASearchTool(
            api_key=settings.EXA_API_KEY,
            content=settings.EXA_CONTENT,      # Fetch full content
            summary=settings.EXA_SUMMARY,      # Get AI summaries (REQUIRED)
            type=settings.EXA_TYPE             # Auto-detect search type
        )
        logger.info("EXA Search Tool initialized successfully")
        return tool
    except Exception as e:
        logger.error(f"Failed to initialize EXA Search Tool: {str(e)}")
        raise


exa_tool = get_exa_tool()