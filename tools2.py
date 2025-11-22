import os
from typing import Type, Any, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from exa_py import Exa
from config import settings
from logger import setup_logger

logger = setup_logger(__name__)

# ==============================================================================
# 1. INPUT SCHEMA
# ==============================================================================
class ExaSearchInput(BaseModel):
    search_query: str = Field(..., description="The specific query to search for travel information.")

# ==============================================================================
# 2. CUSTOM ENHANCED TOOL
# ==============================================================================
class OptimizedEXASearchTool(BaseTool):
    name: str = "Optimized Travel Search"
    description: str = (
        "A robust search tool for flights, hotels, and trains. "
        "Returns AI summaries AND detailed page content to find prices/numbers. "
        "Optimized for finding specific travel details."
    )
    args_schema: Type[BaseModel] = ExaSearchInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    client: Any = None

    def __init__(self, **data):
        super().__init__(**data)
        if not settings.EXA_API_KEY:
            raise ValueError("EXA_API_KEY not found in settings config!")
        self.client = Exa(api_key=settings.EXA_API_KEY)

    def _run(self, search_query: str) -> str:
        logger.info(f"🔍 Executing Deep Search for: '{search_query}'")
        
        # TRICK 1: Enhance the query automatically
        # If the agent asks "flights Mumbai to Delhi", we add "schedule price table"
        # to help Exa find data-rich pages instead of generic blogs.
        enhanced_query = search_query
        if "flight" in search_query.lower() or "train" in search_query.lower():
             if "price" not in search_query.lower():
                 enhanced_query += " price schedule ticket table"
        
        try:
            response = self.client.search_and_contents(
                enhanced_query,
                type="neural",
                num_results=3,          # Top 3 is enough if they are high quality
                summary=True,           # Get the AI summary
                text={
                    # TRICK 2: INCREASE CONTENT LIMIT
                    # 1000 was too short (mostly headers). 
                    # 5000 catches the actual flight tables and prices.
                    "max_characters": 5000, 
                    "include_html_tags": False
                }
            )
            return self._parse_results(response)

        except Exception as e:
            logger.error(f"Search Error: {str(e)}")
            return f"Error: {str(e)}"

    def _parse_results(self, response: Any) -> str:
        if not response or not response.results:
            return "No results found."

        formatted_output = []
        
        for idx, result in enumerate(response.results, 1):
            title = getattr(result, 'title', 'No Title')
            url = getattr(result, 'url', '#')
            summary = getattr(result, 'summary', 'No summary available.')
            
            # TRICK 3: RETURN MORE RAW TEXT
            # We give the agent the first 2000 chars of text.
            # This allows the LLM to scan the raw table data for numbers
            # that the summary might have missed.
            full_text = getattr(result, 'text', '')
            text_preview = full_text[:2000] if full_text else "No text content."
            
            # Clean up newlines to make it dense
            text_preview = text_preview.replace('\n\n', '\n').strip()

            entry = (
                f"=== OPTION {idx} ===\n"
                f"SOURCE: {title}\n"
                f"LINK: {url}\n"
                f"----------------------------------------\n"
                f"🤖 AI SUMMARY: {summary}\n"
                f"----------------------------------------\n"
                f"📄 PAGE CONTENT (Raw Data):\n{text_preview}\n"
                f"========================================\n"
            )
            formatted_output.append(entry)

        return "\n".join(formatted_output)

# ==============================================================================
# 3. FACTORY FUNCTION
# ==============================================================================
def get_exa_tool():
    return OptimizedEXASearchTool()

exa_tool = get_exa_tool()

# ==============================================================================
# 4. TEST
# ==============================================================================
if __name__ == "__main__":
    print("\n🧪 TESTING DEEP SEARCH...\n")
    # Note: We are searching for specific "schedule" info now
    tool = get_exa_tool()
    print(tool._run("flights from Mumbai to Delhi schedule and price"))