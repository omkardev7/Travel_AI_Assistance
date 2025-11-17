# crew.py
"""
CrewAI orchestration for Multi-Lingual Travel Assistant
Creates and manages agent crews WITHOUT CrewAI memory
"""

from crewai import Crew, Process
from agents import (
    language_agent,
    orchestrator_agent,
    flight_agent,
    hotel_agent,
    transport_agent,
    attractions_agent,
    response_agent,
    followup_agent
)
from tasks import (
    task_language_detection,
    task_orchestration,
    task_flight_search,
    task_hotel_search,
    task_transport_search,
    task_attractions_search,
    task_final_response,
    task_followup_response
)
from config import settings
from logger import setup_logger

logger = setup_logger(__name__)

def create_travel_crew(is_followup: bool = False, context_data: dict = None) -> Crew:
    """
    Create CrewAI crew based on query type
    
    FIRST ITERATION (is_followup=False):
        User Input → Agent 1 → Agent 2 → Agent 3 → Agent 4 → Response
        Full pipeline with all agents
    
    SECOND ITERATION (is_followup=True):
        User Input → Agent 5 (uses custom memory) → Response
        Only Agent 5 with context from SQLite
    
    Args:
        is_followup: Whether this is a follow-up question (default: False)
        context_data: Context data from custom memory (for follow-ups)
    
    Returns:
        Configured Crew instance
    """
    
    if is_followup:
        # ==================== FOLLOW-UP MODE ====================
        # Only Follow-up Agent with custom memory context
        logger.info("Creating crew for FOLLOW-UP query (Follow-up Agent only)")
        
        crew = Crew(
            agents=[followup_agent],
            tasks=[task_followup_response],
            process=Process.sequential,
            memory=False,  # DISABLED - using custom SQLite memory
            verbose=settings.CREW_VERBOSE,
            full_output=True
        )
        
        logger.info("Follow-up crew created successfully")
        return crew
    
    else:
        # ==================== INITIAL MODE ====================
        # Full pipeline: All agents
        logger.info("Creating crew for INITIAL query (Full pipeline)")
        
        crew = Crew(
            agents=[
                language_agent,       # Agent 1: Language detection & translation
                orchestrator_agent,   # Agent 2: Routing & orchestration
                flight_agent,         # Agent 3A: Flight search
                hotel_agent,          # Agent 3B: Hotel search
                transport_agent,      # Agent 3C: Train/Bus search
                attractions_agent,    # Agent 3D: Attractions search
                response_agent        # Agent 4: Final response translation
            ],
            tasks=[
                task_language_detection,   # Task 1: Detect language & translate
                task_orchestration,        # Task 2: Route to appropriate agent
                task_flight_search,        # Task 3A: Search flights (conditional)
                task_hotel_search,         # Task 3B: Search hotels (conditional)
                task_transport_search,     # Task 3C: Search transport (conditional)
                task_attractions_search,   # Task 3D: Search attractions (conditional)
                task_final_response        # Task 4: Translate & respond
            ],
            process=Process.sequential,  # Tasks run in order
            memory=False,                # DISABLED - using custom SQLite memory
            verbose=settings.CREW_VERBOSE,
            full_output=True
        )
        
        logger.info("Initial query crew created successfully")
        return crew

def kickoff_crew(crew: Crew, inputs: dict) -> str:
    """
    Execute crew with inputs and return final output
    
    Args:
        crew: Configured Crew instance
        inputs: Input dictionary for the crew
    
    Returns:
        Final output string from the crew
    """
    try:
        logger.info(f"Kicking off crew with inputs: {list(inputs.keys())}")
        result = crew.kickoff(inputs=inputs)
        
        # Extract the output properly
        if hasattr(result, 'raw'):
            output = str(result.raw)
        elif hasattr(result, 'output'):
            output = str(result.output)
        else:
            output = str(result)
        
        logger.info("Crew execution completed successfully")
        return output
    except Exception as e:
        logger.error(f"Error during crew execution: {str(e)}", exc_info=True)
        raise

# # Example usage for testing
# if __name__ == "__main__":
#     # Test initial query
#     print("Testing INITIAL query...")
#     initial_crew = create_travel_crew(is_followup=False)
    
#     test_input = {
#         "user_input": "I need a flight from Mumbai to Delhi tomorrow",
#         "session_id": "test_session_123"
#     }
    
#     try:
#         result = kickoff_crew(initial_crew, test_input)
#         print(f"\nInitial Query Result:\n{result}\n")
#     except Exception as e:
#         print(f"Error: {e}")
    
#     # Test follow-up query
#     print("\nTesting FOLLOW-UP query...")
    
#     # Simulate context from custom memory
#     context_data = {
#         "language": {"detected_language": "en", "language_name": "English"},
#         "search_results": [
#             {
#                 "service_type": "flight",
#                 "results": [
#                     {"airline": "IndiGo", "flight_number": "6E-123", "price": "₹3,500"}
#                 ]
#             }
#         ]
#     }
    
#     followup_crew = create_travel_crew(is_followup=True, context_data=context_data)
    
#     followup_input = {
#         "user_followup_input": "How much is the second flight?",
#         "session_id": "test_session_123",
#         "context": context_data
#     }
    
#     try:
#         result = kickoff_crew(followup_crew, followup_input)
#         print(f"\nFollow-up Query Result:\n{result}\n")
#     except Exception as e:
#         print(f"Error: {e}")