# crew.py
"""
CrewAI orchestration for Multi-Lingual Travel Assistant
CORRECTED: Proper hierarchical process with manager delegation
"""

from crewai import Crew, Process
from agents import (
    language_agent,
    manager_agent,
    flight_agent,
    hotel_agent,
    transport_agent,
    attractions_agent,
    response_agent,
    followup_agent
)
from tasks import (
    task_language_detection,
    task_search,
    task_final_response,
    task_followup_response
)
from config import settings
from logger import setup_logger

logger = setup_logger(__name__)

def create_travel_crew(is_followup: bool = False, context_data: dict = None) -> Crew:
    """
    Create CrewAI crew with hierarchical process
    
    CORRECTED FLOW:
    
    INITIAL QUERY (is_followup=False):
        User Input → 
        Task 1 (Language Agent: detect, translate, extract) →
        Task 2 (Manager delegates to ONE specialist based on service_type) →
        Task 3 (Response Agent: translate back to user's language)
    
    FOLLOW-UP QUERY (is_followup=True):
        User Input → 
        Task 4 (Follow-up Agent: uses custom memory context) →
        Response
    
    KEY CHANGES:
    1. Manager agent is passed to manager_agent parameter (not as a regular agent)
    2. All specialist agents are available for delegation
    3. Single search task - manager routes to appropriate specialist
    4. NO separate tasks for each specialist
    
    Args:
        is_followup: Whether this is a follow-up question
        context_data: Context from custom memory (for follow-ups)
    
    Returns:
        Configured Crew instance
    """
    
    if is_followup:
        # ==================== FOLLOW-UP MODE ====================
        logger.info("Creating crew for FOLLOW-UP query")
        
        crew = Crew(
            agents=[followup_agent],
            tasks=[task_followup_response],
            process=Process.sequential,
            memory=False,  # Using custom SQLite memory
            verbose=settings.CREW_VERBOSE,
            full_output=True
        )
        
        logger.info("Follow-up crew created successfully")
        return crew
    
    else:
        # ==================== INITIAL MODE (HIERARCHICAL) ====================
        logger.info("Creating crew for INITIAL query with hierarchical process")
        
        crew = Crew(
            agents=[
                language_agent,      # Always runs first
                flight_agent,        # Available for delegation
                hotel_agent,         # Available for delegation
                transport_agent,     # Available for delegation
                attractions_agent,   # Available for delegation
                response_agent       # Always runs last
            ],
            tasks=[
                task_language_detection,  # Task 1: Detect & translate
                task_search,              # Task 2: Search (manager delegates)
                task_final_response       # Task 3: Translate response
            ],
            process=Process.hierarchical,    # CRITICAL: Hierarchical mode
            manager_agent=manager_agent,     # CRITICAL: Manager agent
            memory=False,                    # Using custom memory
            verbose=settings.CREW_VERBOSE,
            full_output=True
        )
        
        logger.info("Hierarchical crew created successfully")
        logger.info("Manager: Travel Services Coordinator")
        logger.info("Specialists: Flight, Hotel, Transport, Attractions")
        
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