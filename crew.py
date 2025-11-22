from crewai import Crew, Process
from agents import (
    language_agent,
    manager_agent,
    flight_agent,
    hotel_agent,
    transport_agent,
    attractions_agent,
    response_agent,
    followup_agent,
    booking_agent,
    followup_manager_agent
)
from tasks import (
    task_language_detection,
    task_search,
    task_final_response,
    task_followup_handling, 
)
from config import settings
from logger import setup_logger

logger = setup_logger(__name__)

def create_travel_crew(is_followup: bool = False, context_data: dict = None) -> Crew:
    
    if is_followup:
        # ==================== FOLLOW-UP MODE (PURE HIERARCHICAL) ====================
        logger.info("Creating crew for FOLLOW-UP query (pure hierarchical)")
        
        crew = Crew(
            agents=[
                followup_agent,    # Handles general follow-up questions
                booking_agent      # Handles booking confirmations
            ],
            tasks=[
                task_followup_handling  # Single task - manager decides routing
            ],
            process=Process.hierarchical,
            manager_agent=followup_manager_agent,  # Manager makes all decisions
            memory=False,
            verbose=settings.CREW_VERBOSE,
            max_rpm=10
        )
        
        logger.info("Follow-up hierarchical crew created successfully")
        return crew
    
    else:
        # ==================== INITIAL MODE (HIERARCHICAL) ====================
        logger.info("Creating crew for INITIAL query with hierarchical process")
        
        crew = Crew(
            agents=[
                language_agent,
                flight_agent,
                hotel_agent,
                transport_agent,
                attractions_agent,
                response_agent
            ],
            tasks=[
                task_language_detection,
                task_search,
                task_final_response
            ],
            process=Process.sequential,
            manager_agent=manager_agent,
            memory=False,
            verbose=settings.CREW_VERBOSE,
            max_rpm=10
        )
        
        logger.info("Hierarchical crew created successfully")
        return crew

def kickoff_crew(crew: Crew, inputs: dict) -> str:
   
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