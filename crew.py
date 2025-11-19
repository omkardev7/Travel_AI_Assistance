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
    task_followup_response,
    task_booking_confirmation
)
from config import settings
from logger import setup_logger

logger = setup_logger(__name__)

def create_travel_crew(is_followup: bool = False, is_booking: bool = False, context_data: dict = None) -> Crew:
    
    if is_followup:
        if is_booking:
            # ==================== BOOKING MODE ====================
            logger.info("Creating crew for BOOKING confirmation")
            
            crew = Crew(
                agents=[booking_agent],
                tasks=[task_booking_confirmation],
                process=Process.sequential,
                memory=False,
                verbose=settings.CREW_VERBOSE,
                full_output=True
            )
            
            logger.info("Booking crew created successfully")
            return crew
        else:
            # ==================== FOLLOW-UP MODE (HIERARCHICAL) ====================
            logger.info("Creating crew for FOLLOW-UP query (hierarchical)")
            
            crew = Crew(
                agents=[followup_agent, booking_agent],
                tasks=[task_followup_response],
                process=Process.hierarchical,
                manager_agent=followup_manager_agent,
                memory=False,
                verbose=settings.CREW_VERBOSE,
                full_output=True
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
            process=Process.hierarchical,
            manager_agent=manager_agent,
            memory=False,
            verbose=settings.CREW_VERBOSE,
            full_output=True
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