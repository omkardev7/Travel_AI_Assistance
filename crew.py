# crew.py
"""
CrewAI orchestration for Multi-Lingual Travel Assistant
UPDATED: Added hierarchical followup crew with booking
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
    """
    Create CrewAI crew with hierarchical process
    
    FLOWS:
    
    INITIAL QUERY (is_followup=False, is_booking=False):
        User Input → 
        Task 1 (Language Agent) →
        Task 2 (Manager delegates to specialist) →
        Task 3 (Response Agent) →
        Response
    
    FOLLOW-UP QUERY (is_followup=True, is_booking=False):
        User Input → 
        Task 4 (Follow-up Agent via manager) →
        Response
    
    BOOKING CONFIRMATION (is_followup=True, is_booking=True):
        User Input → 
        Task 5 (Booking Agent via manager) →
        Response
    
    Args:
        is_followup: Whether this is a follow-up question
        is_booking: Whether this is a booking confirmation
        context_data: Context from custom memory
    
    Returns:
        Configured Crew instance
    """
    
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