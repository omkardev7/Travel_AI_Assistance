# tasks.py
"""
Task definitions for Multi-Lingual Travel Assistant
UPDATED: Follow-up task now includes agent_outputs in context
"""

from crewai import Task
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
from logger import setup_logger
from config import settings

logger = setup_logger(__name__, settings.LOG_LEVEL)

# ==================== TASK 1: Language Detection & Translation ====================

task_language_detection = Task(
    description="""
    Analyze the user's input message.
    
    Your tasks:
    1. Detect the language of the input
    2. Translate to English if not English
    3. Validate if this is travel-related (flight, hotel, train, bus, attractions)
    4. Extract key information: service type, origin, destination, dates, preferences
    5. Check if all required information is present
    
    Return ONLY a valid JSON object with this exact structure:
    {{
        "detected_language": "language_code",
        "language_name": "Language Name",
        "english_translation": "translated text",
        "is_travel_related": true/false,
        "service_type": "flight/hotel/train/bus/attractions",
        "entities": {{
            "origin": "city name or null",
            "destination": "city name or null",
            "date": "date or null",
            "guests": number or null,
            "budget": "amount or null"
        }},
        "is_complete": true/false,
        "missing_info": ["list", "of", "missing", "fields"]
    }}
    
    If not travel-related, set is_travel_related to false and provide a polite message.
    
    User input: {user_input}
    """,
    agent=language_agent,
    expected_output="Valid JSON object with language detection and entity extraction results"
)

# ==================== TASK 2: Orchestration ====================

task_orchestration = Task(
    description="""
    Based on the previous agent's JSON output, determine which travel service is needed.
    
    Analyze the service_type and entities from the language detection output.
    
    Return ONLY a valid JSON object with this structure:
    {{
        "service_type": "flight/hotel/train/bus/attractions",
        "search_params": {{
            "origin": "value",
            "destination": "value",
            "date": "value",
            "passengers": number
        }},
        "route_to": "flight_agent/hotel_agent/transport_agent/attractions_agent"
    }}
    
    DO NOT delegate. DO NOT call other agents. 
    Just return the JSON routing decision.
    """,
    agent=orchestrator_agent,
    expected_output="Valid JSON with routing decision and search parameters",
    context=[task_language_detection]
)

# ==================== TASK 3A: Flight Search ====================

task_flight_search = Task(
    description="""
    Check the orchestration output from the previous task.
    
    IF route_to is "flight_agent":
        1. Extract search parameters from the orchestration output
        2. Build a search query based on origin, destination, and date
        3. Use the EXA search tool to find flights
        4. Return top 5 results in this JSON format:
        {{
            "flights": [
                {{
                    "airline": "name",
                    "flight_number": "code",
                    "departure": "time",
                    "arrival": "time",
                    "duration": "Xh Ym",
                    "price": "currency amount",
                    "stops": "Non-stop/1 stop"
                }}
            ]
        }}
    
    ELSE (if not routed to you):
        Return: {{"flights": [], "message": "Not applicable"}}
    """,
    agent=flight_agent,
    expected_output="JSON with flight results or not applicable message",
    context=[task_orchestration]
)

# ==================== TASK 3B: Hotel Search ====================

task_hotel_search = Task(
    description="""
    Check the orchestration output from the previous task.
    
    IF route_to is "hotel_agent":
        1. Extract search parameters
        2. Build a search query for hotels in the destination
        3. Use EXA tool to search for hotels
        4. Return top 5 results in JSON format:
        {{
            "hotels": [
                {{
                    "name": "hotel name",
                    "rating": "X/5",
                    "price_per_night": "currency amount",
                    "amenities": ["WiFi", "Pool"],
                    "location": "description"
                }}
            ]
        }}
    
    ELSE (if not routed to you):
        Return: {{"hotels": [], "message": "Not applicable"}}
    """,
    agent=hotel_agent,
    expected_output="JSON with hotel results or not applicable message",
    context=[task_orchestration]
)

# ==================== TASK 3C: Transport Search ====================

task_transport_search = Task(
    description="""
    Check the orchestration output from the previous task.
    
    IF route_to is "transport_agent":
        1. Extract search parameters
        2. Build a search query for trains or buses
        3. Use EXA tool to search for transportation options
        4. Return top 5 results in JSON format:
        {{
            "trains": [
                {{
                    "name": "train name",
                    "number": "train number",
                    "departure": "time",
                    "arrival": "time",
                    "duration": "Xh Ym",
                    "price": "currency amount"
                }}
            ]
        }}
    
    ELSE (if not routed to you):
        Return: {{"trains": [], "message": "Not applicable"}}
    """,
    agent=transport_agent,
    expected_output="JSON with train/bus results or not applicable message",
    context=[task_orchestration]
)

# ==================== TASK 3D: Attractions Search ====================

task_attractions_search = Task(
    description="""
    Check the orchestration output from the previous task.
    
    IF route_to is "attractions_agent":
        1. Extract destination from search parameters
        2. Build a search query for local attractions
        3. Use EXA tool to search for recommendations
        4. Return top 5 results in JSON format:
        {{
            "attractions": [
                {{
                    "name": "attraction name",
                    "type": "category",
                    "description": "brief description",
                    "rating": "X/5",
                    "entry_fee": "amount or Free"
                }}
            ]
        }}
    
    ELSE (if not routed to you):
        Return: {{"attractions": [], "message": "Not applicable"}}
    """,
    agent=attractions_agent,
    expected_output="JSON with attraction results or not applicable message",
    context=[task_orchestration]
)

# ==================== TASK 4: Final Response Translation ====================

task_final_response = Task(
    description="""
    Translate the search results to the user's original language.
    
    You will receive:
    1. Language detection from task 1 (detected_language, language_name)
    2. Search results from tasks 3A/3B/3C/3D
    
    Your job:
    1. Identify the user's detected language
    2. Get the search results (ignore any "Not applicable" messages)
    3. Translate all results to the detected language naturally
    4. Format with:
       - Proper currency symbols (₹, $, ¥, €)
       - Natural date/time formats
       - Numbered options (1, 2, 3, 4, 5)
       - Clear structure
    5. End with a follow-up question in the user's language
    
    Return plain text response in the user's language, NOT JSON.
    Make it conversational and natural.
    """,
    agent=response_agent,
    expected_output="Natural language response translated to user's original language",
    context=[task_language_detection, task_orchestration, task_flight_search, 
             task_hotel_search, task_transport_search, task_attractions_search]
)

# ==================== TASK 5: Follow-up Response (UPDATED) ====================

task_followup_response = Task(
    description="""
    Handle the user's follow-up question using the provided context.
    
    You have access to COMPLETE context from custom memory including:
    - Detected language and language name
    - Previous entities (origin, destination, dates)
    - Previous search results (flights, hotels, trains, attractions)
    - Conversation history
    - **AGENT OUTPUTS**: Complete outputs from all previous agents including:
      * Language Agent: Detected language, entities, translations
      * Orchestrator Agent: Routing decisions
      * Search Agents: Raw search results with all details
      * Response Agent: Previous translated responses
    
    The agent_outputs field contains a list of dictionaries with:
    - agent_name: Which agent produced this output
    - task_name: What task was executed
    - output_type: 'json' or 'text'
    - output_data: The actual output (parsed JSON if applicable)
    - timestamp: When this was produced
    
    Your tasks:
    1. Understand what the user is asking about
    2. Parse references like:
       - "second one", "option 2", "दूसरी", "இரண்டாவது" → Index 2
       - "first", "option 1", "पहली", "முதல்" → Index 1
       - "how much", "price", "कितना", "எவ்வளவு" → Price information
       - "what time", "timing", "कब", "என்ன நேரம்" → Timing information
    3. Search through agent_outputs to find the most relevant information
    4. Use search_results for structured data or agent_outputs for complete details
    5. Respond in the detected language
    6. Keep response concise and helpful
    
    Return plain text response in the user's language, NOT JSON.
    
    User's follow-up question: {user_followup_input}
    Detected language: {detected_language}
    Language name: {language_name}
    Previous entities: {entities}
    Search results: {search_results}
    Conversation history: {conversation_history}
    Agent outputs: {agent_outputs}
    """,
    agent=followup_agent,
    expected_output="Direct answer to follow-up question in user's original language"
)

logger.info("All tasks defined successfully (with agent outputs context)")