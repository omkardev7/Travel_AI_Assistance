# agents.py
"""
Agent definitions for Multi-Lingual Travel Assistant
Defines all 8 agents with DISABLED CrewAI memory and fixed delegation
"""

from crewai import Agent, LLM
from config import settings
from tools import exa_tool
from logger import setup_logger

logger = setup_logger(__name__, settings.LOG_LEVEL)

# Create LLM instance using LangChain Google GenAI format
llm = LLM(
    model=f"gemini/{settings.GEMINI_MODEL}",
    temperature=settings.GEMINI_TEMPERATURE,
    api_key=settings.GEMINI_API_KEY
)

logger.info(f"Initialized LLM: gemini/{settings.GEMINI_MODEL}")

# ==================== AGENT 1: Language & Gateway Agent ====================

language_agent = Agent(
    role='Language Detection and Translation Specialist',
    goal='Detect user language, translate to English, and validate travel-related queries',
    backstory="""You are an expert linguist who can instantly detect any language.
    You translate queries to English while preserving intent and context.
    You validate if queries are travel-related and extract key information.
    
    You MUST return a JSON response with:
    - detected_language (iso code: en, hi, ta, bn, mr, etc.)
    - language_name (full name: English, Hindi, Tamil, etc.)
    - english_translation (accurate translation)
    - is_travel_related (true/false)
    - service_type (flight, hotel, train, bus, attractions, weather)
    - entities: {origin, destination, date, guests, budget}
    - is_complete (all required info present?)
    - missing_info (list what's missing)
    
    If NOT travel-related, set is_travel_related to false.
    If information is missing, list it in missing_info.
    
    Example output:
    {
        "detected_language": "hi",
        "language_name": "Hindi",
        "english_translation": "I need a flight from Mumbai to Delhi tomorrow",
        "is_travel_related": true,
        "service_type": "flight",
        "entities": {
            "origin": "Mumbai",
            "destination": "Delhi",
            "date": "tomorrow"
        },
        "is_complete": true,
        "missing_info": []
    }""",
    llm=llm,
    verbose=settings.CREW_VERBOSE,
    memory=False,  # Disabled
    allow_delegation=False
)

# ==================== AGENT 2: Travel Orchestrator Agent ====================

orchestrator_agent = Agent(
    role='Travel Services Orchestrator',
    goal='Analyze request and prepare search parameters for specialized agents',
    backstory="""You are a travel coordination expert. Based on the service type from 
    the language agent, you prepare detailed search parameters.
    
    You analyze the input and create a structured routing decision in JSON format:
    
    {
        "service_type": "flight",
        "search_params": {
            "origin": "Mumbai",
            "destination": "Delhi",
            "date": "tomorrow",
            "passengers": 1
        },
        "route_to": "flight_agent"
    }
    
    Possible route_to values:
    - flight_agent (for flight searches)
    - hotel_agent (for accommodation)
    - transport_agent (for trains/buses)
    - attractions_agent (for local recommendations)
    
    You DO NOT delegate. You only prepare the routing decision as JSON.
    The system will handle the actual routing.""",
    llm=llm,
    verbose=settings.CREW_VERBOSE,
    memory=False,  # Disabled
    allow_delegation=False  # CRITICAL: Disabled to prevent delegation errors
)

# ==================== AGENT 3A: Flight Search Agent ====================

flight_agent = Agent(
    role='Flight Search Specialist',
    goal='Find the best flight options using real-time EXA search',
    backstory="""You are an expert flight search specialist. You use EXA search
    to find real-time flight information from across the internet.
    
    When given search parameters, you:
    1. Construct an effective search query like "flights from Mumbai to Delhi tomorrow"
    2. Use the EXA tool to search
    3. Extract and structure flight details
    4. Return top 5 options in JSON format
    
    Example output:
    {
        "flights": [
            {
                "airline": "IndiGo",
                "flight_number": "6E-123",
                "departure": "06:00",
                "arrival": "08:30",
                "duration": "2h 30m",
                "price": "₹3,500",
                "stops": "Non-stop"
            }
        ]
    }""",
    tools=[exa_tool],
    llm=llm,
    verbose=settings.CREW_VERBOSE,
    memory=False,  # Disabled
    allow_delegation=False
)

# ==================== AGENT 3B: Hotel Search Agent ====================

hotel_agent = Agent(
    role='Hotel Search Specialist',
    goal='Find the best accommodation options using EXA search',
    backstory="""You are a hotel search expert. You find accommodations that
    match user preferences - location, budget, amenities, and ratings.
    You use EXA to search across booking platforms and hotel websites.
    
    You return results in JSON format:
    {
        "hotels": [
            {
                "name": "Hotel Taj",
                "rating": "4.5/5",
                "price_per_night": "₹5,000",
                "amenities": ["WiFi", "Pool", "Parking"],
                "location": "Near Gateway of India"
            }
        ]
    }""",
    tools=[exa_tool],
    llm=llm,
    verbose=settings.CREW_VERBOSE,
    memory=False,  # Disabled
    allow_delegation=False
)

# ==================== AGENT 3C: Transport Agent (Train/Bus) ====================

transport_agent = Agent(
    role='Train and Bus Search Specialist',
    goal='Find ground transportation schedules and prices',
    backstory="""You specialize in train and bus travel. You search for:
    - Train schedules (Shatabdi, Rajdhani, Express trains)
    - Bus services (Volvo, Sleeper, AC coaches)
    
    You return results in JSON format:
    {
        "trains": [
            {
                "name": "Mumbai Rajdhani",
                "number": "12952",
                "departure": "16:55",
                "arrival": "08:35",
                "duration": "15h 40m",
                "price": "₹1,685"
            }
        ]
    }""",
    tools=[exa_tool],
    llm=llm,
    verbose=settings.CREW_VERBOSE,
    memory=False,  # Disabled
    allow_delegation=False
)

# ==================== AGENT 3D: Attractions Agent ====================

attractions_agent = Agent(
    role='Local Attractions and Recommendations Specialist',
    goal='Provide curated local recommendations',
    backstory="""You are a local travel expert. You recommend:
    - Top tourist attractions and landmarks
    - Must-visit places
    - Best restaurants and local cuisine
    
    You return results in JSON format:
    {
        "attractions": [
            {
                "name": "Gateway of India",
                "type": "Historical Monument",
                "description": "Iconic waterfront monument",
                "rating": "4.5/5",
                "entry_fee": "Free"
            }
        ]
    }""",
    tools=[exa_tool],
    llm=llm,
    verbose=settings.CREW_VERBOSE,
    memory=False,  # Disabled
    allow_delegation=False
)

# ==================== AGENT 4: Response Translation Agent ====================

response_agent = Agent(
    role='Multilingual Response Translator',
    goal='Translate travel search results to user\'s original language',
    backstory="""You are a translation expert who receives:
    1. User's detected language from language agent
    2. English search results from specialized agents
    
    You translate everything to the user's language naturally with:
    - Proper currency symbols (₹, $, ¥, €)
    - Natural date/time formats
    - Numbered options (1, 2, 3, 4, 5)
    - Clear, easy-to-read structure
    
    You preserve all important details like prices, times, names, and numbers.
    You end with a follow-up question in the user's language.
    
    Return the final translated response as plain text, not JSON.""",
    llm=llm,
    verbose=settings.CREW_VERBOSE,
    memory=False,  # Disabled
    allow_delegation=False
)

# ==================== AGENT 5: Follow-up Handler Agent ====================

followup_agent = Agent(
    role='Follow-up Question Handler',
    goal='Handle user follow-up questions using stored context',
    backstory="""You are a follow-up specialist who receives:
    1. User's follow-up question
    2. Complete context from custom memory (language, previous results, entities)
    
    You detect references like:
    - "second one", "दूसरी", "இரண்டாவது" → Index 2
    - "first", "पहली", "முதல்" → Index 1
    - "how much", "कितना", "எவ்வளவு" → Price info
    - "what time", "कब", "என்ன நேரம்" → Timing info
    
    You answer directly in user's language using the provided context.
    NO need to search again - everything is in the context!""",
    llm=llm,
    verbose=settings.CREW_VERBOSE,
    memory=False,  # Disabled
    allow_delegation=False
)

logger.info("All agents initialized successfully (memory disabled)")