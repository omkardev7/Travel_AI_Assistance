# tasks.py
"""
Task definitions for Multi-Lingual Travel Assistant
CORRECTED: Single search task, manager routes to appropriate specialist
"""

from crewai import Task
from agents import (
    language_agent,
    response_agent,
    followup_agent
)
from logger import setup_logger
from config import settings

logger = setup_logger(__name__, settings.LOG_LEVEL)

# ==================== TASK 1: Language Detection & Translation ====================

task_language_detection = Task(
    description="""
    Analyze the user's input message and extract all travel-related information.
    
    Steps:
    1. Detect the language of the input (return ISO code and full name)
    2. Translate to English if not already English
    3. Validate if this is travel-related
    4. Identify service type: flight, hotel, train, bus, or attractions
    5. Extract key entities:
       - origin: Starting location (for flights, trains, buses)
       - destination: Target location (required for all services)
       - date: Travel date or check-in date
       - guests: Number of travelers (for hotels)
       - budget: Price preference (optional)
    6. Check if all required information is present:
       - flight: needs origin, destination, date
       - hotel: needs destination, check-in date, optionally check-out date
       - train/bus: needs origin, destination, date
       - attractions: needs destination only
    7. If information is missing, ask user to provide ALL missing details in ONE complete message
    
    CRITICAL: When incomplete, request ALL missing information at once. 
    DO NOT ask incremental questions. Be clear about what's needed.
    
    Return ONLY a valid JSON object (no markdown formatting):
    {{
        "detected_language": "hi",
        "language_name": "Hindi",
        "english_translation": "I need a flight from Mumbai to Delhi tomorrow",
        "is_travel_related": true,
        "service_type": "flight",
        "entities": {{
            "origin": "Mumbai",
            "destination": "Delhi",
            "date": "tomorrow",
            "guests": null,
            "budget": null
        }},
        "is_complete": true,
        "missing_info": [],
        "followup_question": null
    }}
    
    Example when incomplete:
    {{
        "detected_language": "mr",
        "language_name": "Marathi",
        "english_translation": "I want to go from Pune to Delhi",
        "is_travel_related": true,
        "service_type": "flight",
        "entities": {{
            "origin": "Pune",
            "destination": "Delhi",
            "date": null,
            "guests": null
        }},
        "is_complete": false,
        "missing_info": ["date", "guests", "service_type"],
        "followup_question": "कृपया संपूर्ण माहिती एकत्र द्या: तुम्ही केव्हा (तारीख) प्रवास करत आहात, किती लोक प्रवास करत आहेत, कोणती सेवा हवी आहे (विमान/ट्रेन/बस/हॉटेल), आणि तुमचा बजेट काय आहे?"
    }}
    
    User input: {user_input}
    """,
    agent=language_agent,
    expected_output="Valid JSON object with language detection, translation, entity extraction, and completeness check"
)

# ==================== TASK 2: Search (Manager Delegates to ONE Specialist) ====================

task_search = Task(
    description="""
    Fulfill the travel request by searching for the appropriate service.
    
    INPUT: You receive the language agent's JSON output containing:
    - service_type: flight/hotel/train/bus/attractions
    - entities: {{origin, destination, date, guests, budget}}
    - is_complete: true/false
    - followup_question: question in user's language (if incomplete)
    
    LOGIC:
    
    **IF is_complete is FALSE:**
    - DO NOT perform any search
    - Return immediately: {{"status": "incomplete", "followup_question": "<the question>"}}
    
    **IF is_complete is TRUE:**
    - Based on service_type, the appropriate specialist will be assigned:
      * service_type="flight" → Flight Search Specialist handles this
      * service_type="hotel" → Hotel Search Specialist handles this
      * service_type="train" or "bus" → Transport Specialist handles this
      * service_type="attractions" → Attractions Specialist handles this
    
    - The specialist will:
      1. Build a search query from entities
      2. Use EXA tool to search relevant platforms
      3. Extract and structure results
      4. Return top 5 options in JSON format
    
    Expected output (if complete):
    {{
        "flights": [...],  // or "hotels", "trains", "buses", "attractions"
        "search_query": "flights from Mumbai to Delhi tomorrow",
        "result_count": 5
    }}
    
    Expected output (if incomplete):
    {{
        "status": "incomplete",
        "followup_question": "आप कहाँ से यात्रा करना चाहते हैं?"
    }}
    
    Context from language agent: Use the output from the previous task
    """,
    agent=None,  # Manager will delegate to the appropriate specialist
    expected_output="JSON with search results OR status='incomplete' with followup_question",
    context=[task_language_detection]
)

# ==================== TASK 3: Final Response Translation ====================

task_final_response = Task(
    description="""
    Translate the search results (or follow-up question) to the user's original language.
    
    INPUT: You receive:
    1. Language detection info from Task 1:
       - detected_language (e.g., "hi")
       - language_name (e.g., "Hindi")
    2. Search results from Task 2:
       - Either search results (flights/hotels/trains/attractions)
       - OR status="incomplete" with followup_question
    
    PROCESS:
    
    **SCENARIO A: Incomplete Input**
    - If search results contain {{"status": "incomplete", "followup_question": "..."}}
    - Return the followup_question AS-IS (it's already in the user's language)
    - DO NOT add any extra text
    
    **SCENARIO B: Complete Search Results**
    - Translate all results to the detected language
    - Format naturally with:
      * Proper currency symbols: ₹ (India), $ (US), € (Europe), ¥ (Japan)
      * Natural date/time formats
      * Numbered list: 1, 2, 3, 4, 5
      * Clear structure with line breaks
      * Preserve all important details: prices, times, names, flight numbers
    - End with a helpful follow-up question in the user's language like:
      * "क्या आप किसी फ्लाइट के बारे में और जानना चाहेंगे?" (Hindi)
      * "Would you like more details about any of these?"
    
    IMPORTANT: Return plain text in the user's language, NOT JSON.
    Make it conversational and easy to read.
    
    EXAMPLE OUTPUT (Hindi - Complete):
    "यहाँ मुंबई से दिल्ली के लिए 5 फ्लाइट्स मिली हैं:
    
    1. IndiGo 6E-123
       प्रस्थान: 06:00 → आगमन: 08:30
       अवधि: 2 घंटे 30 मिनट
       कीमत: ₹3,500
       नॉन-स्टॉप
    
    2. SpiceJet SG-456
       प्रस्थान: 07:15 → आगमन: 09:45
       अवधि: 2 घंटे 30 मिनट
       कीमत: ₹3,200
       नॉन-स्टॉप
    
    3. Air India AI-860
       प्रस्थान: 08:45 → आगमन: 11:00
       अवधि: 2 घंटे 15 मिनट
       कीमत: ₹5,100
       नॉन-स्टॉप
    
    4. Vistara UK-955
       प्रस्थान: 09:30 → आगमन: 11:45
       अवधि: 2 घंटे 15 मिनट
       कीमत: ₹4,800
       नॉन-स्टॉप
    
    5. IndiGo 6E-234
       प्रस्थान: 10:00 → आगमन: 12:30
       अवधि: 2 घंटे 30 मिनट
       कीमत: ₹3,300
       नॉन-स्टॉप
    
    क्या आप इनमें से किसी फ्लाइट को बुक करना चाहेंगे?"
    
    EXAMPLE OUTPUT (Hindi - Incomplete):
    "आप कहाँ से यात्रा करना चाहते हैं?"
    
    Context: Use outputs from Task 1 (language) and Task 2 (search)
    """,
    agent=response_agent,
    expected_output="Natural language response translated to user's original language",
    context=[task_language_detection, task_search]
)

# ==================== TASK 4: Follow-up Response ====================

task_followup_response = Task(
    description="""
    Handle the user's follow-up question using complete conversation context.
    
    INPUT: You receive:
    - user_followup_input: {user_followup_input}
    - detected_language: {detected_language}
    - language_name: {language_name}
    - entities: {entities}
    - search_results: {search_results}
    - conversation_history: {conversation_history}
    - agent_outputs: {agent_outputs}
    
    PROCESS:
    1. Understand what the user is asking
    2. Parse common references:
       - Numbers: "second", "2nd", "option 2", "दूसरी", "இரண்டாவது" → Index 2
       - "first", "पहली", "முதல்" → Index 1
       - "last", "आखिरी", "கடைசி" → Last item
       - "cheapest", "सबसे सस्ता", "மலிவான" → Sort by price
       - "earliest", "सबसे पहले", "முதல்" → Sort by time
    3. Extract relevant information from context:
       - Use search_results to find specific items
       - Use entities to understand the original query
       - Use conversation_history to understand flow
    4. Provide a concise, helpful answer in the detected language
    5. If user wants to book, acknowledge and provide next steps
    
    IMPORTANT: 
    - Answer directly, don't repeat the entire list
    - Be conversational and natural
    - Return plain text in user's language, NOT JSON
    - If the reference is unclear, politely ask for clarification
    
    EXAMPLE FOLLOW-UPS:
    
    User: "दूसरी वाली के बारे में बताओ" (Tell me about the second one)
    Response: "दूसरी फ्लाइट SpiceJet SG-456 है:
    - प्रस्थान: 07:15 से मुंबई
    - आगमन: 09:45 दिल्ली
    - अवधि: 2 घंटे 30 मिनट
    - कीमत: ₹3,200
    - नॉन-स्टॉप फ्लाइट
    
    क्या आप इसे बुक करना चाहेंगे?"
    
    User: "सबसे सस्ती कौन सी है?" (Which is the cheapest?)
    Response: "सबसे सस्ती फ्लाइट SpiceJet SG-456 है जो ₹3,200 में उपलब्ध है। यह 07:15 पर रवाना होती है और 09:45 पर पहुंचती है।"
    
    User: "book it"
    Response: "मैं SpiceJet SG-456 को बुक करने में आपकी मदद करूंगा। कृपया निम्नलिखित जानकारी दें:
    1. यात्रियों के नाम
    2. संपर्क नंबर
    3. ईमेल पता"
    """,
    agent=followup_agent,
    expected_output="Direct, conversational answer to follow-up question in user's original language"
)

logger.info("All tasks defined successfully (corrected hierarchical setup)")