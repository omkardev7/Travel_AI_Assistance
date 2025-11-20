from crewai import Task
from agents import (
    language_agent,
    response_agent,
    followup_agent,
    booking_agent
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
      * service_type="flight" → "flight_search_specialist" handles this
      * service_type="hotel" → "hotel_search_specialist" handles this
      * service_type="train" or "bus" → "train_and_bus_search_specialist" handles this
      * service_type="attractions" → "local_attractions_and_recommendations_specialist" handles this
    
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
    agent=None,
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
    
    Context: Use outputs from Task 1 (language) and Task 2 (search)
    """,
    agent=response_agent,
    expected_output="Natural language response translated to user's original language",
    context=[task_language_detection, task_search]
)

# ==================== TASK 4: Follow-up Handling (REFACTORED) ====================

task_followup_handling = Task(
    description="""
    Handle the user's follow-up question OR booking request.
    
    The Follow-up Manager will analyze the request and delegate to the appropriate specialist:
    YOUR TEAM OF SPECIALISTS:
    - "followup_question_handler": For general questions about search results
    - "booking_confirmation_specialist": For booking confirmations
    
    IMPORTANT RULES FOR ALL AGENTS:
    - DO NOT use pre-trained data or world knowledge
    - ONLY use user-provided inputs: conversation_history, search_results, 
      agent_outputs, entities, and user_followup_input
    - If information needed is NOT in these inputs, respond: 
      "No information available for your search."
    
    INPUT VARIABLES:
    - user_followup_input: {user_followup_input}
    - detected_language: {detected_language}
    - language_name: {language_name}
    - entities: {entities}
    - search_results: {search_results}
    - conversation_history: {conversation_history}
    - agent_outputs: {agent_outputs}
    
    ROUTING LOGIC (Manager decides):
    
    The manager will analyze the user_followup_input and determine:
    
    1. **BOOKING REQUEST DETECTION**:
       - Keywords: "book", "reserve", "confirm", "बुक", "பதிவு", "বুকিং", etc.
       - Has passenger details: names, contact number, email
       - Has reference to specific option: "first", "second", "option 2", etc.
       → Delegate to Booking Agent
    
    2. **GENERAL FOLLOW-UP QUESTION**:
       - Questions about prices: "how much", "कितना", "எவ்வளவு"
       - Questions about details: "timing", "when", "कब"
       - Comparisons: "cheapest", "fastest", "सबसे सस्ती"
       - Option references: "first one", "second", "दूसरी"
       → Delegate to Follow-up Handler
    
    3. **MISSING INFORMATION**:
       - No relevant data in search_results or context
       → Respond: "No information available for your search."
    
    EXPECTED OUTPUTS:
    
    **From Follow-up Handler:**
    - Plain text answer in user's language
    - Extracted from search_results context
    - Concise and direct
    
    **From Booking Agent:**
    - If booking details are complete:
      * Full booking confirmation in user's language
      * Include PNR/Booking ID, seats/rooms, passenger details
    - If booking details are incomplete:
      * Request for missing details (names, contact, email) in user's language
    
    Manager: Coordinate the appropriate specialist based on request type.
    
    Return plain text in user's language, NOT JSON.
    """,
    agent=None,  # Manager delegates
    expected_output="Direct answer, booking details request, or booking confirmation in user's language",
    context=[task_language_detection, task_search,task_final_response]
)
# ==================== TASK 5: Booking Confirmation ====================

# task_booking_confirmation = Task(
#     description="""
#     Generate a complete mock booking confirmation.
    
#     INPUT: You receive:
#     - user_booking_input: {user_booking_input}
#     - detected_language: {detected_language}
#     - selected_service: {selected_service} (flight/hotel/train/bus details)
#     - passenger_details: {passenger_details} (names, contact, email)
#     - service_type: {service_type} (flight/hotel/train/bus)
    
#     PROCESS:
#     1. Extract passenger names, contact number, email from user_booking_input
#     2. Generate appropriate booking confirmation based on service_type:
       
#        FLIGHT: PNR (6 chars), Seat numbers, Flight details
#        TRAIN: PNR (10 digits), Coach-Berth numbers, Train details
#        BUS: Booking ID (8 chars), Seat numbers, Bus details
#        HOTEL: Booking ID (8 chars), Room numbers, Hotel details
    
#     3. Format in user's language with:
#        - 
#        - Clear confirmation message
#        - All booking details
#        - Passenger/guest information
#        - Total amount
#        - DO NOT mention payment
    
#     4. Return plain text in user's language, NOT JSON
    
#     EXAMPLE FORMATS:
    
#     Hindi (Flight):
#     "✅ आपकी बुकिंग कन्फर्म हो गई है!
    
#     PNR: A7B2K9
    
#     फ्लाइट: IndiGo 6E-123
#     मुंबई → दिल्ली
#     तारीख: 20 नवंबर 2025
#     समय: 06:00 - 08:30
    
#     यात्री:
#     1. राज शर्मा - सीट 12A
#     2. प्रिया शर्मा - सीट 12B
    
#     संपर्क: +91-9876543210
#     ईमेल: raj@example.com
    
#     कुल किराया: ₹7,000"
    
#     Hindi (Train):
#     "✅ रेल टिकट बुक हो गया!
    
#     PNR: 2345678901
    
#     ट्रेन: Mumbai Rajdhani 12952
#     मुंबई → दिल्ली
#     तारीख: 20 नवंबर 2025
#     समय: 16:55 - 08:35
    
#     यात्री:
#     1. राज शर्मा - A1-23 (Lower Berth)
#     2. प्रिया शर्मा - A1-24 (Upper Berth)
    
#     क्लास: 2AC
    
#     संपर्क: +91-9876543210
#     ईमेल: raj@example.com
    
#     कुल किराया: ₹3,370"
    
#     Hindi (Hotel):
#     "✅ होटल बुकिंग कन्फर्म!
    
#     बुकिंग ID: HTL98765
    
#     होटल: Hotel Taj
#     स्थान: Gateway of India, Mumbai
    
#     रूम नंबर: 304, 305
#     रूम टाइप: Deluxe Room
    
#     चेक-इन: 20 नवंबर 2025
#     चेक-आउट: 22 नवंबर 2025
#     रातें: 2
    
#     मेहमान:
#     1. राज शर्मा
#     2. प्रिया शर्मा
    
#     संपर्क: +91-9876543210
#     ईमेल: raj@example.com
    
#     कुल राशि: ₹10,000"
    
#     Return plain text in user's language, NOT JSON.
#     """,
#     agent=booking_agent,
#     expected_output="Complete mock booking confirmation in user's language",
#     context=[task_followup_handling]
# )

# logger.info("All tasks defined successfully (with booking task)")