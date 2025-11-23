from crewai import Task
from agents import (
    language_agent,
    response_agent,
    followup_agent,
    booking_agent,
    manager_agent
)
from logger import setup_logger
from config import settings

logger = setup_logger(__name__, settings.LOG_LEVEL)

# ==================== TASK 1: Language Detection & Translation ====================

task_language_detection = Task(
    description=f"""
   Parse the user's travel query to extract all booking-relevant information.
    
    USER INPUT: {{user_input}}
    
    EXECUTE THIS SYSTEMATIC PROCESS:
    
    **STEP 1**: DETECT LANGUAGE
      Identify the language and return both ISO code (en, hi, ta, mr, bn, etc.) and full name.
    
    **STEP 2**: TRANSLATE TO ENGLISH
      Convert query to English while preserving all travel details, intent, and context.
    
    **STEP 3**: VALIDATE IF TRAVEL-RELATED
      Determine if this concerns flights, hotels, trains, buses, attractions, or travel packages.
      Set is_travel_related: true/false
    
    **STEP 4**: IDENTIFY SERVICE TYPE
      Map to one of: flight, hotel, train, bus, attractions
    
    **STEP 5**: EXTRACT ENTITIES (Use inject_date feature for current date)
    
      🌍 origin: Departure city with code if known (BOM, DEL, NDLS)
      🎯 destination: Arrival city (REQUIRED for all services)
      📅 date: CONVERT to YYYY-MM-DD format (use injected current date for relative dates)
         - Store both converted date and original phrase
         - Examples: "tomorrow" → "2024-01-16", "kal" → "2024-01-16"
      🔄 return_date: For round trips, also in YYYY-MM-DD
      👥 passengers: Number of travelers (default: 1 if not mentioned)
      💺 class: Travel class (default: "economy" for flights/trains, "standard" for hotels)
      💰 budget: Price range if mentioned (can be null)
      ⏰ time_preference: morning/afternoon/evening/night if mentioned
    
    **STEP 6**: CHECK COMPLETENESS
    
      Mark COMPLETE if:
      - Flight/Train/Bus: has origin + destination + date
      - Hotel: has destination + check-in date + optionally check-out date
      - Attractions: has destination only
    
    Mark INCOMPLETE if critical info missing.
    
    **STEP 7**: GENERATE FOLLOW-UP IF INCOMPLETE
    
      Ask for ALL missing information in ONE message using user's language:
    
      Hindi: "आपकी यात्रा की पूरी जानकारी के लिए कृपया बताएं:\\n\\n1️⃣ कहाँ से कहाँ जाना है?\\n2️⃣ यात्रा की तारीख?\\n3️⃣ कितने यात्री?\\n4️⃣ कौन सी सेवा चाहिए?\\n\\nएक साथ सारी जानकारी दें! 🙏"
    
      English: "To help you, please provide ALL details:\\n\\n1️⃣ From where to where?\\n2️⃣ Travel date?\\n3️⃣ Number of travelers?\\n4️⃣ Service needed?\\n\\nProvide complete info for faster help! 🙏"
    
      Tamil: "உங்கள் பயணத்திற்கு:\\n\\n1️⃣ எங்கிருந்து எங்கு?\\n2️⃣ பயண தேதி?\\n3️⃣ எத்தனை பேர்?\\n4️⃣ என்ன சேவை?\\n\\nமுழு தகவலை ஒரே செய்தியில் கொடுங்கள்! 🙏"
    
    **EXPECTED OUTPUT**: Valid JSON (NO markdown)
    
    {{
        "detected_language": "hi",
        "language_name": "Hindi",
        "english_translation": "I need flight from Mumbai to Delhi on 2024-01-15 for 2 passengers",
        "is_travel_related": true,
        "service_type": "flight",
        "entities": {{
            "origin": "Mumbai",
            "origin_code": "BOM",
            "destination": "Delhi",
            "destination_code": "DEL",
            "date": "2024-01-15",
            "date_original": "tomorrow",
            "return_date": null,
            "passengers": 2,
            "class": "economy",
            "budget": null,
            "time_preference": null
        }},
        "is_complete": true,
        "missing_info": [],
        "followup_question": null,
        "assumptions_made": ["Assumed economy class", "Converted 'tomorrow' to 2024-01-15"]
    }}
    FINAL INSTRUCTION:
    Once you have the JSON output, your job is DONE. 
    Simply return the JSON object as your final answer.
    DO NOT attempt to communicate with the 'travel_manager'.
    DO NOT use the 'Delegate work to coworker' tool to send this to the manager.
    """,
    agent=language_agent,
    expected_output="Valid JSON object with language detection, translation, entity extraction, and completeness check"
)

# ==================== TASK 2: Search (Manager Delegates to ONE Specialist) ====================

task_search = Task(
    description=f"""
    Coordinate the travel search by routing to the appropriate specialist.
    
    **STEP 1**: PARSE INPUT FROM LANGUAGE AGENT
    INPUT: You receive the language agent's JSON output containing:
    - service_type: flight/hotel/train/bus/attractions
    - entities: {{origin, destination, date, guests, budget}}
    - is_complete: true/false
    
    **STEP 2**: IF INCOMPLETE (is_complete == false)
    ❌ DO NOT search
    ✅ Return immediately:
    {{
        "status": "incomplete",
        "followup_question": "<question from language agent>"
    }}
    STOP EXECUTION
    
    **STEP 3**: IF COMPLETE (is_complete == true)
    - Based on service_type, the appropriate specialist will be assigned:
      * service_type="flight" → "flight_agent" handles this
            Pass: "Search flights from (origin) to (destination) on (date) for (passengers) passengers in (class) class"

      * service_type="hotel" → "hotel_agent" handles this
            Pass: "Search hotels in (destination) for check-in (date) with (guests) guests"

      * service_type="train" or "bus" → "train_and_bus_agent" handles this
            Pass: "Search (service_type) from (origin) to (destination) on (date) for (passengers) passengers"

      
      * service_type="attractions" → "local_attractions_agent" handles this
            Pass: "Search top attractions in (destination)"
            
    - The specialist will:
      1. Build a search query from entities
      2. Use EXA tool to search relevant platforms
      3. Extract and structure results
      4. Return top 6 options in JSON format
    
    EXPECTED OUTPUT:
    
    If incomplete:
    {{
        "status": "incomplete",
        "followup_question": "आपकी यात्रा की पूरी जानकारी..."
    }}
    
    If complete (example for flights):
    {{
        "status": "success",
        "service_type": "flight",
        "route": "Mumbai (BOM) → Delhi (DEL)",
        "travel_date": "2024-01-15",
        "passengers": 2,
        "flights": [
            {{
                "airline": "IndiGo",
                "flight_number": "6E-2341",
                "departure_time": "06:15",
                "arrival_time": "08:25",
                "duration": "2h 10m",
                "price": "₹4,250",
                "price_numeric": 4250,
                "class": "Economy",
                "stops": "Non-stop",
                "baggage": "15kg check-in + 7kg cabin"
            }}
        ],
        "result_count": 6,
        "price_range": "₹4,250 - ₹8,900"
    }}
    
    Context from language agent: Use the output from the previous task
    """,
    agent=manager_agent,
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
      IF search results contain actual travel options:
    
    STEP 1: TRANSLATE NATURALLY
    - Translate to detected_language
    - Use conversational tone
    - Maintain technical accuracy (prices, times, names)
    
    STEP 2: FORMAT WITH STRUCTURE
    
    Basic structure:
    📍 Brief acknowledgment + summary (1-2 lines)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    [Numbered options 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣]
    
    💡 [Price range or tip]
    
    ❓ [Follow-up question]
    
    Service-specific formats:
    
    FLIGHTS:
    ✈️ [Airline] [Flight No]
       🕐 [Depart] → [Arrive] ([Duration])
       💰 [Price] | [Class] | [Stops]
    
    HOTELS:
    🏨 [Name] ⭐ [Rating]
       📍 [Location]
       💰 [Price]/night | [Room Type]
       ✅ [Amenities]
    
    TRAINS:
    🚂 [Train Name] ([Number])
       🕐 [Depart] → [Arrive] ([Duration])
       💺 [Classes]: [Prices]
       📊 [Availability]
    
    BUSES:
    🚌 [Operator] - [Type]
       🕐 [Depart] → [Arrive] ([Duration])
       💰 [Price] | 💺 [Seats]
    
    ATTRACTIONS:
    📍 [Name]
       🏛️ [Category] | ⭐ [Rating]
       💰 [Entry Fee] | ⏰ [Timings]
       📝 [Brief description]
    
    STEP 3: FORMAT NUMBERS & CURRENCY
    - Currency: Use ₹ symbol (₹4,250 not Rs. 4,250)
    - Time: 12-hour with AM/PM (06:30 AM not 0630)
    - Keep proper spacing and line breaks
    
    STEP 4: ADD ENGAGING FOLLOW-UP
    
    Hindi: "क्या आप किसी विकल्प के बारे में और जानना चाहेंगे? या बुकिंग करें? 😊"
    
    English: "Would you like more details about any option? Or shall I help you book? 😊"
    
    Tamil: "ஏதேனும் விருப்பத்தைப் பற்றி மேலும் தெரிந்து கொள்ள விரும்புகிறீர்களா? 😊"
    
    Marathi: "अधिक माहिती हवी आहे का? किंवा बुकिंग करू? 😊"
    
    QUALITY CHECKLIST:
    ✅ Natural and conversational
    ✅ Easy to scan (emojis, spacing)
    ✅ All prices and times visible
    ✅ Translated to user's language
    ✅ No technical jargon
    ✅ Actionable next steps
    ✅ PLAIN TEXT (NO JSON, NO markdown code blocks)
    
    IMPORTANT: Return plain text in the user's language, NOT JSON.
    Make it conversational and easy to read.
    
    Context: Use outputs from Task 1 (language) and Task 2 (search)
    
    CRITICAL COORDINATION RULES (TO PREVENT ERRORS):
    1. The 'travel_manager' is your SUPERVISOR, NOT a coworker. 
    2. DO NOT attempt to delegate this task back to 'travel_manager'.
    3. DO NOT use the 'Delegate work to coworker' tool on 'travel_manager'.
    4. Your job is to simply RETURN the final translated text string as your output.
    5. Do not ask the manager for confirmation; just produce the best translation you can.
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
    - "followup_question_agent": For general questions about search results
    - "booking_confirmation_agent": For booking confirmations
    
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
      * Use emojis and friendly toneorigin
      * It should look realistic and professional
      
    - If booking details are incomplete:
      * Request for missing details (names, contact, email) in user's language
    
    Manager: Coordinate the appropriate specialist based on request type.
    
    Return plain text in user's language, NOT JSON.
    """,
    agent=None,  # Manager delegates
    expected_output="Direct answer, booking details request, or booking confirmation in user's language",
    context=[task_language_detection, task_search,task_final_response]
)


logger.info("All tasks defined successfully (with booking task)")