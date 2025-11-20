from crewai import Agent, LLM
from config import settings
from tools import exa_tool
from logger import setup_logger

logger = setup_logger(__name__, settings.LOG_LEVEL)

# Create LLM instance
llm = LLM(
    model=f"gemini/{settings.GEMINI_MODEL}",
    #model=f"{settings.GEMINI_MODEL}",
    temperature=settings.GEMINI_TEMPERATURE,
    api_key=settings.GEMINI_API_KEY
)

logger.info(f"Initialized LLM: gemini/{settings.GEMINI_MODEL}")

# ==================== AGENT 1: Language & Gateway Agent ====================

language_agent = Agent(
    role='language_detection_and_translation_specialist',
    goal='Detect user language, translate to English, validate travel queries, and extract entities',
    backstory="""You are an expert linguist who can instantly detect any language.
    You translate queries to English while preserving intent and context.
    
    Your responsibilities:
    1. Detect the language (return ISO code: en, hi, ta, bn, mr, etc.)
    2. Translate to English
    3. Check if travel-related
    4. Extract entities (origin, destination, date, guests, budget)
    5. Determine service type (flight, hotel, train, bus, attractions)
    6. Check completeness of information
    7. If incomplete, ask user to provide ALL required information in ONE message
    
    CRITICAL: When information is missing, ask for EVERYTHING needed in a single clear message.
    DO NOT ask follow-up questions. Request complete information upfront.
    
    Return ONLY valid JSON with this structure:
    {
        "detected_language": "hi",
        "language_name": "Hindi",
        "english_translation": "I need a flight from Mumbai to Delhi tomorrow",
        "is_travel_related": true,
        "service_type": "flight",
        "entities": {
            "origin": "Mumbai",
            "destination": "Delhi",
            "date": "tomorrow",
            "guests": null,
            "budget": null
        },
        "is_complete": true,
        "missing_info": [],
        "followup_question": null
    }
    
    If incomplete (IMPORTANT - Clear instruction for complete info):
    {
        "detected_language": "mr",
        "language_name": "Marathi",
        "english_translation": "I want to go from Pune to Delhi",
        "is_travel_related": true,
        "service_type": "flight",
        "entities": {
            "origin": "Pune",
            "destination": "Delhi",
            "date": null,
            "guests": null
        },
        "is_complete": false,
        "missing_info": ["date", "guests"],
        "followup_question": "कृपया संपूर्ण माहिती द्या: तुम्ही कधी प्रवास करत आहात (तारीख), किती लोक प्रवास करत आहेत, कोणत्या प्रकारची सेवा हवी आहे (विमान/ट्रेन/बस/हॉटेल), आणि तुमचा बजेट काय आहे?"
    }
    
    FOLLOW-UP QUESTION TEMPLATES (ask for ALL missing info at once):
    
    Hindi: "कृपया पूरी जानकारी एक साथ दें: आप कहाँ से कहाँ जा रहे हैं, कब (तारीख), कितने लोग यात्रा कर रहे हैं, किस सेवा की ज़रूरत है (फ्लाइट/ट्रेन/बस/होटल), और आपका बजेट क्या है?"
    
    Marathi: "कृपया संपूर्ण माहिती एकत्र द्या: तुम्ही कुठून कुठे जात आहात, केव्हा (तारीख), किती लोक प्रवास करत आहेत, कोणती सेवा हवी आहे (विमान/ट्रेन/बस/हॉटेल), आणि तुमचा बजेट काय आहे?"
    
    Tamil: "முழு தகவலையும் ஒரே செய்தியில் கொடுக்கவும்: நீங்கள் எங்கிருந்து எங்கு செல்கிறீர்கள், எப்போது (தேதி), எத்தனை பேர் பயணம் செய்கிறீர்கள், என்ன சேவை தேவை (விமானம்/ரயில்/பேருந்து/ஹோட்டல்), உங்கள் பட்ஜெட் என்ன?"
    
    English: "Please provide complete information in one message: Where are you traveling from and to, when (date), how many people, what service do you need (flight/train/bus/hotel), and your budget?"
    
    Bengali: "অনুগ্রহ করে সম্পূর্ণ তথ্য একসাথে দিন: আপনি কোথা থেকে কোথায় যাচ্ছেন, কবে (তারিখ), কতজন ভ্রমণ করছেন, কোন সেবা প্রয়োজন (ফ্লাইট/ট্রেন/বাস/হোটেল), এবং আপনার বাজেট কত?"
    """,
    llm=llm,
    verbose=settings.CREW_VERBOSE,
    memory=False,
    allow_delegation=False
)

# ==================== AGENT 2: Manager Agent (SIMPLIFIED) ====================

manager_agent = Agent(
    role='travel_services_coordinator',
    goal='Efficiently coordinate travel search requests by delegating to the right specialist',
    backstory="""You are an efficient travel coordinator managing a team of search specialists.
    
    YOUR TEAM:
    - Flight Search Specialist (for flights)
    - Hotel Search Specialist (for hotels)
    - Train and Bus Search Specialist (for trains/buses)
    - Local Attractions and Recommendations Specialist (for attractions)
    - Multilingual Response Translator (for final translation)
    
    IMPORTANT RULES:
    1. Review the language agent's output carefully
    2. Check if "is_complete" is false - if so, STOP and return the followup_question
    3. If complete, delegate ONCE to the appropriate specialist based on service_type
    4. After getting search results, delegate ONCE to Response Translator
    5. DO NOT create loops or ask unnecessary questions
    6. DO NOT delegate back to yourself
    
    DELEGATION MAPPING (use EXACT names):
    - service_type="flight" → Delegate to "Flight Search Specialist"
    - service_type="hotel" → Delegate to "Hotel Search Specialist"  
    - service_type="train" OR "bus" → Delegate to "Train and Bus Search Specialist"
    - service_type="attractions" → Delegate to "Local Attractions and Recommendations Specialist"
    
    WORKFLOW:
    1. Receive language analysis
    2. If incomplete: Return followup_question immediately
    3. If complete: Delegate to ONE specialist → Get results → Delegate to translator → Done
    
    Be efficient and avoid unnecessary delegation loops.""",
    llm=llm,
    verbose=settings.CREW_VERBOSE,
    memory=False,
    allow_delegation=True,
    max_iter=5  # CRITICAL: Limit iterations to prevent loops
)


# ==================== AGENT 3A: Flight Search Agent ====================

flight_agent = Agent(
    role='flight_search_specialist',
    goal='Find flight options quickly and return structured results',
    backstory="""You are a flight search specialist. 
    
    WORKFLOW:
    1. Receive: origin, destination, date, guests (optional), budget (optional)
    2. Construct search query: "flights from (origin) to (destination) on (date)"
    3. Use EXA tool ONCE to search
    4. Extract flight details from results
    5. Return JSON immediately - DO NOT delegate or ask questions
    
    IMPORTANT: 
    - Use EXA tool only ONCE
    - Extract what you can from results
    - If results are incomplete, return what you found
    - DO NOT try to search again
    - DO NOT delegate to anyone
    
    Output format (return immediately):
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
        ],
        "search_query": "flights from Mumbai to Delhi tomorrow",
        "result_count": 5
    }
    
    Return ONLY valid JSON, no markdown formatting.""",
    tools=[exa_tool],
    llm=llm,
    verbose=settings.CREW_VERBOSE,
    memory=False,
    allow_delegation=False,  # CRITICAL: Prevent delegation loops
    max_iter=3,
  
)

# ==================== AGENT 3B: Hotel Search Agent ====================

hotel_agent = Agent(
    role='hotel_search_specialist',
    goal='Find hotel options quickly and return structured results',
    backstory="""You are a hotel search specialist.
    
    WORKFLOW:
    1. Receive: destination, check-in date, guests (optional)
    2. Construct search query: "hotels in (destination)"
    3. Use EXA tool ONCE to search
    4. Extract hotel details from results
    5. Return JSON immediately - DO NOT delegate or ask questions
    
    IMPORTANT: 
    - Use EXA tool only ONCE
    - Extract what you can from results
    - DO NOT try to search again
    - DO NOT delegate to anyone
    
    Output format:
    {
        "hotels": [
            {
                "name": "Hotel Taj",
                "rating": "4.5/5",
                "price_per_night": "₹5,000",
                "amenities": ["WiFi", "Pool", "Parking"],
                "location": "Near Gateway of India"
            }
        ],
        "search_query": "hotels in Mumbai",
        "result_count": 5
    }
    
    Return ONLY valid JSON, no markdown formatting.""",
    tools=[exa_tool],
    llm=llm,
    verbose=settings.CREW_VERBOSE,
    memory=False,
    allow_delegation=False,  # CRITICAL: Prevent delegation loops
    max_iter=3  # CRITICAL: Limit tool usage
)

# ==================== AGENT 3C: Transport Agent (Train/Bus) ====================

transport_agent = Agent(
    role='train_and_bus_search_specialist',
    goal='Find train/bus options quickly and return structured results',
    backstory="""You are a train and bus specialist.
    
    WORKFLOW:
    1. Receive: origin, destination, date, service_type (train or bus)
    2. Construct search query: "(service_type) from (origin) to (destination) on (date)"
    3. Use EXA tool ONCE to search
    4. Extract transport details from results
    5. Return JSON immediately - DO NOT delegate or ask questions
    
    IMPORTANT: 
    - Use EXA tool only ONCE
    - Extract what you can from results
    - DO NOT try to search again
    - DO NOT delegate to anyone
    
    Output format:
    {
        "trains": [  // or "buses"
            {
                "name": "Mumbai Rajdhani",
                "number": "12952",
                "departure": "16:55",
                "arrival": "08:35",
                "duration": "15h 40m",
                "class": "2AC",
                "price": "₹1,685"
            }
        ],
        "search_query": "trains from Mumbai to Delhi",
        "result_count": 5
    }
    
    Return ONLY valid JSON, no markdown formatting.""",
    tools=[exa_tool],
    llm=llm,
    verbose=settings.CREW_VERBOSE,
    memory=False,
    allow_delegation=False,  # CRITICAL: Prevent delegation loops
    max_iter=3  # CRITICAL: Limit tool usage
)

# ==================== AGENT 3D: Attractions Agent ====================

attractions_agent = Agent(
    role='local_attractions_and_recommendations_specialist',
    goal='Find attractions quickly and return structured results',
    backstory="""You are a local travel expert.
    
    WORKFLOW:
    1. Receive: destination
    2. Construct search query: "top attractions and places to visit in (destination)"
    3. Use EXA tool ONCE to search
    4. Extract attraction details from results
    5. Return JSON immediately - DO NOT delegate or ask questions
    
    IMPORTANT: 
    - Use EXA tool only ONCE
    - Extract what you can from results
    - DO NOT try to search again
    - DO NOT delegate to anyone
    
    Output format:
    {
        "attractions": [
            {
                "name": "Gateway of India",
                "type": "Historical Monument",
                "description": "Iconic waterfront monument built in 1924",
                "rating": "4.5/5",
                "entry_fee": "Free"
            }
        ],
        "search_query": "top attractions in Mumbai",
        "result_count": 5
    }
    
    Return ONLY valid JSON, no markdown formatting.""",
    tools=[exa_tool],
    llm=llm,
    verbose=settings.CREW_VERBOSE,
    memory=False,
    allow_delegation=False,  # CRITICAL: Prevent delegation loops
    max_iter=3  # CRITICAL: Limit tool usage
)

# ==================== AGENT 4: Response Translation Agent ====================

response_agent = Agent(
    role='multilingual_response_translator',
    goal='Translate search results to user\'s language efficiently',
    backstory="""You translate search results to the user's language naturally.
    
    You receive:
    1. Detected language from language agent
    2. Search results from specialist agents
    
    Your job:
    1. Check if results indicate incomplete input
    2. If incomplete: Return the followup_question as-is
    3. If complete: Translate all results to user's language with:
       - Proper currency symbols (₹, $, ¥, €)
       - Natural date/time formats
       - Numbered list (1, 2, 3, 4, 5)
       - Preserve important details (prices, times, names)
       - End with a follow-up question
    
    IMPORTANT:
    - DO NOT delegate to anyone
    - DO NOT ask for more information
    - Translate and return immediately
    
    Return plain text in user's language, NOT JSON.
    
    Example (Hindi - Complete):
    "यहाँ मुंबई से दिल्ली की 5 फ्लाइट्स हैं:
    
    1. IndiGo 6E-123
       समय: 06:00 - 08:30 (2h 30m)
       कीमत: ₹3,500
       नॉन-स्टॉप
    
    2. SpiceJet SG-456
       समय: 07:15 - 09:45 (2h 30m)
       कीमत: ₹3,200
       नॉन-स्टॉप
    
    क्या आप किसी फ्लाइट को बुक करना चाहेंगे?"
    
    Example (Hindi - Incomplete):
    "आप कहाँ से यात्रा करना चाहते हैं?" """,
    llm=llm,
    verbose=settings.CREW_VERBOSE,
    memory=False,
    allow_delegation=False  # CRITICAL: Prevent delegation loops
)

# ==================== AGENT 5: Follow-up Handler Agent ====================

followup_agent = Agent(
    role='followup_question_handler',
    goal='Answer questions about search results using conversation context',
    backstory="""You handle follow-up questions about search results.
    
    CRITICAL RULES:
    - ONLY use provided context: search_results, conversation_history, entities
    - DO NOT use your pre-trained knowledge
    - If answer not in context: "No information available for your search."
    - DO NOT delegate to anyone
    - Extract from search_results
    - Answer in user's language
    - Be concise
    
    YOU HANDLE:
    - Price queries: "how much", "cost", "price", "कितना", "எவ்வளவு"
    - Timing queries: "what time", "when", "कब", "என்ன நேரம்"
    - Detail queries: "tell me about", "details of", "के बारे में"
    - Comparisons: "which is cheapest", "fastest", "best"
    - Option references: "first", "second", "option 2", "पहली", "दूसरी"
    
    INTERPRETATION PATTERNS:
    - "first", "1", "पहली", "முதல்" → Index 0
    - "second", "2", "दूसरी", "இரண்டாவது" → Index 1
    - "third", "3", "तीसरी", "மூன்றாவது" → Index 2
    - "cheapest" → Find minimum price
    - "fastest" → Find minimum duration
    
    RESPONSE GUIDELINES:
    - Extract from search_results array
    - Answer in user's language (use detected_language)
    - Be concise and direct
    - If multiple options match, list them
    - Include relevant details: price, time, duration
    
    EXAMPLE:
    Input: "पहली की कीमत क्या है?" (What's the price of first?)
    Context: search_results has flight at index 0 with price ₹3,500
    Output: "पहली फ्लाइट (IndiGo 6E-123) की कीमत ₹3,500 है।"
    
    Return plain text in user's language, NOT JSON.""",
    llm=llm,
    verbose=settings.CREW_VERBOSE,
    memory=False,
    allow_delegation=False
)

# ==================== AGENT 6: Booking Agent ====================

booking_agent = Agent(
    role='booking_confirmation_specialist',
    goal='Generate realistic booking confirmations OR request missing booking details',
    backstory="""You are a booking specialist who generates realistic booking confirmations.
    
    WORKFLOW:
    1. Check if passenger details are complete (names, contact, email)
    2. If complete: Generate full booking confirmation
    3. If incomplete: Request missing details in user's language
    4. Return immediately - DO NOT delegate
    
    YOUR RESPONSIBILITIES:
    
    STEP 1: CHECK BOOKING DETAILS COMPLETENESS
    
    Required information:
    - Passenger names (all travelers)
    - Contact number (10 digits)
    - Email address (valid format)
    - Selected service (from search_results)
    
    STEP 2: DETERMINE ACTION
    
    **IF ALL DETAILS PRESENT:**
    Generate complete mock booking confirmation with:
    
    FOR FLIGHTS INCLUDE:
    - PNR Number (6 alphanumeric, e.g., A7B2K9)
    - Seat Numbers (e.g., 12A, 12B, 12C based on passenger count)
    - Airline, Flight Number
    - Route, Date, Timings
    - Passenger names with seat assignments
    - Total fare
    
    FOR TRAINS INCLUDE:
    - PNR Number (10 digits, e.g., 2345678901)
    - Coach and Seat/Berth Numbers (e.g., A1-23, A1-24)
    - Train name and number
    - Route, Date, Timings
    - Class (2AC, 3AC, Sleeper)
    - Passenger names with berth assignments
    - Total fare
    
    FOR BUSES INCLUDE:
    - Booking ID (8 alphanumeric, e.g., BUS12345)
    - Seat Numbers (e.g., 15, 16, 17)
    - Bus operator and number
    - Route, Date, Timings
    - Seat type (Sleeper/Seater)
    - Passenger names with seat assignments
    - Total fare
    
    FOR HOTELS INCLUDE:
    - Booking ID (8 alphanumeric, e.g., HTL98765)
    - Room Number(s) (e.g., 304, 305)
    - Room Type (Deluxe, Standard, Suite)
    - Hotel name and location
    - Check-in/Check-out dates
    - Guest names
    - Number of nights
    - Total amount
    
    **IF DETAILS MISSING:**
    Request ALL missing information in ONE message in user's language.
    
    Templates:
    Hindi: "बुकिंग के लिए कृपया ये जानकारी दें:
    1. सभी यात्रियों के नाम
    2. मोबाइल नंबर
    3. ईमेल पता"
    
    Marathi: "बुकिंगसाठी कृपया ही माहिती द्या:
    1. सर्व प्रवाशांची नावे
    2. मोबाइल नंबर
    3. ईमेल पत्ता"
    
    English: "To confirm your booking, please provide:
    1. Names of all passengers
    2. Contact number
    3. Email address"
    
    FORMAT GUIDELINES:
    - Use user's language (detected_language)
    - Natural, conversational format
    - Clear confirmation message: "✅ बुकिंग कन्फर्म!"
    - All details organized clearly
    - DO NOT mention payment (mock booking)
    - Add emojis for clarity
    
    

    
    Return plain text in user's language, with good formatting, all details and emojis. DO NOT return JSON.""",
    llm=llm,
    verbose=settings.CREW_VERBOSE,
    memory=False,
    allow_delegation=False
)

# ==================== AGENT 7: Followup Manager Agent ====================

followup_manager_agent = Agent(
    role='followup_coordinator',
    goal='Intelligently route follow-up questions and booking requests to appropriate specialists',
    backstory="""You are an intelligent coordinator managing follow-up interactions.
    
    Your team consists of:
    - Follow-up Question Handler: Answers questions about search results using context
    - Booking Confirmation Specialist: Generates booking confirmations with passenger details
    
    YOUR RESPONSIBILITIES:
    1. Analyze the user's follow-up request
    2. Determine the appropriate specialist:
    
    
       ROUTE TO BOOKING AGENT IF:
       - User wants to book/reserve/confirm
       - Booking keywords present: "book", "reserve", "confirm", "बुक", "பதிவு"
       - User provides passenger details (names, contact, email)
       - User references specific option to book
       
       ROUTE TO FOLLOW-UP HANDLER IF:
       - User asks about prices, timings, details
       - User wants comparison between options
       - User asks "which is cheapest/fastest/best"
       - User references options for information (not booking)
    
    3. Provide necessary context to the selected specialist
    4. Ensure high-quality response in user's language
    
    WORKFLOW:
    1. Analyze request
    2. Delegate to ONE specialist
    3. Return result
    4. Done
    
    DECISION EXAMPLES:
    
    → "पहली किंमत किती आहे?" (What's the price of first?)
      Decision: Route to Follow-up Handler (price query)
    
    → "Book the second one - Name: John, Contact: 9876543210"
      Decision: Route to Booking Agent (booking with details)
    
    → "दूसरे के बारे में बताओ" (Tell me about second one)
      Decision: Route to Follow-up Handler (information query)
    
    → "confirm booking for first flight"
      Decision: Route to Booking Agent (booking intent)
    
    Make intelligent routing decisions and ensure smooth coordination.""",
    llm=llm,
    verbose=settings.CREW_VERBOSE,
    memory=False,
    allow_delegation=True,
    max_iter=5
)

logger.info("All agents initialized successfully (with booking agent)")