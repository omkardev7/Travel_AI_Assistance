# app.py


import streamlit as st
import requests
import json
from datetime import datetime
import uuid

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Multi-Lingual Travel Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .session-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f5f5f5;
        border: 2px solid #2196F3;
        margin-bottom: 0.5rem;
    }
    .session-card:hover {
        background-color: #e3f2fd;
        border-color: #1976D2;
    }
    .active-session {
        background-color: #e8f5e9;
        border-color: #4CAF50;
    }
    .stat-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .mode-indicator {
        padding: 0.5rem 1rem;
        border-radius: 1rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .mode-initial {
        background-color: #e3f2fd;
        color: #1976D2;
    }
    .mode-followup {
        background-color: #fff3e0;
        color: #f57c00;
    }
    .session-id-display {
        background-color: #e8f5e9;
        padding: 0.8rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .toggle-container {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'active_session_id' not in st.session_state:
    st.session_state.active_session_id = None
if 'available_sessions' not in st.session_state:
    st.session_state.available_sessions = []
if 'followup_mode_enabled' not in st.session_state:
    st.session_state.followup_mode_enabled = False

def fetch_session_details(session_id):
    """Fetch session details from backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/session/{session_id}", timeout=300)  # 5 minutes
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching session: {e}")
        return None

def create_new_session():
    """Create a new session"""
    session_id = str(uuid.uuid4())
    
    # Add to available sessions
    if session_id not in st.session_state.available_sessions:
        st.session_state.available_sessions.append(session_id)
    
    # Set as active
    st.session_state.active_session_id = session_id
    st.session_state.followup_mode_enabled = False  # Start with follow-up disabled
    
    return session_id

def load_session(session_id):
    """Load an existing session"""
    st.session_state.active_session_id = session_id
    # Don't automatically enable follow-up mode, let user control it

def send_message(message, session_id, is_followup):
    """Send message to API with 5-minute timeout"""
    try:
        payload = {
            "session_id": session_id,
            "message": message,
            "is_followup": is_followup
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/chat",
            json=payload,
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out after 5 minutes. Please try again.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

# Sidebar
with st.sidebar:
    st.title("🌍 Travel Assistant")
    st.markdown("---")
    
    # Session Management
    st.subheader("📋 Session Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New Session", use_container_width=True, type="primary"):
            create_new_session()
            st.success(f"✅ New session created!")
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.active_session_id = None
            st.session_state.followup_mode_enabled = False
            st.rerun()
    
    # Load existing session
    st.markdown("### 🔍 Load Existing Session")
    session_id_input = st.text_input(
        "Enter Session ID:",
        placeholder="e.g., 550e8400-e29b-41d4...",
        key="session_id_input"
    )
    
    if st.button("📂 Load Session", use_container_width=True):
        if session_id_input:
            # Verify session exists in backend
            session_details = fetch_session_details(session_id_input)
            
            if session_details:
                # Add to available sessions if not already there
                if session_id_input not in st.session_state.available_sessions:
                    st.session_state.available_sessions.append(session_id_input)
                
                # Load the session
                load_session(session_id_input)
                st.success(f"✅ Session loaded successfully!")
                st.rerun()
            else:
                st.error("❌ Session not found in database!")
        else:
            st.warning("⚠️ Please enter a session ID")
    
    st.markdown("---")
    
    # FOLLOW-UP MODE TOGGLE
    if st.session_state.active_session_id:
        st.subheader("🔧 Query Mode")
        
        st.markdown('<div class="toggle-container">', unsafe_allow_html=True)
        
        # Toggle switch for follow-up mode
        followup_enabled = st.toggle(
            "Enable Follow-up Mode",
            value=st.session_state.followup_mode_enabled,
            key="followup_toggle",
            help="Enable this to ask follow-up questions about existing search results. Disable for new queries."
        )
        
        # Update state if changed
        if followup_enabled != st.session_state.followup_mode_enabled:
            st.session_state.followup_mode_enabled = followup_enabled
            st.rerun()
        
        # Show current mode
        if st.session_state.followup_mode_enabled:
            st.markdown("""
            <div style='color: #f57c00; font-weight: bold; margin-top: 0.5rem;'>
                🔄 Follow-up Mode: ENABLED<br/>
                <span style='font-size: 0.85rem; font-weight: normal;'>
                is_followup = True
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='color: #1976D2; font-weight: bold; margin-top: 0.5rem;'>
                🆕 New Query Mode: ENABLED<br/>
                <span style='font-size: 0.85rem; font-weight: normal;'>
                is_followup = False
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Show session info
        st.subheader("📊 Session Info")
        st.markdown(f"""
        <div class="session-id-display">
            <strong>Active Session ID:</strong><br/>
            {st.session_state.active_session_id}
        </div>
        """, unsafe_allow_html=True)
        
        # Fetch and show session stats
        session_details = fetch_session_details(st.session_state.active_session_id)
        if session_details:
            stats = session_details.get("stats", {})
            col1, col2 = st.columns(2)
            with col1:
                st.metric("💬 Messages", stats.get("message_count", 0))
            with col2:
                st.metric("🤖 Agents", stats.get("agent_call_count", 0))
            
            # Show language if detected
            if session_details.get("language"):
                lang = session_details["language"]
                st.info(f"🗣️ **{lang.get('language_name', 'Unknown')}**")
    
    st.markdown("---")
    
    # Available Sessions (for reference)
    if st.session_state.available_sessions:
        with st.expander("📂 Recent Sessions"):
            for sess_id in reversed(st.session_state.available_sessions[-10:]):  # Show last 10
                is_active = sess_id == st.session_state.active_session_id
                
                if is_active:
                    st.markdown(f"**✅ {sess_id[:16]}...**")
                else:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(f"📄 {sess_id[:16]}...")
                    with col2:
                        if st.button("Load", key=f"load_{sess_id[:8]}", use_container_width=True):
                            load_session(sess_id)
                            st.rerun()
    
    st.markdown("---")
    
    # View details button
    if st.session_state.active_session_id:
        if st.button("📊 View Details", use_container_width=True):
            st.session_state.show_details = True
    
    st.markdown("---")
    
    # Language support
    st.subheader("🗣️ Languages")
    st.markdown("""
    - 🇮🇳 Hindi • Marathi
    - 🇮🇳 Tamil • Bengali
    - 🇬🇧 English
    """)
    
    st.markdown("---")
    
    # Services
    st.subheader("✈️ Services")
    st.markdown("""
    ✈️ Flights • 🏨 Hotels
    🚆 Trains • 🚌 Buses
    🎭 Attractions • 📋 Booking
    """)
    
    st.markdown("---")
    
    # Timeout info
    st.caption("⏱️ Request timeout: 5 minutes")

# Main content
st.title("✈️ Multi-Lingual Travel Assistant")

# Display current mode and session status
if st.session_state.active_session_id:
    col1, col2 = st.columns([4, 1])
    
    with col1:
        if st.session_state.followup_mode_enabled:
            st.markdown('<div class="mode-indicator mode-followup">🔄 Follow-up Mode (is_followup=True)</div>', unsafe_allow_html=True)
            st.info("💬 **Follow-up mode enabled**: Ask questions about previous search results or proceed with booking.")
        else:
            st.markdown('<div class="mode-indicator mode-initial">🆕 New Query Mode (is_followup=False)</div>', unsafe_allow_html=True)
            st.info("🎯 **New query mode enabled**: Start a fresh travel search.")
        
        st.caption(f"📝 Session ID: `{st.session_state.active_session_id}`")
    
    with col2:
        # Quick toggle button
        if st.button("🔄 Toggle Mode", type="secondary", use_container_width=True):
            st.session_state.followup_mode_enabled = not st.session_state.followup_mode_enabled
            st.rerun()
else:
    st.warning("⚠️ No active session. Please create a new session or load an existing one from the sidebar.")

# Example queries
with st.expander("💡 Example Queries"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **New Query Mode (is_followup=False):**
        - "मला उद्या मुंबई ते दिल्ली फ्लाइट हवी 2 जणांसाठी"
        - "I need a hotel in Goa for Dec 25-27"
        - "Trains from Pune to Chennai on Dec 10"
        - "मला १० डिसेंबर २०२५ रोजी पुण्याहून चेन्नईला जायचे आहे"
        - "Find buses from Delhi to Jaipur tomorrow"
        """)
    
    with col2:
        st.markdown("""
        **Follow-up Mode (is_followup=True):**
        - "पहिली किती आहे?" (What's the price of first?)
        - "CSMT MS SF मेल साठी तिकीट बुक करा"
        - "Show me details of option 2"
        - "Book the third one - Name: John Doe, Contact: 9876543210, Email: john@example.com"
        - "सबसे सस्ती कौन सी है?"
        """)

# Display chat messages if session is loaded
if st.session_state.active_session_id:
    session_details = fetch_session_details(st.session_state.active_session_id)
    
    if session_details and session_details.get("conversation_history"):
        st.markdown("### 💬 Conversation History")
        
        for message in session_details["conversation_history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show metadata
                if message.get("metadata"):
                    metadata = message["metadata"]
                    
                    # Show mode that was used
                    if message["role"] == "user":
                        is_followup = metadata.get("is_followup", False)
                        mode_text = "Follow-up" if is_followup else "New Query"
                        mode_color = "#f57c00" if is_followup else "#1976D2"
                        st.markdown(f"<span style='color: {mode_color}; font-size: 0.8rem;'>📌 {mode_text} Mode</span>", unsafe_allow_html=True)
                    
                    # Show detected language
                    if metadata.get("detected_language"):
                        st.caption(f"🗣️ Language: {metadata['detected_language']}")
                    
                    # Show booking status
                    if metadata.get("is_booking"):
                        st.success("✅ Booking Confirmed!")
                    
                    # Show completeness
                    if message["role"] == "assistant" and not metadata.get("is_complete", True):
                        st.warning("⚠️ Incomplete information - Please provide all details")
                    
                    # Show agents
                    if metadata.get("agents_called"):
                        st.caption(f"🤖 Agents: {', '.join(metadata['agents_called'][:3])}")

# Chat input - only allow if session exists
if st.session_state.active_session_id:
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Use the current follow-up mode setting
        is_followup = st.session_state.followup_mode_enabled
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
            mode_text = "Follow-up" if is_followup else "New Query"
            mode_color = "#f57c00" if is_followup else "#1976D2"
            st.markdown(f"<span style='color: {mode_color}; font-size: 0.8rem;'>📌 {mode_text} Mode (is_followup={is_followup})</span>", unsafe_allow_html=True)
        
        # Show loading spinner
        with st.chat_message("assistant"):
            with st.spinner("🤖 Processing your request... (timeout: 5 minutes)"):
                # Send to API
                response_data = send_message(
                    message=user_input,
                    session_id=st.session_state.active_session_id,
                    is_followup=is_followup
                )
                
                if response_data:
                    # Display response
                    st.markdown(response_data["response"])
                    
                    # Show metadata
                    if response_data.get("detected_language"):
                        st.caption(f"🗣️ Language: {response_data['detected_language']}")
                    
                    if response_data.get("is_booking"):
                        st.success("✅ Booking Confirmed!")
                    
                    if response_data.get("agents_called"):
                        st.caption(f"🤖 Agents: {', '.join(response_data['agents_called'][:3])}")
                    
                    # Add session to available sessions if new
                    if st.session_state.active_session_id not in st.session_state.available_sessions:
                        st.session_state.available_sessions.append(st.session_state.active_session_id)
                    
                    # Suggest enabling follow-up mode after first successful query
                    if not is_followup and response_data.get("is_complete", True) and not st.session_state.followup_mode_enabled:
                        st.info("💡 **Tip**: You can now enable **Follow-up Mode** in the sidebar to ask questions about these results or proceed with booking.")
                    
                    # Rerun to refresh conversation history
                    st.rerun()
else:
    st.info("💡 Please create a new session or load an existing one from the sidebar to start chatting.")

# Session details modal
if st.session_state.get('show_details', False) and st.session_state.active_session_id:
    st.markdown("---")
    st.subheader("📊 Detailed Session Information")
    
    session_details = fetch_session_details(st.session_state.active_session_id)
    
    if session_details:
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        
        stats = session_details.get("stats", {})
        
        with col1:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.metric("📧 Messages", stats.get("message_count", 0))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.metric("🤖 Agent Calls", stats.get("agent_call_count", 0))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            if session_details.get("language"):
                st.metric("🗣️ Language", session_details["language"]["language_name"])
            else:
                st.metric("🗣️ Language", "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.metric("🔍 Results", len(session_details.get("search_results", [])))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📍 Entities", 
            "🔍 Search Results", 
            "💬 Full Conversation",
            "🤖 Agent Outputs"
        ])
        
        with tab1:
            if session_details.get("entities"):
                for key, value in session_details["entities"].items():
                    if value:
                        st.markdown(f"**{key.title()}:** {value}")
            else:
                st.info("No entities extracted yet")
        
        with tab2:
            if session_details.get("search_results"):
                for idx, result in enumerate(session_details["search_results"]):
                    st.markdown(f"### Result Set {idx + 1} - {result['service_type'].title()}")
                    st.markdown(f"*Timestamp: {result['timestamp']}*")
                    
                    if isinstance(result["results"], list) and len(result["results"]) > 0:
                        st.json(result["results"])
                    
                    st.markdown("---")
            else:
                st.info("No search results yet")
        
        with tab3:
            if session_details.get("conversation_history"):
                for msg in session_details["conversation_history"]:
                    role_icon = "👤" if msg["role"] == "user" else "🤖"
                    st.markdown(f"**{role_icon} {msg['role'].title()}** - *{msg['timestamp']}*")
                    
                    # Show mode if available
                    if msg.get("metadata", {}).get("is_followup") is not None:
                        mode = "Follow-up" if msg["metadata"]["is_followup"] else "New Query"
                        st.caption(f"Mode: {mode}")
                    
                    st.markdown(msg["content"])
                    st.markdown("---")
            else:
                st.info("No conversation history yet")
        
        with tab4:
            if session_details.get("agent_outputs"):
                for output in session_details["agent_outputs"]:
                    with st.expander(f"🤖 {output['agent_name']} - {output['timestamp']}"):
                        st.markdown(f"**Task:** {output['task_name']}")
                        st.markdown(f"**Type:** {output['output_type']}")
                        
                        if output['output_type'] == 'json':
                            st.json(output['output_data'])
                        else:
                            st.code(output['output_data'], language="text")
            else:
                st.info("No agent outputs yet")
        
        if st.button("❌ Close Details", use_container_width=True):
            st.session_state.show_details = False
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>Multi-Lingual Travel Assistant v2.4.0</strong></p>
    <p>Powered by CrewAI • FastAPI • Gemini • Streamlit</p>
    <p style='font-size: 0.8rem; margin-top: 0.5rem;'>
        💡 <strong>Tip:</strong> Toggle follow-up mode to switch between new queries and follow-up questions
    </p>
    <p style='font-size: 0.75rem; color: #999;'>
        ⏱️ API Timeout: 5 minutes • Session-based conversation management
    </p>
</div>
""", unsafe_allow_html=True)