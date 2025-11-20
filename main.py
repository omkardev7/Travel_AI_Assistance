from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid
from typing import Optional
import json
import re
from datetime import datetime

from crew import create_travel_crew, kickoff_crew
from config import settings
from logger import setup_logger
from memory_manager import get_memory_manager

# Setup logging
logger = setup_logger(__name__)

# Get memory manager singleton
memory = get_memory_manager()

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Lingual Travel Assistant",
    description="AI-powered travel assistant with booking functionality",
    version="2.4.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    message: str = Field(..., description="User message in any language")
    is_followup: bool = Field(False, description="Whether this is a follow-up question")

class ChatResponse(BaseModel):
    session_id: str
    response: str
    detected_language: Optional[str] = None
    is_followup: bool
    is_booking: bool = False
    is_complete: bool = True
    agents_called: Optional[list] = None
    status: str = "success"

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Multi-Lingual Travel Assistant API",
        "version": "2.4.0",
        "status": "running",
        "features": [
            "Hierarchical routing",
            "Multi-language support",
            "Mock booking confirmations",
            "Context-aware conversations"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "travel-assistant",
        "process_type": "hierarchical",
        "memory_system": "optimized SQLite"
    }

def extract_json_from_text(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in markdown code blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL)
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        
        # Try to find any JSON object
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    return None

def merge_entities_from_context(session_id: str, new_message: str) -> str:

    try:
        context = memory.get_full_context(session_id)
        
        if context['conversation_history']:
            last_messages = context['conversation_history'][-2:]
            
            if len(last_messages) >= 2:
                last_assistant_msg = None
                for msg in reversed(last_messages):
                    if msg['role'] == 'assistant' and msg.get('metadata', {}).get('is_complete') == False:
                        last_assistant_msg = msg
                        break
                
                if last_assistant_msg and context['entities']:
                    entities = context['entities']
                    context_parts = []
                    
                    if entities.get('origin'):
                        context_parts.append(f"Origin: {entities['origin']}")
                    if entities.get('destination'):
                        context_parts.append(f"Destination: {entities['destination']}")
                    if entities.get('date'):
                        context_parts.append(f"Date: {entities['date']}")
                    if entities.get('service_type'):
                        context_parts.append(f"Service: {entities['service_type']}")
                    
                    if context_parts:
                        context_str = " | ".join(context_parts)
                        enhanced_message = f"[Previous context: {context_str}] {new_message}"
                        logger.info(f"Enhanced message with context: {enhanced_message}")
                        return enhanced_message
        
        return new_message
    except Exception as e:
        logger.warning(f"Could not merge context: {e}")
        return new_message

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint - SIMPLIFIED"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        logger.info("="*80)
        logger.info(f"🚀 NEW REQUEST - Session: {session_id}")
        logger.info(f"📝 Message: {request.message}")
        logger.info(f"🔄 Is follow-up: {request.is_followup}")
        logger.info("="*80)
        
        memory.create_session(session_id)
        
        memory.add_message(
            session_id=session_id,
            role="user",
            content=request.message,
            metadata={
                "is_followup": request.is_followup,
                "timestamp": str(datetime.now())
            }
        )
        
        agents_called = []
        is_complete = True
        is_booking = False  # Will be detected from agent outputs
        
        if request.is_followup:
            # ==================== FOLLOW-UP MODE ====================
            logger.info("📌 MODE: Follow-up query (manager decides routing)")
            
            # Get full context
            context = memory.get_full_context(session_id)
            
            # Create hierarchical crew (manager decides follow-up vs booking)
            crew = create_travel_crew(is_followup=True)
            
            # Prepare inputs
            inputs = {
                "user_followup_input": request.message,
                "session_id": session_id,
                "detected_language": context['language']['detected_language'] if context['language'] else "en",
                "language_name": context['language']['language_name'] if context['language'] else "English",
                "entities": json.dumps(context['entities'], ensure_ascii=False),
                "search_results": json.dumps(context['search_results'], ensure_ascii=False),
                "conversation_history": json.dumps(context['conversation_history'][-5:], ensure_ascii=False),
                "agent_outputs": json.dumps(context['agent_outputs'], ensure_ascii=False)
            }
            
        else:
            # ==================== INITIAL MODE ====================
            logger.info("📌 MODE: Initial query (hierarchical)")
            
            enhanced_message = merge_entities_from_context(session_id, request.message)
            
            crew = create_travel_crew(is_followup=False)
            
            inputs = {
                "user_input": enhanced_message,
                "session_id": session_id
            }
        
        # Execute crew
        logger.info("⚙️  Executing crew...")
        result = crew.kickoff(inputs=inputs)
        logger.info("✅ Crew execution completed")
        
        # Capture agent outputs
        if hasattr(result, 'tasks_output') and result.tasks_output:
            logger.info(f"💾 Capturing {len(result.tasks_output)} agent outputs...")
            
            for idx, task_output in enumerate(result.tasks_output):
                try:
                    agent_name = str(task_output.agent) if hasattr(task_output, 'agent') else f"agent_{idx}"
                    task_name = f"task_{agent_name.split()[0].lower() if ' ' in agent_name else idx}"
                    
                    output_data = None
                    if hasattr(task_output, 'raw'):
                        output_data = task_output.raw
                    elif hasattr(task_output, 'output'):
                        output_data = task_output.output
                    else:
                        output_data = str(task_output)
                    
                    output_type = "text"
                    if isinstance(output_data, dict):
                        output_type = "json"
                    elif isinstance(output_data, str):
                        parsed_json = extract_json_from_text(output_data)
                        if parsed_json:
                            output_data = parsed_json
                            output_type = "json"
                    
                    # Detect booking from agent name
                    if 'booking' in agent_name.lower():
                        is_booking = True
                        logger.info("🎫 Booking agent was called - this is a booking confirmation")
                    
                    # Check completeness (only for initial query)
                    if idx == 0 and not request.is_followup and output_type == "json" and isinstance(output_data, dict):
                        if not output_data.get('is_complete', True):
                            is_complete = False
                            logger.info("⚠️  Input is INCOMPLETE - follow-up question required")
                    
                    memory.store_agent_output(
                        session_id=session_id,
                        agent_name=agent_name,
                        task_name=task_name,
                        output_data=output_data,
                        output_type=output_type
                    )
                    
                    agents_called.append(agent_name.split()[0] if ' ' in agent_name else agent_name)
                    
                except Exception as e:
                    logger.warning(f"Could not process task {idx}: {e}")
                    continue
            
            logger.info(f"✅ Stored {len(result.tasks_output)} agent outputs")
        
        response_text = str(result.raw) if hasattr(result, 'raw') else str(result)
        
        detected_lang = None
        if not request.is_followup:
            context = memory.get_full_context(session_id)
            if context['language']:
                detected_lang = context['language']['detected_language']
        
        memory.add_message(
            session_id=session_id,
            role="assistant",
            content=response_text,
            metadata={
                "detected_language": detected_lang,
                "is_followup": request.is_followup,
                "is_booking": is_booking,
                "is_complete": is_complete,
                "agents_called": agents_called
            }
        )
        
        logger.info(f"✅ Request completed - Booking: {is_booking}, Complete: {is_complete}, Agents: {len(agents_called)}")
        
        return ChatResponse(
            session_id=session_id,
            response=response_text,
            detected_language=detected_lang,
            is_followup=request.is_followup,
            is_booking=is_booking,
            is_complete=is_complete,
            agents_called=agents_called,
            status="success"
        )
    
    except Exception as e:
        logger.error(f"❌ Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    try:
        context = memory.get_full_context(session_id)
        
        if not context['conversation_history']:
            raise HTTPException(status_code=404, detail="Session not found")
        
        stats = memory.get_session_stats(session_id)
        
        return {
            "session_id": session_id,
            "language": context['language'],
            "entities": context['entities'],
            "conversation_history": context['conversation_history'],
            "search_results": context['search_results'],
            "agent_outputs": context['agent_outputs'],
            "stats": stats
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    try:
        success = memory.clear_session(session_id)
        if success:
            return {"message": "Session deleted successfully"}
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():

    logger.info("Shutting down application...")
    memory.close()
    logger.info("Memory manager closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )