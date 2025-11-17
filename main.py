# main.py
"""
Multi-Lingual Travel Assistant - FastAPI Application
OPTIMIZED: Works with simplified database (3 tables only)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid
from typing import Optional
import json
import re

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
    description="AI-powered travel assistant with optimized memory storage",
    version="2.1.0"
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
    agents_called: Optional[list] = None
    status: str = "success"

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Multi-Lingual Travel Assistant API",
        "version": "2.1.0",
        "status": "running",
        "memory_system": "Optimized SQLite (3 tables only)"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "travel-assistant",
        "database_tables": ["sessions", "messages", "agent_outputs"],
        "memory_system": "optimized"
    }

def extract_json_from_text(text: str) -> Optional[dict]:
    """Extract JSON from text that might contain markdown or other content"""
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

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with optimized memory storage"""
    try:
        # Generate or retrieve session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        logger.info("="*80)
        logger.info(f"🚀 NEW REQUEST - Session: {session_id}")
        logger.info(f"📝 Message: {request.message}")
        logger.info(f"🔄 Is follow-up: {request.is_followup}")
        logger.info("="*80)
        
        # Create session if new
        memory.create_session(session_id)
        
        # Store user message with metadata
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
        
        if request.is_followup:
            # ==================== FOLLOW-UP MODE ====================
            logger.info("📌 MODE: Follow-up query")
            
            # Get full context (extracts data from agent_outputs automatically)
            context = memory.get_full_context(session_id)
            
            logger.info(f"📚 Context loaded: "
                       f"Language={context['language']}, "
                       f"Entities={len(context['entities'])}, "
                       f"SearchResults={len(context['search_results'])}, "
                       f"AgentOutputs={len(context['agent_outputs'])}")
            
            # Create follow-up crew
            crew = create_travel_crew(is_followup=True, context_data=context)
            
            # Prepare inputs with context
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
            
            agents_called.append("follow-up handler")
            
        else:
            # ==================== INITIAL MODE ====================
            logger.info("📌 MODE: Initial query")
            
            # Create initial crew
            crew = create_travel_crew(is_followup=False)
            
            # Prepare inputs
            inputs = {
                "user_input": request.message,
                "session_id": session_id
            }
        
        # Execute crew
        logger.info("⚙️  Executing crew...")
        result = crew.kickoff(inputs=inputs)
        logger.info("✅ Crew execution completed")
        
        # ============ CAPTURE AND STORE AGENT OUTPUTS ============
        if hasattr(result, 'tasks_output') and result.tasks_output:
            logger.info(f"💾 Capturing {len(result.tasks_output)} agent outputs...")
            
            for idx, task_output in enumerate(result.tasks_output):
                try:
                    # Extract agent information
                    agent_name = str(task_output.agent) if hasattr(task_output, 'agent') else f"agent_{idx}"
                    task_name = f"task_{agent_name.split()[0].lower() if ' ' in agent_name else idx}"
                    
                    # Get output data
                    output_data = None
                    if hasattr(task_output, 'raw'):
                        output_data = task_output.raw
                    elif hasattr(task_output, 'output'):
                        output_data = task_output.output
                    else:
                        output_data = str(task_output)
                    
                    # Try to parse JSON
                    output_type = "text"
                    if isinstance(output_data, dict):
                        output_type = "json"
                    elif isinstance(output_data, str):
                        parsed_json = extract_json_from_text(output_data)
                        if parsed_json:
                            output_data = parsed_json
                            output_type = "json"
                    
                    # Store in database
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
        
        # Extract final response
        response_text = str(result.raw) if hasattr(result, 'raw') else str(result)
        
        # Get detected language from context (extracted from agent_outputs)
        detected_lang = None
        if not request.is_followup:
            context = memory.get_full_context(session_id)
            if context['language']:
                detected_lang = context['language']['detected_language']
        
        # Store assistant response
        memory.add_message(
            session_id=session_id,
            role="assistant",
            content=response_text,
            metadata={
                "detected_language": detected_lang,
                "is_followup": request.is_followup,
                "agents_called": agents_called
            }
        )
        
        logger.info(f"✅ Request completed - {len(agents_called)} agents called")
        
        return ChatResponse(
            session_id=session_id,
            response=response_text,
            detected_language=detected_lang,
            is_followup=request.is_followup,
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
    """Get complete session information"""
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

@app.get("/api/session/{session_id}/agents")
async def get_session_agents(session_id: str):
    """Get all agent outputs for a session"""
    try:
        agent_outputs = memory.get_agent_outputs(session_id)
        
        if not agent_outputs:
            raise HTTPException(status_code=404, detail="No agent outputs found")
        
        return {
            "session_id": session_id,
            "agent_outputs": agent_outputs,
            "total_agents_called": len(agent_outputs)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving agent outputs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}/stats")
async def get_session_statistics(session_id: str):
    """Get session statistics"""
    try:
        stats = memory.get_session_stats(session_id)
        
        if not stats.get('created_at'):
            raise HTTPException(status_code=404, detail="Session not found")
        
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session stats: {e}")
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

@app.post("/api/cleanup")
async def cleanup_old_sessions(days: int = 30):
    """Cleanup sessions older than specified days"""
    try:
        deleted = memory.cleanup_old_sessions(days)
        return {
            "message": f"Cleaned up {deleted} old sessions",
            "deleted_count": deleted
        }
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application...")
    memory.close()
    logger.info("Memory manager closed")

if __name__ == "__main__":
    import uvicorn
    from datetime import datetime
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )