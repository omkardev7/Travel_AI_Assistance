import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from logger import setup_logger

logger = setup_logger(__name__)


class TravelMemoryManager:

    def __init__(self, db_path: str = "travel_memory.db"):
        
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
        logger.info(f"Memory manager initialized with database: {db_path}")
    
    def _initialize_database(self):
        """Create only essential database tables"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        cursor = self.conn.cursor()
        
        # ==================== TABLE 1: Sessions ====================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)
        
        # ==================== TABLE 2: Messages ====================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # ==================== TABLE 3: Agent Outputs ====================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_outputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                task_name TEXT NOT NULL,
                output_type TEXT NOT NULL,
                output_data TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Create indexes for better query performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session 
            ON messages(session_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_outputs_session 
            ON agent_outputs(session_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_outputs_agent 
            ON agent_outputs(session_id, agent_name)
        """)
        
        self.conn.commit()
        logger.info("Database initialized with 3 essential tables: sessions, messages, agent_outputs")
    
    def create_session(self, session_id: str, metadata: Optional[Dict] = None) -> bool:

        try:
            cursor = self.conn.cursor()
            metadata_json = json.dumps(metadata) if metadata else "{}"
            
            cursor.execute("""
                INSERT OR IGNORE INTO sessions (session_id, metadata)
                VALUES (?, ?)
            """, (session_id, metadata_json))
            
            self.conn.commit()
            logger.info(f"Session created: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return False
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> bool:

        try:
            cursor = self.conn.cursor()
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute("""
                INSERT INTO messages (session_id, role, content, metadata)
                VALUES (?, ?, ?, ?)
            """, (session_id, role, content, metadata_json))
            
            # Update session last activity
            cursor.execute("""
                UPDATE sessions SET last_activity = CURRENT_TIMESTAMP
                WHERE session_id = ?
            """, (session_id,))
            
            self.conn.commit()
            logger.debug(f"Message added to session {session_id}: {role}")
            return True
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            return False
    
    def store_agent_output(
        self,
        session_id: str,
        agent_name: str,
        task_name: str,
        output_data: Any,
        output_type: str = "json"
    ) -> bool:

        try:
            cursor = self.conn.cursor()
            
            # Convert to string for storage
            if isinstance(output_data, (dict, list)):
                output_str = json.dumps(output_data, ensure_ascii=False)
                output_type = "json"
            else:
                output_str = str(output_data)
                output_type = "text" if output_type != "json" else "json"
            
            cursor.execute("""
                INSERT INTO agent_outputs 
                (session_id, agent_name, task_name, output_type, output_data)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, agent_name, task_name, output_type, output_str))
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error storing agent output: {e}")
            return False
    
    def get_agent_outputs(
        self,
        session_id: str,
        agent_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:

        try:
            cursor = self.conn.cursor()
            
            if agent_name:
                cursor.execute("""
                    SELECT agent_name, task_name, output_type, output_data, timestamp
                    FROM agent_outputs
                    WHERE session_id = ? AND agent_name = ?
                    ORDER BY timestamp ASC
                """, (session_id, agent_name))
            else:
                cursor.execute("""
                    SELECT agent_name, task_name, output_type, output_data, timestamp
                    FROM agent_outputs
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                """, (session_id,))
            
            rows = cursor.fetchall()
            
            outputs = []
            for row in rows:
                output_dict = {
                    'agent_name': row['agent_name'],
                    'task_name': row['task_name'],
                    'output_type': row['output_type'],
                    'timestamp': row['timestamp']
                }
                
                # Parse JSON if applicable
                if row['output_type'] == 'json':
                    try:
                        output_dict['output_data'] = json.loads(row['output_data'])
                    except json.JSONDecodeError:
                        output_dict['output_data'] = row['output_data']
                else:
                    output_dict['output_data'] = row['output_data']
                
                outputs.append(output_dict)
            
            return outputs
        except Exception as e:
            logger.error(f"Error retrieving agent outputs: {e}")
            return []
    
    def get_latest_agent_output(
        self,
        session_id: str,
        agent_name: str
    ) -> Optional[Dict[str, Any]]:

        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT agent_name, task_name, output_type, output_data, timestamp
                FROM agent_outputs
                WHERE session_id = ? AND agent_name = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (session_id, agent_name))
            
            row = cursor.fetchone()
            
            if row:
                output_dict = {
                    'agent_name': row['agent_name'],
                    'task_name': row['task_name'],
                    'output_type': row['output_type'],
                    'timestamp': row['timestamp']
                }
                
                if row['output_type'] == 'json':
                    try:
                        output_dict['output_data'] = json.loads(row['output_data'])
                    except json.JSONDecodeError:
                        output_dict['output_data'] = row['output_data']
                else:
                    output_dict['output_data'] = row['output_data']
                
                return output_dict
            return None
        except Exception as e:
            logger.error(f"Error retrieving latest agent output: {e}")
            return None
    
    def get_conversation_history(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:

        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT role, content, metadata, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, limit))
            
            rows = cursor.fetchall()
            
            messages = []
            for row in rows:
                message = {
                    'role': row['role'],
                    'content': row['content'],
                    'timestamp': row['timestamp']
                }
                
                # Parse metadata if exists
                if row['metadata']:
                    try:
                        message['metadata'] = json.loads(row['metadata'])
                    except json.JSONDecodeError:
                        message['metadata'] = {}
                else:
                    message['metadata'] = {}
                
                messages.append(message)
            
            # Reverse to get chronological order
            return list(reversed(messages))
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []
    
    def get_full_context(self, session_id: str) -> Dict[str, Any]:

        # Get raw data
        agent_outputs = self.get_agent_outputs(session_id)
        conversation_history = self.get_conversation_history(session_id, limit=10)
        
        # Extract structured information from agent outputs
        language_context = None
        entities = {}
        search_results = []
        
        for output in agent_outputs:
            agent_name_lower = output['agent_name'].lower()
            
            # Extract language context from Language Agent
            if ('language' in agent_name_lower or 'detection' in agent_name_lower) and output['output_type'] == 'json':
                data = output['output_data']
                if isinstance(data, dict):
                    language_context = {
                        'detected_language': data.get('detected_language'),
                        'language_name': data.get('language_name')
                    }
                    # Extract entities
                    if 'entities' in data and isinstance(data['entities'], dict):
                        entities = data['entities']
            
            # Extract search results from Search Agents
            if output['output_type'] == 'json' and isinstance(output['output_data'], dict):
                data = output['output_data']
                
                # Check for different result types
                for key in ['flights', 'hotels', 'trains', 'buses', 'attractions']:
                    if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                        # Determine service type
                        service_type_map = {
                            'flights': 'flight',
                            'hotels': 'hotel',
                            'trains': 'transport',
                            'buses': 'transport',
                            'attractions': 'attractions'
                        }
                        service_type = service_type_map.get(key, 'unknown')
                        
                        search_results.append({
                            'service_type': service_type,
                            'results': data[key],
                            'timestamp': output['timestamp']
                        })
                        break
        
        context = {
            'session_id': session_id,
            'language': language_context,
            'entities': entities,
            'search_results': search_results,
            'conversation_history': conversation_history,
            'agent_outputs': agent_outputs
        }
        
        logger.info(f"Full context retrieved for {session_id}: "
                   f"{len(agent_outputs)} agent outputs, "
                   f"{len(conversation_history)} messages, "
                   f"{len(search_results)} search result sets")
        
        return context
    
    def clear_session(self, session_id: str) -> bool:

        try:
            cursor = self.conn.cursor()
            
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM agent_outputs WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            
            self.conn.commit()
            logger.info(f"Session cleared: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return False
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:

        try:
            cursor = self.conn.cursor()
            
            # Get message count
            cursor.execute("""
                SELECT COUNT(*) as count FROM messages WHERE session_id = ?
            """, (session_id,))
            message_count = cursor.fetchone()['count']
            
            # Get agent output count
            cursor.execute("""
                SELECT COUNT(*) as count FROM agent_outputs WHERE session_id = ?
            """, (session_id,))
            agent_count = cursor.fetchone()['count']
            
            # Get session info
            cursor.execute("""
                SELECT created_at, last_activity FROM sessions WHERE session_id = ?
            """, (session_id,))
            session_info = cursor.fetchone()
            
            return {
                'session_id': session_id,
                'message_count': message_count,
                'agent_call_count': agent_count,
                'created_at': session_info['created_at'] if session_info else None,
                'last_activity': session_info['last_activity'] if session_info else None
            }
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {}
    
    def cleanup_old_sessions(self, days: int = 30) -> int:

        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT session_id FROM sessions 
                WHERE last_activity < datetime('now', '-' || ? || ' days')
            """, (days,))
            
            old_sessions = [row['session_id'] for row in cursor.fetchall()]
            
            for session_id in old_sessions:
                self.clear_session(session_id)
            
            logger.info(f"Cleaned up {len(old_sessions)} old sessions")
            return len(old_sessions)
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}")
            return 0
    
    def close(self):

        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __del__(self):

        self.close()


# Singleton instance
_memory_manager = None


def get_memory_manager() -> TravelMemoryManager:
    """Get singleton memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = TravelMemoryManager()
    return _memory_manager