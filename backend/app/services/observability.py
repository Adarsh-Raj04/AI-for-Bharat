"""
Observability - LangSmith integration for trace logging
"""
import os
from typing import Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import LangSmith
try:
    from langsmith import Client
    from langsmith.run_helpers import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    logger.warning("LangSmith not available. Install with: pip install langsmith")


class ObservabilityService:
    """
    Handles observability and trace logging using LangSmith
    """
    
    def __init__(self):
        self.enabled = False
        self.client = None
        self.project_name = "medresearch-ai"
        
        if LANGSMITH_AVAILABLE:
            self._initialize_langsmith()
    
    def _initialize_langsmith(self):
        """Initialize LangSmith client"""
        api_key = os.getenv("LANGSMITH_API_KEY")
        project_name = os.getenv("LANGSMITH_PROJECT", "medresearch-ai")
        
        if not api_key:
            logger.info("LangSmith API key not found. Tracing disabled.")
            return
        
        try:
            self.client = Client(api_key=api_key)
            self.project_name = project_name
            self.enabled = True
            logger.info(f"✅ LangSmith tracing enabled (project: {project_name})")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize LangSmith: {e}. Tracing disabled.")
            self.enabled = False
    
    def log_query(
        self,
        query: str,
        user_id: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Log a user query
        
        Args:
            query: User's query
            user_id: User identifier
            session_id: Session identifier
            metadata: Additional metadata
            
        Returns:
            Run ID if successful, None otherwise
        """
        if not self.enabled:
            return None
        
        try:
            # Create run and get the ID
            run = self.client.create_run(
                name="user_query",
                run_type="chain",
                inputs={"query": query},
                project_name=self.project_name,
                extra={
                    "user_id": user_id,
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    **(metadata or {})
                }
            )
            
            # Extract UUID properly - handle Run object
            run_id = None
            if hasattr(run, 'id'):
                # It's a Run object
                run_id = str(run.id) if run.id else None
            elif isinstance(run, dict) and 'id' in run:
                # It's a dict
                run_id = str(run['id'])
            
            if not run_id:
                logger.debug("Failed to extract run_id from LangSmith response")
                self.enabled = False
                return None
            
            logger.debug(f"LangSmith run created: {run_id}")
            return run_id
            
        except Exception as e:
            logger.debug(f"Failed to log query: {e}")
            self.enabled = False  # Disable to avoid repeated errors
            return None
    
    def log_retrieval(
        self,
        run_id: str,
        query: str,
        documents: list,
        top_k: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log document retrieval
        
        Args:
            run_id: Parent run ID
            query: Search query
            documents: Retrieved documents
            top_k: Number of documents requested
            metadata: Additional metadata
        """
        if not self.enabled or not run_id:
            return
        
        try:
            self.client.create_run(
                name="document_retrieval",
                run_type="retriever",
                inputs={"query": query, "top_k": top_k},
                outputs={"documents": [
                    {
                        "id": doc.get("id"),
                        "score": doc.get("score"),
                        "source_type": doc.get("source_type")
                    }
                    for doc in documents[:10]  # Limit to first 10
                ]},
                parent_run_id=run_id,
                project_name=self.project_name,
                extra=metadata or {}
            )
        except Exception as e:
            logger.debug(f"LangSmith logging failed (non-critical): {e}")
    
    def log_generation(
        self,
        run_id: str,
        prompt: str,
        response: str,
        model: str,
        tokens_used: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log response generation
        
        Args:
            run_id: Parent run ID
            prompt: Input prompt
            response: Generated response
            model: Model used
            tokens_used: Number of tokens
            metadata: Additional metadata
        """
        if not self.enabled or not run_id:
            return
        
        try:
            self.client.create_run(
                name="response_generation",
                run_type="llm",
                inputs={"prompt": prompt[:1000]},  # Truncate long prompts
                outputs={"response": response[:1000]},  # Truncate long responses
                parent_run_id=run_id,
                project_name=self.project_name,
                extra={
                    "model": model,
                    "tokens_used": tokens_used,
                    **(metadata or {})
                }
            )
        except Exception as e:
            logger.debug(f"LangSmith logging failed (non-critical): {e}")
    
    def log_safety_check(
        self,
        run_id: str,
        query: str,
        is_safe: bool,
        block_reason: Optional[str],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log safety guardrail check
        
        Args:
            run_id: Parent run ID
            query: User query
            is_safe: Whether query passed safety check
            block_reason: Reason for blocking if not safe
            metadata: Additional metadata
        """
        if not self.enabled or not run_id:
            return
        
        try:
            self.client.create_run(
                name="safety_check",
                run_type="tool",
                inputs={"query": query},
                outputs={
                    "is_safe": is_safe,
                    "block_reason": block_reason
                },
                parent_run_id=run_id,
                project_name=self.project_name,
                extra=metadata or {}
            )
        except Exception as e:
            logger.debug(f"LangSmith logging failed (non-critical): {e}")
    
    def log_bias_analysis(
        self,
        run_id: str,
        documents: list,
        bias_analysis: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log bias detection analysis
        
        Args:
            run_id: Parent run ID
            documents: Analyzed documents
            bias_analysis: Bias analysis results
            metadata: Additional metadata
        """
        if not self.enabled or not run_id:
            return
        
        try:
            self.client.create_run(
                name="bias_analysis",
                run_type="tool",
                inputs={"document_count": len(documents)},
                outputs={
                    "has_bias": bias_analysis.get("has_bias"),
                    "bias_score": bias_analysis.get("bias_score"),
                    "bias_flags": [
                        {
                            "type": flag.get("type"),
                            "severity": flag.get("severity")
                        }
                        for flag in bias_analysis.get("bias_flags", [])
                    ]
                },
                parent_run_id=run_id,
                project_name=self.project_name,
                extra=metadata or {}
            )
        except Exception as e:
            logger.debug(f"LangSmith logging failed (non-critical): {e}")
    
    def complete_run(
        self,
        run_id: str,
        outputs: Dict[str, Any],
        error: Optional[str] = None
    ):
        """
        Complete a run
        
        Args:
            run_id: Run ID to complete
            outputs: Final outputs
            error: Error message if failed
        """
        if not self.enabled or not run_id:
            return
        
        try:
            self.client.update_run(
                run_id=run_id,
                outputs=outputs,
                error=error,
                end_time=datetime.utcnow()
            )
            logger.debug(f"LangSmith run completed: {run_id}")
        except Exception as e:
            logger.debug(f"LangSmith logging failed (non-critical): {e}")
            self.enabled = False  # Disable for this session


# Singleton instance
_observability_service = None


def get_observability_service() -> ObservabilityService:
    """Get or create the observability service singleton"""
    global _observability_service
    if _observability_service is None:
        _observability_service = ObservabilityService()
    return _observability_service
