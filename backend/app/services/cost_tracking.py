"""
Cost Tracking Service - Track and analyze LLM costs using LangSmith data
Phase 3 Feature #23
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CostTrackingService:
    """
    Service for tracking and analyzing LLM costs.
    Uses LangSmith data for comprehensive cost analysis.
    """
    
    # Token costs per 1M tokens (as of 2026)
    TOKEN_COSTS = {
        # Anthropic Claude
        'claude-sonnet-4-5-20250929': {
            'input': 3.00,   # $3 per 1M input tokens
            'output': 15.00  # $15 per 1M output tokens
        },
        'claude-3-opus-20240229': {
            'input': 15.00,
            'output': 75.00
        },
        'claude-3-sonnet-20240229': {
            'input': 3.00,
            'output': 15.00
        },
        'claude-3-haiku-20240307': {
            'input': 0.25,
            'output': 1.25
        },
        
        # OpenAI
        'gpt-4o': {
            'input': 5.00,
            'output': 15.00
        },
        'gpt-4o-mini': {
            'input': 0.15,
            'output': 0.60
        },
        'gpt-4.1-nano': {
            'input': 0.10,
            'output': 0.40
        },
        'gpt-4-turbo': {
            'input': 10.00,
            'output': 30.00
        },
        'gpt-3.5-turbo': {
            'input': 0.50,
            'output': 1.50
        },
        
        # AWS Bedrock (approximate)
        'anthropic.claude-3-sonnet-20240229-v1:0': {
            'input': 3.00,
            'output': 15.00
        },
    }
    
    def __init__(self):
        self.langsmith_client = None
        self._initialize_langsmith()
    
    def _initialize_langsmith(self):
        """Initialize LangSmith client if available"""
        try:
            from app.core.config import settings
            if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
                from langsmith import Client
                self.langsmith_client = Client(
                    api_key=settings.LANGSMITH_API_KEY
                )
                logger.info("LangSmith client initialized for cost tracking")
        except Exception as e:
            logger.warning(f"Could not initialize LangSmith client: {e}")
            self.langsmith_client = None
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost for a single LLM call
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        if model not in self.TOKEN_COSTS:
            logger.warning(f"Unknown model for cost calculation: {model}")
            # Use default Claude Sonnet pricing
            model = 'claude-sonnet-4-5-20250929'
        
        costs = self.TOKEN_COSTS[model]
        input_cost = (input_tokens / 1_000_000) * costs['input']
        output_cost = (output_tokens / 1_000_000) * costs['output']
        
        return input_cost + output_cost
    
    def get_cost_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        project_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get cost summary from LangSmith data
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            project_name: LangSmith project name
            
        Returns:
            Cost summary dict
        """
        if not self.langsmith_client:
            return {
                "error": "LangSmith not configured",
                "total_cost": 0.0,
                "total_queries": 0
            }
        
        try:
            # Default to last 7 days
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=7)
            
            # Get runs from LangSmith
            runs = list(self.langsmith_client.list_runs(
                project_name=project_name,
                start_time=start_date,
                end_time=end_date,
                is_root=True  # Only root runs (not intermediate steps)
            ))
            
            total_cost = 0.0
            total_input_tokens = 0
            total_output_tokens = 0
            costs_by_model = {}
            costs_by_intent = {}
            costs_by_day = {}
            
            for run in runs:
                # Extract token usage
                if run.outputs and 'token_usage' in run.outputs:
                    usage = run.outputs['token_usage']
                    input_tokens = usage.get('prompt_tokens', 0)
                    output_tokens = usage.get('completion_tokens', 0)
                elif hasattr(run, 'prompt_tokens') and hasattr(run, 'completion_tokens'):
                    input_tokens = run.prompt_tokens or 0
                    output_tokens = run.completion_tokens or 0
                else:
                    continue
                
                # Get model name
                model = run.extra.get('metadata', {}).get('model', 'claude-sonnet-4-5-20250929')
                
                # Calculate cost
                cost = self.calculate_cost(model, input_tokens, output_tokens)
                total_cost += cost
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                # Track by model
                if model not in costs_by_model:
                    costs_by_model[model] = {
                        'cost': 0.0,
                        'queries': 0,
                        'input_tokens': 0,
                        'output_tokens': 0
                    }
                costs_by_model[model]['cost'] += cost
                costs_by_model[model]['queries'] += 1
                costs_by_model[model]['input_tokens'] += input_tokens
                costs_by_model[model]['output_tokens'] += output_tokens
                
                # Track by intent
                intent = run.extra.get('metadata', {}).get('intent', 'unknown')
                if intent not in costs_by_intent:
                    costs_by_intent[intent] = {
                        'cost': 0.0,
                        'queries': 0
                    }
                costs_by_intent[intent]['cost'] += cost
                costs_by_intent[intent]['queries'] += 1
                
                # Track by day
                day = run.start_time.strftime('%Y-%m-%d')
                if day not in costs_by_day:
                    costs_by_day[day] = {
                        'cost': 0.0,
                        'queries': 0
                    }
                costs_by_day[day]['cost'] += cost
                costs_by_day[day]['queries'] += 1
            
            return {
                "total_cost": round(total_cost, 4),
                "total_queries": len(runs),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "average_cost_per_query": round(total_cost / len(runs), 4) if runs else 0.0,
                "costs_by_model": costs_by_model,
                "costs_by_intent": costs_by_intent,
                "costs_by_day": costs_by_day,
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting cost summary: {e}")
            return {
                "error": str(e),
                "total_cost": 0.0,
                "total_queries": 0
            }
    
    def get_cost_by_user(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get cost summary for a specific user
        
        Args:
            user_id: User identifier
            start_date: Start date
            end_date: End date
            
        Returns:
            User cost summary
        """
        if not self.langsmith_client:
            return {
                "error": "LangSmith not configured",
                "user_id": user_id,
                "total_cost": 0.0
            }
        
        try:
            # Default to last 30 days
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Get runs for this user
            runs = list(self.langsmith_client.list_runs(
                start_time=start_date,
                end_time=end_date,
                filter=f'eq(metadata.user_id, "{user_id}")',
                is_root=True
            ))
            
            total_cost = 0.0
            for run in runs:
                if run.outputs and 'token_usage' in run.outputs:
                    usage = run.outputs['token_usage']
                    input_tokens = usage.get('prompt_tokens', 0)
                    output_tokens = usage.get('completion_tokens', 0)
                    model = run.extra.get('metadata', {}).get('model', 'claude-sonnet-4-5-20250929')
                    total_cost += self.calculate_cost(model, input_tokens, output_tokens)
            
            return {
                "user_id": user_id,
                "total_cost": round(total_cost, 4),
                "total_queries": len(runs),
                "average_cost_per_query": round(total_cost / len(runs), 4) if runs else 0.0,
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting user cost: {e}")
            return {
                "error": str(e),
                "user_id": user_id,
                "total_cost": 0.0
            }
    
    def estimate_monthly_cost(
        self,
        queries_per_day: int,
        avg_input_tokens: int = 1000,
        avg_output_tokens: int = 500,
        model: str = 'claude-sonnet-4-5-20250929'
    ) -> Dict[str, Any]:
        """
        Estimate monthly cost based on usage patterns
        
        Args:
            queries_per_day: Expected queries per day
            avg_input_tokens: Average input tokens per query
            avg_output_tokens: Average output tokens per query
            model: Model name
            
        Returns:
            Cost estimate
        """
        cost_per_query = self.calculate_cost(model, avg_input_tokens, avg_output_tokens)
        daily_cost = cost_per_query * queries_per_day
        monthly_cost = daily_cost * 30
        
        return {
            "model": model,
            "queries_per_day": queries_per_day,
            "cost_per_query": round(cost_per_query, 4),
            "daily_cost": round(daily_cost, 2),
            "monthly_cost": round(monthly_cost, 2),
            "yearly_cost": round(monthly_cost * 12, 2)
        }


# Singleton instance
_cost_tracking_service = None


def get_cost_tracking_service() -> CostTrackingService:
    """Get or create the cost tracking service singleton"""
    global _cost_tracking_service
    if _cost_tracking_service is None:
        _cost_tracking_service = CostTrackingService()
    return _cost_tracking_service
