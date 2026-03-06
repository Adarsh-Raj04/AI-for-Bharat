"""
Analytics API endpoints - Cost tracking and usage analytics
Phase 3 Feature #23
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from typing import Dict, Optional
from datetime import datetime, timedelta
import logging


from app.core.auth import get_current_user
from app.core.config import settings
from app.core.rate_limit import limiter
from app.services.cost_tracking import get_cost_tracking_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/costs/summary")
@limiter.limit("20/minute")
async def get_cost_summary(
    request: Request,
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
    current_user: Dict = Depends(get_current_user),
):
    """
    Get cost summary for the specified time period
    
    Args:
        days: Number of days to analyze (1-90)
    
    Returns:
        Cost summary with breakdowns by model, intent, and day
    """
    try:
        cost_service = get_cost_tracking_service()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        summary = cost_service.get_cost_summary(
            start_date=start_date,
            end_date=end_date,
            project_name = settings.LANGSMITH_PROJECT or "medresearch-ai"
        )
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting cost summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/costs/user")
@limiter.limit("20/minute")
async def get_user_costs(
    request: Request,
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    current_user: Dict = Depends(get_current_user),
):
    """
    Get cost summary for the current user
    
    Args:
        days: Number of days to analyze (1-365)
    
    Returns:
        User-specific cost summary
    """
    try:
        cost_service = get_cost_tracking_service()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        user_costs = cost_service.get_cost_by_user(
            user_id=current_user["auth0_id"],
            start_date=start_date,
            end_date=end_date
        )
        
        return user_costs
        
    except Exception as e:
        logger.error(f"Error getting user costs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/costs/estimate")
@limiter.limit("20/minute")
async def estimate_costs(
    request: Request,
    queries_per_day: int = Query(..., ge=1, le=100000, description="Expected queries per day"),
    avg_input_tokens: int = Query(1000, ge=100, le=10000, description="Average input tokens"),
    avg_output_tokens: int = Query(500, ge=100, le=5000, description="Average output tokens"),
    model: str = Query("claude-sonnet-4-5-20250929", description="Model name"),
    current_user: Dict = Depends(get_current_user),
):
    """
    Estimate monthly costs based on usage patterns
    
    Args:
        queries_per_day: Expected number of queries per day
        avg_input_tokens: Average input tokens per query
        avg_output_tokens: Average output tokens per query
        model: Model name to use for estimation
    
    Returns:
        Cost estimates (daily, monthly, yearly)
    """
    try:
        cost_service = get_cost_tracking_service()
        
        estimate = cost_service.estimate_monthly_cost(
            queries_per_day=queries_per_day,
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
            model=model
        )
        
        return estimate
        
    except Exception as e:
        logger.error(f"Error estimating costs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/costs/models")
@limiter.limit("20/minute")
async def get_model_costs(
    request: Request,
    current_user: Dict = Depends(get_current_user),
):
    """
    Get pricing information for all supported models
    
    Returns:
        Model pricing information
    """
    try:
        cost_service = get_cost_tracking_service()
        
        return {
            "models": cost_service.TOKEN_COSTS,
            "note": "Costs are per 1 million tokens in USD"
        }
        
    except Exception as e:
        logger.error(f"Error getting model costs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
