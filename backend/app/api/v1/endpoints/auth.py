# app/api/auth.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Dict

from app.core.database import get_db
from app.core.config import settings

if settings.SKIP_AUTH:
    from app.core.auth_optional import get_current_user
else:
    from app.core.auth import get_current_user

from app.models import User, DisclaimerAcceptance
from app.schemas.auth import (
    DisclaimerRequest,
    DisclaimerResponse,
    UserResponse,
)

router = APIRouter()


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get current user.
    DB acts as cached user profile.
    """

    user = (
        db.query(User)
        .filter(User.auth0_id == current_user["auth0_id"])
        .first()
    )

    print(current_user)

    # First login → create user
    if not user:
        user = User(
            auth0_id=current_user["auth0_id"],
            email=current_user["email"],
            name=current_user.get("username"),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    # Optional auto-sync name changes
    else:
        if user.name != current_user.get("username") :
            user.name = current_user.get("name")
            if current_user.get("username"):
                user.name = current_user.get("username")
            db.commit()

    return user


@router.post("/disclaimer/accept", response_model=DisclaimerResponse)
async def accept_disclaimer(
    request: DisclaimerRequest,
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Accept disclaimer"""

    user = (
        db.query(User)
        .filter(User.auth0_id == current_user["auth0_id"])
        .first()
    )

    if not user:
        user = User(
            auth0_id=current_user["auth0_id"],
            email=current_user["email"],
            name=current_user.get("name"),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    existing = (
        db.query(DisclaimerAcceptance)
        .filter(DisclaimerAcceptance.user_id == user.id)
        .first()
    )

    if existing:
        return {"accepted": True, "version": 1}

    acceptance = DisclaimerAcceptance(
        user_id=user.id,
        ip_address=request.ip_address,
        user_agent=request.user_agent,
    )

    db.add(acceptance)
    db.commit()

    return {"accepted": True, "version": 1}


@router.get("/disclaimer/status")
async def check_disclaimer_status(
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Check disclaimer status"""

    CURRENT_VERSION = 1

    user = (
        db.query(User)
        .filter(User.auth0_id == current_user["auth0_id"])
        .first()
    )

    if not user:
        return {"accepted": False, "version": CURRENT_VERSION}

    acceptance = (
        db.query(DisclaimerAcceptance)
        .filter(DisclaimerAcceptance.user_id == user.id)
        .first()
    )

    return {
        "accepted": acceptance is not None,
        "version": CURRENT_VERSION,
    }