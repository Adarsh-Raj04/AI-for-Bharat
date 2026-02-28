# app/core/auth.py

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
import requests
from app.core.config import settings

security = HTTPBearer()


class Auth0JWTBearer:
    def __init__(self):
        self.domain = settings.AUTH0_DOMAIN
        self.audience = settings.AUTH0_API_AUDIENCE
        self.issuer = settings.AUTH0_ISSUER
        self.algorithms = [settings.AUTH0_ALGORITHMS]
        self.jwks_uri = f"https://{self.domain}/.well-known/jwks.json"
        self._jwks_cache = None

    def get_jwks(self):
        """Fetch JWKS (cached)"""
        if not self._jwks_cache:
            response = requests.get(self.jwks_uri, timeout=5)
            self._jwks_cache = response.json()
        return self._jwks_cache

    def get_signing_key(self, token: str):
        """Get signing key from JWKS"""
        unverified_header = jwt.get_unverified_header(token)
        jwks = self.get_jwks()

        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                return {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n": key["n"],
                    "e": key["e"],
                }

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unable to find appropriate signing key",
        )

    async def __call__(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> dict:

        token = credentials.credentials

        try:
            signing_key = self.get_signing_key(token)

            payload = jwt.decode(
                token,
                signing_key,
                algorithms=self.algorithms,
                audience=self.audience,
                issuer=self.issuer,
            )

            return payload

        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )


# Dependency
auth_scheme = Auth0JWTBearer()


async def get_current_user(
    token_payload: dict = Depends(auth_scheme),
) -> dict:
    """
    Extract user info directly from JWT.
    No Auth0 /userinfo call (FAST).
    """

    auth0_id = token_payload.get("sub")
    namespace = "https://medresearch-ai-api"

    email = (
        token_payload.get("email")
        or token_payload.get(f"{namespace}/email")
    )

    name = (
        token_payload.get("name")
        or token_payload.get(f"{namespace}/name")
        or token_payload.get(f"{namespace}/nickname")
    )
    username = (
    token_payload.get(f"{namespace}/username")
)

    if not auth0_id or not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user payload - missing auth0_id or email",
        )

    result = {
        "auth0_id": auth0_id,
        "email": email,
        "name": name or "User",
        "username": username or name,
        "permissions": token_payload.get("permissions", []),
    }
    
    return result