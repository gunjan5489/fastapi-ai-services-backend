import os
import hashlib
import secrets
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from fastapi import HTTPException, Security, Depends, status
from fastapi.security import APIKeyHeader, APIKeyQuery
from starlette.status import HTTP_403_FORBIDDEN
from pydantic import BaseModel

# API Key Configuration
API_KEY_NAME = "X-API-Key"
API_KEY_HEADER = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
API_KEY_QUERY = APIKeyQuery(name="api_key", auto_error=False)

# Store for API keys (in production, use a database)
class APIKey(BaseModel):
    key_hash: str
    name: str
    environment: str  # 'development', 'testing', 'production'
    created_at: datetime
    last_used: Optional[datetime] = None
    is_active: bool = True
    rate_limit: int = 100  # requests per hour
    permissions: List[str] = []  # specific endpoint permissions

# In production, store these in a database
# For now, using environment variables
VALID_API_KEYS: Dict[str, APIKey] = {}

def load_api_keys_from_env():
    """Load API keys from environment variables"""
    global VALID_API_KEYS

    # Development keys
    if dev_key := os.getenv("DEV_API_KEY"):
        key_hash = hashlib.sha256(dev_key.encode()).hexdigest()
        VALID_API_KEYS[key_hash] = APIKey(
            key_hash=key_hash,
            name="Development Team",
            environment="development",
            created_at=datetime.now(),
            permissions=["*"]  # All permissions
        )

    # Testing keys (for QC team)
    if test_key := os.getenv("TEST_API_KEY"):
        key_hash = hashlib.sha256(test_key.encode()).hexdigest()
        VALID_API_KEYS[key_hash] = APIKey(
            key_hash=key_hash,
            name="QC Team",
            environment="testing",
            created_at=datetime.now(),
            rate_limit=50,  # Lower rate limit for testing
            # permissions=[
            #     "tags:resolve",
            #     "translate",
            #     "image:analyze"
            #     # No image generation to save costs
            # ]
            permissions=["*"]  # All permissions
        )

    # Production keys
    if prod_key := os.getenv("PROD_API_KEY"):
        key_hash = hashlib.sha256(prod_key.encode()).hexdigest()
        VALID_API_KEYS[key_hash] = APIKey(
            key_hash=key_hash,
            name="Production Service",
            environment="production",
            created_at=datetime.now(),
            rate_limit=1000,
            permissions=["*"]
        )

def validate_api_key(
    api_key_header: Optional[str] = Security(API_KEY_HEADER),
    api_key_query: Optional[str] = Security(API_KEY_QUERY)
) -> APIKey:
    """
    Validate API key from header or query parameter
    Returns the APIKey object if valid
    """
    # Try header first, then query parameter
    api_key = api_key_header or api_key_query

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key missing. Please provide X-API-Key header or api_key query parameter"
        )

    # Hash the provided key to compare with stored hashes
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    if key_hash not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )

    api_key_data = VALID_API_KEYS[key_hash]

    if not api_key_data.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key has been deactivated"
        )

    # Update last used timestamp
    api_key_data.last_used = datetime.now()

    return api_key_data

def check_permission(required_permission: str):
    """
    Decorator to check if API key has required permission
    """
    def permission_checker(api_key: APIKey = Depends(validate_api_key)) -> APIKey:
        if "*" in api_key.permissions:  # Admin access
            return api_key

        if required_permission not in api_key.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key lacks permission: {required_permission}"
            )
        return api_key

    return permission_checker