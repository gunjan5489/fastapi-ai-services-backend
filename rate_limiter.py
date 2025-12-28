# rate_limiter.py - Rate limiting implementation
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import asyncio
from fastapi import HTTPException, Depends, status
from auth import APIKey, validate_api_key

class RateLimiter:
    def __init__(self):
        # Store request counts: {key_hash: [timestamps]}
        self.requests: Dict[str, List[datetime]] = defaultdict(list)
        self.lock = asyncio.Lock()

    async def check_rate_limit(self, api_key: APIKey) -> Tuple[bool, dict]:
        """
        Check if API key has exceeded rate limit
        Returns (is_allowed, stats)
        """
        async with self.lock:
            now = datetime.now()
            hour_ago = now - timedelta(hours=1)

            # Clean old requests
            key_requests = self.requests[api_key.key_hash]
            key_requests = [ts for ts in key_requests if ts > hour_ago]
            self.requests[api_key.key_hash] = key_requests

            # Check limit
            request_count = len(key_requests)

            if request_count >= api_key.rate_limit:
                return False, {
                    "requests_made": request_count,
                    "limit": api_key.rate_limit,
                    "reset_time": (hour_ago + timedelta(hours=1)).isoformat()
                }

            # Add current request
            key_requests.append(now)

            return True, {
                "requests_made": request_count + 1,
                "limit": api_key.rate_limit,
                "remaining": api_key.rate_limit - (request_count + 1)
            }

# Global rate limiter instance
rate_limiter = RateLimiter()

async def check_rate_limit(api_key: APIKey = Depends(validate_api_key)):
    """Dependency to check rate limiting"""
    is_allowed, stats = await rate_limiter.check_rate_limit(api_key)

    if not is_allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. {stats}",
            headers={"Retry-After": "3600"}  # 1 hour in seconds
        )

    return api_key