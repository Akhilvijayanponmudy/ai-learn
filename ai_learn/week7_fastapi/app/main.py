from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from app.core.config import settings
from app.core.rate_limit import limiter
from app.api.routes import router

def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME)

    # rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # routes
    app.include_router(router)

    return app

app = create_app()
