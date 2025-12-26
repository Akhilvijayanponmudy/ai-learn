import logging
import json
from pythonjsonlogger import jsonlogger
import contextvars
import uuid

# Context variable to store request ID
request_id_ctx_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default=None)

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_ctx_var.get()
        return True

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logHandler = logging.StreamHandler()
    
    # Custom formatter
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s"
    )
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)
    
    # Add filter to root logger
    logger.addFilter(RequestIdFilter())
    
    # Quiet down some noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

def get_logger(name):
    return logging.getLogger(name)
