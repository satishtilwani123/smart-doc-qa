import logging
import sys
import os

# Prevent Python from creating .pyc files
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import qa as qa_router

# Configure logging with timestamps and levels
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app_debug.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
	app = FastAPI(title="smart-doc-qa")
	app.add_middleware(
		CORSMiddleware,
		allow_origins=["*"],
		allow_credentials=True,
		allow_methods=["*"],
		allow_headers=["*"],
	)

	app.include_router(qa_router.router, prefix="/qa", tags=["qa"])
	logger.info("FastAPI app created and routes registered")
	return app


app = create_app()
logger.info("Application started")

