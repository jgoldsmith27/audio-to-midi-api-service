from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import JSONResponse
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the FastAPI app from main.py
from main import app as main_app

# Create a new FastAPI app for Vercel
app = FastAPI()

# Mount the main app
app.mount("/", main_app)

# Add a root endpoint for health check
@app.get("/")
async def root():
    return JSONResponse({"status": "ok", "message": "Basic Pitch Converter Service is running"}) 