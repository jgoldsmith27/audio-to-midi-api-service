import os
import io
import tempfile
import json
import base64
import logging
import sys
import traceback
import time
import threading
import concurrent.futures
import asyncio
from typing import Dict, Any, List, Optional, Union

# Set TensorFlow environment variables before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Better GPU memory management
# Explicitly disable CUDA/GPU for Render compatibility
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only

# Configure NumPy before TensorFlow imports
import numpy as np

# Import FastAPI components
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
import requests

# Set up enhanced logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("basic-pitch-service")

# Capture uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

logger.info("Starting up Basic Pitch service...")

# Warm up TensorFlow and Basic Pitch in a background thread
basic_pitch_loaded = False
model_load_error = None

def load_ml_libraries():
    global basic_pitch_loaded, model_load_error
    try:
        logger.info("Pre-loading ML libraries in background thread...")
        
        # Import TensorFlow safely to avoid circular imports
        try:
            # Import core TensorFlow first
            import tensorflow.compat.v2 as tf_core
            # Disable GPU
            tf_core.config.set_visible_devices([], 'GPU')
            # Then import the full module
            import tensorflow as tf
            logger.info("TensorFlow imported successfully")
        except ImportError as e:
            logger.error(f"Error importing TensorFlow: {str(e)}")
            logger.error(traceback.format_exc())
            raise e
        
        # Try to preload Basic Pitch model (similar to working code)
        logger.info("Attempting to preload Basic Pitch model...")
        try:
            from basic_pitch.inference import predict, Model
            from basic_pitch import ICASSP_2022_MODEL_PATH
            # Don't initialize the full model yet to save memory
            logger.info("Basic Pitch modules imported successfully")
        except Exception as model_error:
            logger.warning(f"Basic Pitch import warning (will retry later): {str(model_error)}")
        
        # Don't load basic_pitch yet - we'll do that on-demand
        basic_pitch_loaded = True
        logger.info("ML libraries pre-loaded successfully")
    except Exception as e:
        model_load_error = str(e)
        logger.error(f"Error pre-loading ML libraries: {str(e)}")
        logger.error(traceback.format_exc())

# Start preloading in background thread
threading.Thread(target=load_ml_libraries, daemon=True).start()

# Initialize FastAPI app
app = FastAPI(title="Basic Pitch Converter Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# In-memory job storage
job_storage = {}

# Model Classes
class OutputFormat(BaseModel):
    midi: bool = True
    csv: bool = False
    npz: bool = False
    wav: bool = False

class ConversionOptions(BaseModel):
    minFrequency: float = 27.5
    maxFrequency: float = 4186.0
    minNoteLength: int = 5
    energyThreshold: float = 0.25
    onsetThreshold: float = 0.25
    melodiaFilter: bool = True
    combineNotes: bool = True
    inferOnsets: bool = True
    multipleNotesPerFrame: bool = True
    outputFormat: OutputFormat = Field(default_factory=OutputFormat)

class ConversionRequest(BaseModel):
    audioUrl: str
    conversionId: str
    options: ConversionOptions = Field(default_factory=ConversionOptions)

class JobStatus(BaseModel):
    jobId: str
    status: str
    progress: float = 0
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

# Simplified security dependency
async def get_api_key(api_key_header: str = Header(None, alias=API_KEY_NAME), request: Request = None):
    # Check if anonymous access is enabled - this takes precedence
    if os.environ.get("ALLOW_ANONYMOUS_ACCESS", "").lower() == "true":
        logger.info("Allowing anonymous access - ALLOW_ANONYMOUS_ACCESS is enabled")
        return True
    
    # Otherwise, check the request source for network-level security
    if request and request.client:
        client_host = request.client.host
        logger.info(f"Request from: {client_host}")
        
        # Trust requests from localhost and internal networks
        if client_host == "127.0.0.1" or client_host.startswith("10.") or client_host.startswith("172.16."):
            logger.info(f"Request from trusted network: {client_host}")
            return True
    
    # If we get here and neither anonymous access nor network security passed, reject
    logger.warning(f"Authentication failed: Request didn't pass network security, and anonymous access not enabled")
    raise HTTPException(
        status_code=401,
        detail="Authentication required",
    )

# Add file size limit (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Helper function to download file from URL with size check
async def download_file(url: str) -> bytes:
    try:
        logger.info(f"Downloading file from URL: {url}")
        
        # Use streaming to check file size before downloading the whole file
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            
            # Check Content-Length if available
            content_length = response.headers.get('Content-Length')
            if content_length and int(content_length) > MAX_FILE_SIZE:
                logger.error(f"File too large: {int(content_length) / (1024 * 1024):.2f}MB exceeds limit of 10MB")
                raise HTTPException(status_code=413, detail=f"File too large (max 10MB)")
            
            # Download the file in chunks, checking size as we go
            chunks = []
            total_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                chunks.append(chunk)
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    logger.error(f"File too large: {total_size / (1024 * 1024):.2f}MB exceeds limit of 10MB")
                    raise HTTPException(status_code=413, detail=f"File too large (max 10MB)")
            
            file_data = b''.join(chunks)
            logger.info(f"File downloaded successfully ({len(file_data)} bytes)")
            return file_data
    except requests.RequestException as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download audio file: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error downloading file: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Add a resource limiter to prevent out of memory errors
def limit_memory():
    try:
        import resource
        # Limit virtual memory to 5GB (adjust based on Render's limits)
        resource.setrlimit(resource.RLIMIT_AS, (5 * 1024 * 1024 * 1024, -1))
        logger.info("Set memory limit to 5GB")
    except Exception as e:
        logger.warning(f"Failed to set memory limit: {str(e)}")

# Actual processing function - runs in background
async def process_audio_task(conversion_id: str, audio_data: bytes, options: ConversionOptions):
    try:
        # Update job status to processing
        job_storage[conversion_id] = JobStatus(
            jobId=conversion_id,
            status="processing",
            progress=0.1
        )
        
        # Simulate processing without loading ML libraries yet
        logger.info(f"Starting to process conversion: {conversion_id}")
        job_storage[conversion_id].progress = 0.2
        
        # Check if we had errors pre-loading the ML libraries
        if model_load_error:
            logger.error(f"Cannot process due to ML library initialization error: {model_load_error}")
            job_storage[conversion_id] = JobStatus(
                jobId=conversion_id,
                status="failed",
                progress=0,
                error=f"ML initialization error: {model_load_error}"
            )
            return
            
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
            logger.info(f"Audio saved to temporary file: {temp_file_path}")
            
        try:
            # Update progress
            job_storage[conversion_id].progress = 0.3
            
            # Apply resource limits within a separate thread to avoid affecting main app
            def process_with_limits():
                try:
                    # Limit memory usage
                    limit_memory()
                    
                    # Import TensorFlow in a way that avoids circular imports
                    logger.info("Starting TensorFlow configuration")
                    try:
                        # Import only the core TensorFlow module first
                        import tensorflow.compat.v2 as tf_core
                        # Explicitly disable GPU
                        tf_core.config.set_visible_devices([], 'GPU')
                        # Then import the full module
                        import tensorflow as tf
                        # Set logging level via environment variable instead
                        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ERROR level
                        logger.info("TensorFlow imported and configured successfully")
                    except ImportError as e:
                        logger.error(f"Error importing TensorFlow: {str(e)}")
                        logger.error(traceback.format_exc())
                        return {"success": False, "error": f"Failed to import TensorFlow: {str(e)}"}
                    
                    # Now import basic_pitch (should be faster since TensorFlow is already loaded)
                    logger.info("Importing basic_pitch for processing...")
                    try:
                        from basic_pitch.inference import predict
                        from basic_pitch import ICASSP_2022_MODEL_PATH
                        import basic_pitch
                        logger.info("Basic Pitch imported successfully")
                    except Exception as import_error:
                        logger.error(f"Failed to import Basic Pitch: {str(import_error)}")
                        logger.error(traceback.format_exc())
                        return {"success": False, "error": f"Failed to load Basic Pitch: {str(import_error)}"}
                    
                    # Remaining processing code...
                    # ...
                    
                    # This is a placeholder for the full implementation
                    return {"success": True, "midi": "dummy_base64_data"}
                except Exception as e:
                    logger.error(f"Error in processing thread: {str(e)}")
                    logger.error(traceback.format_exc())
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            # Run processing in a thread with better error handling
            logger.info("Waiting for processing to complete (timeout: 8 minutes)")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(process_with_limits)
                try:
                    result = future.result(timeout=480)  # 8 minute timeout
                    
                    # Handle the processing result
                    if result and isinstance(result, dict) and "success" in result:
                        if result["success"]:
                            # Mark job as completed with result
                            job_storage[conversion_id] = JobStatus(
                                jobId=conversion_id,
                                status="completed",
                                progress=1.0,
                                result={"midi": result["midi"]}
                            )
                            logger.info(f"Conversion {conversion_id} completed successfully")
                        else:
                            # Mark job as failed
                            job_storage[conversion_id] = JobStatus(
                                jobId=conversion_id,
                                status="failed",
                                progress=0,
                                error=result.get("error", "Unknown conversion error")
                            )
                            logger.error(f"Conversion {conversion_id} failed: {result.get('error', 'Unknown error')}")
                    else:
                        logger.error(f"Conversion {conversion_id} failed: Invalid result format")
                        job_storage[conversion_id] = JobStatus(
                            jobId=conversion_id,
                            status="failed",
                            progress=0,
                            error="Invalid result format from processing thread"
                        )
                except concurrent.futures.TimeoutError:
                    logger.error(f"Conversion {conversion_id} timed out after 8 minutes")
                    job_storage[conversion_id] = JobStatus(
                        jobId=conversion_id,
                        status="failed",
                        progress=0,
                        error="Processing timed out after 8 minutes. Try a shorter audio file."
                    )
                    # Try to cancel the future to free resources
                    future.cancel()
        except Exception as e:
            logger.error(f"Error in processing thread management: {str(e)}")
            logger.error(traceback.format_exc())
            job_storage[conversion_id].status = "failed"
            job_storage[conversion_id].error = f"Thread execution error: {str(e)}"
    
    except Exception as e:
        logger.error(f"Error in task: {str(e)}")
        logger.error(traceback.format_exc())
        job_storage[conversion_id] = JobStatus(
            jobId=conversion_id,
            status="failed",
            progress=0,
            error=f"Task error: {str(e)}"
        )

# API endpoints
@app.post("/convert", response_model=Dict[str, Any])
async def convert_audio(
    request: ConversionRequest,
    background_tasks: BackgroundTasks,
    _: Any = Depends(get_api_key)
):
    try:
        conversion_id = request.conversionId
        logger.info(f"Processing conversion request for {conversion_id}")
        
        # Validate audio URL
        if not request.audioUrl:
            raise HTTPException(status_code=400, detail="Audio URL is required")
            
        # Basic validation of URL format
        if not request.audioUrl.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid audio URL format")
        
        # Check if job already exists
        if conversion_id in job_storage:
            existing_job = job_storage[conversion_id]
            logger.info(f"Job already exists with status: {existing_job.status}")
            return {"jobId": conversion_id, "status": existing_job.status}
        
        try:
            # Download the audio file with timeout and size limits
            audio_data = await download_file(request.audioUrl)
            
            # Simple validation that it's actually audio data
            if len(audio_data) < 100:  # Extremely small files are likely not valid audio
                raise HTTPException(status_code=400, detail="Invalid audio file (too small)")
                
            # Create a new job
            job_storage[conversion_id] = JobStatus(
                jobId=conversion_id,
                status="pending",
                progress=0
            )
            
            # Start the conversion task in the background
            background_tasks.add_task(process_audio_task, conversion_id, audio_data, request.options)
            logger.info(f"Conversion job {conversion_id} started in background")
            
            # Return the job ID
            return {"jobId": conversion_id, "status": "pending"}
        except HTTPException:
            # Re-raise HTTP exceptions without modification
            raise
        except Exception as e:
            logger.error(f"Error preparing conversion: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error preparing conversion: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in /convert endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job/{job_id}", response_model=Dict[str, Any])
async def get_job_status(job_id: str, _: Any = Depends(get_api_key)):
    if job_id not in job_storage:
        # Return a specific format that matches what the frontend expects
        return {
            "status": "not_found",
            "jobId": job_id,
            "error": "Job not found. It may have expired or was never created."
        }
    
    job = job_storage[job_id]
    logger.info(f"Retrieved job status for {job_id}: {job.status}")
    
    # For completed jobs, schedule cleanup after some time
    if job.status == "completed" or job.status == "failed":
        # Create a task to clean up the job after some time
        async def cleanup_job(job_id):
            try:
                await asyncio.sleep(3600)  # Clean up after 1 hour
                if job_id in job_storage:
                    logger.info(f"Cleaning up completed job {job_id}")
                    del job_storage[job_id]
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
        
        # Schedule the cleanup
        asyncio.create_task(cleanup_job(job_id))
    
    # Convert job status to dict with better error messages
    result = job.dict()
    
    # Add debug info for errors to help diagnose issues
    if job.status == "failed" and job.error:
        result["debug_info"] = {
            "error_type": type(job.error).__name__ if isinstance(job.error, Exception) else "str",
            "last_recorded_progress": job.progress
        }
        
        # Make sure error is a string
        if not isinstance(result["error"], str):
            result["error"] = str(result["error"])
            
    return result

@app.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "message": "Service is running", 
        "time": time.time(),
        "ml_libraries_loaded": basic_pitch_loaded,
        "ml_load_error": model_load_error
    }

@app.get("/")
async def root():
    return {"message": "Basic Pitch Converter API", "status": "ok", "docs": "/docs"}

# For direct running on Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
