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
from typing import Dict, Any, List, Optional, Union
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
import requests

# Set TensorFlow environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Better GPU memory management

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
        import numpy as np
        # Preload just the necessary TensorFlow components without initializing models
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')  # Disable GPU to save memory
        logger.info("TensorFlow imported successfully")
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

# Helper function to download file from URL
async def download_file(url: str) -> bytes:
    try:
        logger.info(f"Downloading file from URL: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        file_data = response.content
        logger.info(f"File downloaded successfully ({len(file_data)} bytes)")
        return file_data
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download audio file: {str(e)}")

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
            
            # First, make sure numpy and tensorflow are imported - should be quick since they're preloaded
            import numpy as np
            import tensorflow as tf
            logger.info("NumPy and TensorFlow imported for processing")
            
            # Now import basic_pitch (should be faster since TensorFlow is already loaded)
            logger.info("Importing basic_pitch for processing...")
            import basic_pitch
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import predict
            logger.info("Basic Pitch imported successfully")
            
            # Update progress
            job_storage[conversion_id].progress = 0.4
            
            # Process with Basic Pitch
            logger.info(f"Running Basic Pitch on file: {temp_file_path}")
            midi_data, notes, _ = predict(
                temp_file_path,
                ICASSP_2022_MODEL_PATH,
                min_frequency=options.minFrequency,
                max_frequency=options.maxFrequency,
                onset_threshold=options.onsetThreshold,
                frame_threshold=options.energyThreshold,
                min_note_length=options.minNoteLength,
                min_frequency_note=options.minFrequency,
                max_frequency_note=options.maxFrequency,
                melodia_trick=options.melodiaFilter,
                combine_notes=options.combineNotes,
                infer_onsets=options.inferOnsets,
                multiple_pitch_bends=not options.multipleNotesPerFrame
            )
            
            # Update progress
            job_storage[conversion_id].progress = 0.8
            
            # Save MIDI to temporary file
            midi_temp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
            midi_temp.close()
            
            basic_pitch.save_midi(
                midi_data,
                notes,
                midi_temp.name,
                options.minFrequency,
                options.maxFrequency,
                options.multipleNotesPerFrame
            )
            
            # Read the MIDI file and encode to base64
            with open(midi_temp.name, "rb") as midi_file:
                midi_bytes = midi_file.read()
                midi_base64 = base64.b64encode(midi_bytes).decode("utf-8")
            
            # Mark job as completed with result
            job_storage[conversion_id] = JobStatus(
                jobId=conversion_id,
                status="completed",
                progress=1.0,
                result={"midi": midi_base64}
            )
            
            # Clean up temporary files
            os.unlink(temp_file_path)
            os.unlink(midi_temp.name)
            
            logger.info(f"Conversion {conversion_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error in basic-pitch processing: {str(e)}")
            logger.error(traceback.format_exc())
            job_storage[conversion_id] = JobStatus(
                jobId=conversion_id,
                status="failed",
                progress=0,
                error=f"Error processing audio: {str(e)}"
            )
            
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
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
        
        # Check if job already exists
        if conversion_id in job_storage:
            return {"jobId": conversion_id, "status": "already_exists"}
        
        # Download the audio file
        audio_data = await download_file(request.audioUrl)
        
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
    
    except Exception as e:
        logger.error(f"Error in /convert endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job/{job_id}", response_model=Dict[str, Any])
async def get_job_status(job_id: str, _: Any = Depends(get_api_key)):
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_storage[job_id]
    logger.info(f"Retrieved job status for {job_id}: {job.status}")
    
    # For completed jobs, schedule cleanup after some time
    if job.status == "completed" or job.status == "failed":
        async def cleanup_job():
            await asyncio.sleep(3600)  # Clean up after 1 hour
            if job_id in job_storage:
                del job_storage[job_id]
        
        return job.dict()
    
    return job.dict()

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
