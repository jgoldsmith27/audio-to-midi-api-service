import os
import io
import tempfile
import json
import base64
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, HttpUrl
import requests
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("basic-pitch-service")

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

# For Vercel deployment - import basic-pitch only when needed
# This helps reduce cold start time
basic_pitch_module = None

def get_basic_pitch():
    global basic_pitch_module
    if basic_pitch_module is None:
        try:
            import basic_pitch
            basic_pitch_module = basic_pitch
        except ImportError:
            logger.error("Failed to import basic_pitch module")
            raise HTTPException(status_code=500, detail="Basic Pitch module not available")
    return basic_pitch_module

# Security configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Job tracking
jobs = {}

# Model classes
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
        logger.info(f"File downloaded successfully ({len(response.content)} bytes)")
        return response.content
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Could not download file: {str(e)}")

# Process audio in a background task
async def process_audio_task(conversion_id: str, audio_data: bytes, options: ConversionOptions):
    job_id = conversion_id
    jobs[job_id] = {"status": "processing", "progress": 0}
    
    try:
        # Save audio data to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_audio.write(audio_data)
            audio_path = temp_audio.name
        
        # Create output directory
        output_dir = tempfile.mkdtemp()
        
        # Update job status
        jobs[job_id]["progress"] = 10
        
        try:
            # Get the basic_pitch module using our lazy-loading function
            basic_pitch = get_basic_pitch()
            from basic_pitch.inference import predict_and_save, predict
            
            # Update job status
            jobs[job_id]["progress"] = 20
            
            # Determine outputs to save
            save_midi = options.outputFormat.midi
            save_notes = options.outputFormat.csv
            save_model_outputs = options.outputFormat.npz
            sonify_midi = options.outputFormat.wav
            
            # Process in chunks for large files
            max_audio_length = 600  # 10 minutes
            
            # Normalize Python parameters from JS camelCase
            basic_pitch_params = {
                "minimum_frequency": options.minFrequency,
                "maximum_frequency": options.maxFrequency,
                "min_note_length": options.minNoteLength,
                "energy_threshold": options.energyThreshold,
                "onset_threshold": options.onsetThreshold,
                "melodia_trick": options.melodiaFilter,
                "combine_notes_with_same_pitch": options.combineNotes,
                "inference_onset": options.inferOnsets,
                "multiple_pitch_bends": options.multipleNotesPerFrame
            }
            
            # Update job status
            jobs[job_id]["progress"] = 30
            
            # Call basic-pitch
            logger.info(f"Processing audio with basic-pitch: {audio_path}")
            logger.info(f"Basic Pitch parameters: {basic_pitch_params}")
            
            # Start the prediction process
            model_output, midi_data, note_events = predict(
                audio_path,
                **basic_pitch_params
            )
            
            # Update job status
            jobs[job_id]["progress"] = 70
            
            # Save files if requested
            output_file_base = os.path.join(output_dir, f"basic_pitch_{conversion_id}")
            
            # Get the MIDI data as bytes
            midi_bytes = io.BytesIO()
            midi_data.save(midi_bytes)
            midi_bytes.seek(0)
            midi_data_bytes = midi_bytes.read()
            
            # Convert MIDI data to base64
            midi_base64 = base64.b64encode(midi_data_bytes).decode('utf-8')
            
            # Convert note events to serializable format
            note_events_serializable = []
            for note in note_events:
                note_events_serializable.append({
                    "pitch": int(note.pitch),
                    "start_time": float(note.start_time),
                    "end_time": float(note.end_time),
                    "velocity": int(note.velocity),
                    "instrument": int(note.instrument) if hasattr(note, 'instrument') else 0,
                    "channel": int(note.channel) if hasattr(note, 'channel') else 0
                })
            
            # Update job status
            jobs[job_id]["progress"] = 90
            
            # Clean up temporary files
            os.unlink(audio_path)
            
            # Update job status with results
            jobs[job_id] = {
                "status": "completed",
                "progress": 100,
                "result": {
                    "success": True,
                    "midiData": midi_base64,
                    "noteEvents": note_events_serializable,
                    "conversionId": conversion_id
                }
            }
            
            logger.info(f"Conversion completed successfully for job {job_id}")
            
        except Exception as e:
            logger.error(f"Error in basic-pitch processing: {str(e)}")
            jobs[job_id] = {
                "status": "failed",
                "progress": 0,
                "error": str(e)
            }
    except Exception as e:
        logger.error(f"Error in background task: {str(e)}")
        jobs[job_id] = {
            "status": "failed",
            "progress": 0,
            "error": str(e)
        }

# API endpoints
@app.post("/convert", response_model=Dict[str, Any])
async def convert_audio(
    request: ConversionRequest,
    background_tasks: BackgroundTasks,
    _: Any = Depends(get_api_key)
):
    try:
        # Download the audio file
        logger.info(f"Processing conversion request for {request.conversionId}")
        audio_data = await download_file(request.audioUrl)
        
        # Start background processing
        job_id = request.conversionId
        background_tasks.add_task(
            process_audio_task,
            request.conversionId,
            audio_data,
            request.options
        )
        
        # Initialize job status
        jobs[job_id] = {"status": "pending", "progress": 0}
        
        # Return job ID for client to check status
        logger.info(f"Conversion job {job_id} started in background")
        return {
            "success": True,
            "status": "pending",
            "jobId": job_id,
            "message": "Conversion started in background"
        }
    except Exception as e:
        logger.error(f"Error starting conversion: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/job/{job_id}", response_model=Dict[str, Any])
async def get_job_status(job_id: str, _: Any = Depends(get_api_key)):
    if job_id not in jobs:
        logger.warning(f"Job not found: {job_id}")
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_status = jobs[job_id]
    logger.info(f"Retrieved job status for {job_id}: {job_status['status']}")
    
    # Clean up completed jobs after they're retrieved
    if job_status["status"] in ["completed", "failed"]:
        # Keep result but mark for cleanup
        job_status["cleanup"] = True
        
        # Schedule cleanup
        async def cleanup_job():
            await asyncio.sleep(300)  # Wait 5 minutes
            if job_id in jobs and jobs[job_id].get("cleanup"):
                del jobs[job_id]
                logger.info(f"Cleaned up job {job_id}")
        
        asyncio.create_task(cleanup_job())
    
    return job_status

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Main entry point
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Basic Pitch service")
    logger.info(f"Anonymous access: {os.environ.get('ALLOW_ANONYMOUS_ACCESS', 'false')}")
    logger.info(f"API key set: {bool(API_KEY)}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
