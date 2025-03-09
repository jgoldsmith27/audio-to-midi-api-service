import os
import io
import tempfile
import json
import base64
import logging
import asyncio
import jwt
from typing import Dict, Any, List, Optional, Union
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, HttpUrl
import requests
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
oauth2_scheme = HTTPBearer(auto_error=False)

# Get JWT configuration from environment
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET")
SUPABASE_PUBLIC_KEY = os.environ.get("SUPABASE_PUBLIC_KEY")
ALLOW_ANONYMOUS_ACCESS = os.environ.get("ALLOW_ANONYMOUS_ACCESS", "false").lower() == "true"
ALLOW_INTERNAL_NETWORK = os.environ.get("ALLOW_INTERNAL_NETWORK", "false").lower() == "true"

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

# JWT verification function
async def verify_jwt_token(token: str) -> bool:
    if not token:
        logger.warning("No JWT token provided")
        return False
    
    try:
        # Log token structure and first part
        token_parts = token.split('.')
        logger.info(f"Token structure: {len(token_parts)} parts")
        
        if len(token_parts) != 3:
            logger.warning(f"Malformed JWT token: expected 3 parts, got {len(token_parts)}")
            # Log the token format if it's not a valid JWT
            if len(token_parts) > 0 and len(token) < 100:
                logger.info(f"Token format: {token}")
            return False
        
        # Try to decode header without verification to log info
        try:
            import base64
            import json
            
            # Decode header
            header_padding = token_parts[0] + '=' * (4 - len(token_parts[0]) % 4)
            header_json = base64.b64decode(header_padding).decode('utf-8')
            header = json.loads(header_json)
            logger.info(f"Token header: {header}")
            
            # Decode payload
            payload_padding = token_parts[1] + '=' * (4 - len(token_parts[1]) % 4)
            payload_json = base64.b64decode(payload_padding).decode('utf-8')
            payload = json.loads(payload_json)
            
            # Log important claims
            logger.info(f"Token issuer (iss): {payload.get('iss')}")
            logger.info(f"Token audience (aud): {payload.get('aud')}")
            logger.info(f"Token subject (sub): {payload.get('sub', 'not present')}")
            logger.info(f"Token expiration (exp): {payload.get('exp')}")
            logger.info(f"Expected issuer: {SUPABASE_URL}")
            
            # Check if the token is a Supabase token
            if 'supabase' not in str(payload.get('iss', '')).lower():
                logger.warning("Token doesn't appear to be a Supabase token based on issuer")
        except Exception as e:
            logger.warning(f"Error decoding token for inspection: {str(e)}")
        
        # Try to decode with SUPABASE_JWT_SECRET (HS256)
        if SUPABASE_JWT_SECRET:
            try:
                logger.info(f"Attempting HS256 verification with JWT_SECRET (length: {len(SUPABASE_JWT_SECRET)})")
                
                # Try with more permissive options for debugging
                options = {
                    "verify_signature": True,
                    "verify_aud": False,  # Don't verify audience claim
                    "verify_iss": False,  # We'll check issuer manually
                    "require_exp": True
                }
                
                payload = jwt.decode(
                    token,
                    SUPABASE_JWT_SECRET,
                    algorithms=["HS256"],
                    options=options
                )
                issuer = payload.get("iss")
                logger.info(f"HS256 verification succeeded, issuer: {issuer}")
                if issuer and SUPABASE_URL:
                    # Allow any Supabase URL as issuer for flexibility
                    if "supabase" in str(issuer).lower():
                        logger.info(f"JWT verified using HS256 algorithm with Supabase issuer: {issuer}")
                        return True
                    elif issuer == SUPABASE_URL:
                        logger.info(f"JWT verified using HS256 algorithm with exact issuer match: {issuer}")
                        return True
                    else:
                        logger.warning(f"JWT issuer mismatch: {issuer} != {SUPABASE_URL}")
            except Exception as e:
                logger.info(f"HS256 verification failed: {str(e)}")
                logger.info(f"Error type: {type(e).__name__}")
                if hasattr(e, '__dict__'):
                    logger.info(f"Error details: {e.__dict__}")
        else:
            logger.warning("SUPABASE_JWT_SECRET is not set, skipping HS256 verification")
        
        # Try to decode with SUPABASE_PUBLIC_KEY (RS256)
        if SUPABASE_PUBLIC_KEY:
            try:
                logger.info(f"Attempting RS256 verification with PUBLIC_KEY (length: {len(SUPABASE_PUBLIC_KEY)})")
                # Check if the public key is a JWT itself or an actual public key
                if SUPABASE_PUBLIC_KEY.startswith("eyJ"):
                    logger.warning("SUPABASE_PUBLIC_KEY appears to be a JWT token, not an actual public key")
                
                # Try with more permissive options for debugging
                options = {
                    "verify_signature": True,
                    "verify_aud": False,  # Don't verify audience claim
                    "verify_iss": False,  # We'll check issuer manually
                    "require_exp": True
                }
                
                payload = jwt.decode(
                    token,
                    SUPABASE_PUBLIC_KEY,
                    algorithms=["RS256"],
                    options=options
                )
                issuer = payload.get("iss")
                logger.info(f"RS256 verification succeeded, issuer: {issuer}")
                if issuer and SUPABASE_URL:
                    # Allow any Supabase URL as issuer for flexibility
                    if "supabase" in str(issuer).lower():
                        logger.info(f"JWT verified using RS256 algorithm with Supabase issuer: {issuer}")
                        return True
                    elif issuer == SUPABASE_URL:
                        logger.info(f"JWT verified using RS256 algorithm with exact issuer match: {issuer}")
                        return True
                    else:
                        logger.warning(f"JWT issuer mismatch: {issuer} != {SUPABASE_URL}")
            except Exception as e:
                logger.info(f"RS256 verification failed: {str(e)}")
                logger.info(f"Error type: {type(e).__name__}")
                if hasattr(e, '__dict__'):
                    logger.info(f"Error details: {e.__dict__}")
        else:
            logger.warning("SUPABASE_PUBLIC_KEY is not set, skipping RS256 verification")
        
        # Try with None algorithm as a fallback (just for debugging)
        try:
            # This is just for debugging, not for actual verification
            decoded = jwt.decode(token, options={"verify_signature": False})
            logger.info(f"JWT decoded without verification: {decoded}")
            logger.info(f"JWT claims: {', '.join(decoded.keys())}")
        except Exception as e:
            logger.warning(f"Failed to decode JWT even without verification: {str(e)}")
        
        logger.warning("JWT verification failed for all methods")
        return False
    except Exception as e:
        logger.error(f"Unexpected error verifying JWT: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# Enhanced security dependency with JWT support
async def get_api_key(
    api_key_header: str = Header(None, alias=API_KEY_NAME),
    auth: HTTPAuthorizationCredentials = Depends(oauth2_scheme),
    request: Request = None
):
    # Log request details for debugging
    client_info = "unknown"
    if request and request.client:
        client_info = request.client.host
    
    logger.info(f"Request from {client_info} to {request.url.path}")
    
    # Log headers (excluding sensitive parts)
    headers_log = {}
    if request.headers:
        for key, value in request.headers.items():
            if key.lower() in ["authorization", "x-api-key"]:
                headers_log[key] = value[:10] + "..." if value and len(value) > 10 else "[empty]"
            else:
                headers_log[key] = value
        logger.info(f"Request headers: {headers_log}")
    
    # Check if anonymous access is enabled - this takes precedence
    if ALLOW_ANONYMOUS_ACCESS:
        logger.info("Allowing anonymous access - ALLOW_ANONYMOUS_ACCESS is enabled")
        return True
    
    # Check for JWT token in Authorization header
    if auth and auth.credentials:
        scheme = auth.scheme.lower() if hasattr(auth, 'scheme') else "bearer"
        logger.info(f"Found {scheme} token in Authorization header, verifying...")
        is_valid = await verify_jwt_token(auth.credentials)
        if is_valid:
            logger.info("JWT token successfully verified")
            return True
        else:
            logger.warning("JWT token verification failed")
    else:
        logger.warning("No Authorization header with Bearer token found")
    
    # Check if internal network access is allowed
    if ALLOW_INTERNAL_NETWORK and request and request.client:
        client_host = request.client.host
        logger.info(f"Checking network security for: {client_host}")
        
        # Trust requests from localhost and internal networks
        if client_host == "127.0.0.1" or client_host.startswith("10.") or \
           client_host.startswith("172.16.") or client_host.startswith("192.168."):
            logger.info(f"Request from trusted network: {client_host}")
            return True
    
    # If we get here, authentication failed
    logger.warning("Authentication failed: Request didn't pass JWT verification or network security, and anonymous access not enabled")
    raise HTTPException(
        status_code=401,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
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
            
            # Try all possible import methods
            try:
                # First try the standard import
                from basic_pitch.inference import predict_and_save, predict
                predict_func = predict
                logger.info("Successfully imported standard basic_pitch predict function")
            except ImportError:
                # Fallback to direct model loading
                logger.warning("Could not import standard predict function, attempting fallback")
                try:
                    # Try the command-line script's functions
                    from basic_pitch.commandline import predict_and_save as cmdline_predict_and_save
                    predict_func = lambda audio_path, **kwargs: cmdline_predict_and_save(
                        [audio_path], 
                        output_dir, 
                        save_midi=True,
                        save_model_outputs=False,
                        save_notes=False,
                        sonify_midi=False,
                        **kwargs
                    )
                    logger.info("Using commandline predict_and_save as fallback")
                except ImportError:
                    logger.error("All fallback import methods failed")
                    raise ImportError("Could not import any basic_pitch functions")
            
            # Update job status
            jobs[job_id]["progress"] = 20
            
            # Store the original parameters for logging purposes
            original_params = {
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
            
            # Based on documentation, the basic-pitch predict() function only accepts
            # a limited set of parameters. We'll use only the parameters it can handle.
            basic_pitch_params = {
                "minimum_frequency": options.minFrequency,
                "maximum_frequency": options.maxFrequency
                # Other parameters are not supported by the predict function
            }
            
            # Update job status
            jobs[job_id]["progress"] = 30
            
            # Call basic-pitch
            logger.info(f"Processing audio with basic-pitch: {audio_path}")
            logger.info(f"Original parameters: {original_params}")
            logger.info(f"Using valid parameters: {basic_pitch_params}")
            
            # Start the prediction process
            model_output, midi_data, note_events = predict_func(
                audio_path,
                **basic_pitch_params
            )
            
            # Update job status
            jobs[job_id]["progress"] = 70
            
            # Save files if requested
            output_file_base = os.path.join(output_dir, f"basic_pitch_{conversion_id}")
            
            # Get the MIDI data as bytes
            midi_bytes = io.BytesIO()
            midi_data.write(midi_bytes)
            midi_bytes.seek(0)
            midi_data_bytes = midi_bytes.read()
            
            # Convert MIDI data to base64
            midi_base64 = base64.b64encode(midi_data_bytes).decode('utf-8')
            
            # Convert note events to serializable format
            note_events_serializable = []
            for note in note_events:
                note_events_serializable.append({
                    "pitch": int(note[2]),  # pitch_midi is at index 2
                    "start_time": float(note[0]),  # start_time_s is at index 0
                    "end_time": float(note[1]),  # end_time_s is at index 1
                    "velocity": int(note[3] * 127) if note[3] <= 1.0 else int(note[3]),  # amplitude is at index 3, convert to velocity
                    "instrument": 0,  # Default to instrument 0
                    "channel": 0  # Default to channel 0
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
    logger.info(f"Anonymous access: {ALLOW_ANONYMOUS_ACCESS}")
    logger.info(f"Internal network access: {ALLOW_INTERNAL_NETWORK}")
    logger.info(f"JWT config: URL={bool(SUPABASE_URL)}, Secret={bool(SUPABASE_JWT_SECRET)}, Public Key={bool(SUPABASE_PUBLIC_KEY)}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
