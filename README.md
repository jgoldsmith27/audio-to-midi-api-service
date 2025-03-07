# Audio to MIDI API Service

A powerful API service that converts audio files to MIDI format using [Basic Pitch](https://github.com/spotify/basic-pitch) technology.

## Features

- Convert audio files (MP3, WAV, etc.) to MIDI format
- REST API with FastAPI
- Background job processing
- Customizable conversion options
- Health check endpoint

## Tech Stack

- Python 3.10
- FastAPI
- Basic Pitch (Spotify's neural network for music transcription)
- TensorFlow
- Librosa for audio processing
- Render for deployment

## API Endpoints

- `POST /convert`: Submit an audio file URL for conversion to MIDI
- `GET /job/{job_id}`: Check the status of a conversion job
- `GET /health`: Health check endpoint

## Local Development

### Prerequisites

- Python 3.10
- pip

### Setup

1. Clone this repository:
```bash
git clone https://github.com/your-username/audio-to-midi-api-service.git
cd audio-to-midi-api-service
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the development server:
```bash
uvicorn main:app --reload
```

The API will be available at http://localhost:8000.

## Deployment on Render

This service is configured for easy deployment on Render.com:

1. Create a new Web Service on Render
2. Connect to your GitHub repository
3. Use the following settings:
   - Environment: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

For more detailed deployment instructions, see [Render's Python deployment guide](https://render.com/docs/deploy-python).

## API Usage Example

```python
import requests
import json

# Submit audio for conversion
conversion_request = {
    "audioUrl": "https://example.com/audio-file.mp3",
    "conversionId": "unique-conversion-id",
    "options": {
        "minFrequency": 27.5,
        "maxFrequency": 4186.0,
        "minNoteLength": 5,
        "energyThreshold": 0.25,
        "onsetThreshold": 0.25,
        "melodiaFilter": True,
        "combineNotes": True,
        "inferOnsets": True,
        "multipleNotesPerFrame": True,
        "outputFormat": {
            "midi": True,
            "csv": False,
            "npz": False,
            "wav": False
        }
    }
}

response = requests.post("https://your-service.onrender.com/convert", json=conversion_request)
job_data = response.json()
job_id = job_data["jobId"]

# Check job status
status_response = requests.get(f"https://your-service.onrender.com/job/{job_id}")
job_status = status_response.json()

# If job is complete, get the MIDI data
if job_status["status"] == "completed":
    midi_base64 = job_status["result"]["midi"]
    # Convert base64 to MIDI file
    import base64
    with open("output.mid", "wb") as midi_file:
        midi_file.write(base64.b64decode(midi_base64))
```

## License

MIT

## Acknowledgments

- [Basic Pitch](https://github.com/spotify/basic-pitch) by Spotify
- [FastAPI](https://fastapi.tiangolo.com/) 