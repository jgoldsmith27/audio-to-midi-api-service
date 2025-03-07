#!/usr/bin/env python3
"""
Test script for the Audio to MIDI API service
"""

import requests
import json
import base64
import time
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Test the Audio to MIDI API service')
    parser.add_argument('--url', type=str, default='http://localhost:8000',
                        help='Base URL of the API service')
    parser.add_argument('--audio', type=str, required=True,
                        help='URL of the audio file to convert')
    parser.add_argument('--output', type=str, default='output.mid',
                        help='Path to save the output MIDI file')
    parser.add_argument('--api-key', type=str,
                        help='API key for authentication (if required)')
    
    args = parser.parse_args()
    
    # Prepare headers
    headers = {}
    if args.api_key:
        headers['X-API-Key'] = args.api_key
    
    # Prepare conversion request
    conversion_request = {
        "audioUrl": args.audio,
        "conversionId": f"test-{int(time.time())}",
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
    
    # Submit conversion request
    print(f"Submitting audio URL: {args.audio}")
    response = requests.post(f"{args.url}/convert", 
                             json=conversion_request, 
                             headers=headers)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return
    
    job_data = response.json()
    job_id = job_data["jobId"]
    print(f"Job submitted with ID: {job_id}")
    
    # Poll for job completion
    while True:
        print("Checking job status...")
        status_response = requests.get(f"{args.url}/job/{job_id}", headers=headers)
        
        if status_response.status_code != 200:
            print(f"Error checking status: {status_response.status_code}")
            print(status_response.text)
            return
        
        job_status = status_response.json()
        status = job_status["status"]
        progress = job_status.get("progress", 0) * 100
        
        print(f"Status: {status}, Progress: {progress:.1f}%")
        
        if status == "completed":
            print("Conversion completed!")
            midi_base64 = job_status["result"]["midi"]
            
            # Save MIDI file
            with open(args.output, "wb") as midi_file:
                midi_file.write(base64.b64decode(midi_base64))
            
            print(f"MIDI file saved to {args.output}")
            break
        elif status == "failed":
            print(f"Conversion failed: {job_status.get('error', 'Unknown error')}")
            break
        
        # Wait before polling again
        time.sleep(2)

if __name__ == "__main__":
    main() 