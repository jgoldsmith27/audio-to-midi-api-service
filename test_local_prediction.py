#!/usr/bin/env python
"""
Test script to run the Basic Pitch model locally on an audio file.
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

# Add the replicate-basic-pitch directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "replicate-basic-pitch"))

# Create a compatibility layer for the cog.Path class
# This allows us to use the same predict.py file locally and on Replicate
import cog
original_path = cog.Path

# Override the Path class in the cog module for local testing
class LocalPath(str):
    """A simple Path compatibility class for local testing with cog.Path"""
    def __new__(cls, path):
        return str.__new__(cls, path)

# Replace the Path class in the cog module with our LocalPath class
cog.Path = LocalPath

# Get the system temp directory
TMP_DIR = "/tmp"
if not os.path.exists(TMP_DIR):
    # Use system temp directory if /tmp doesn't exist
    TMP_DIR = tempfile.gettempdir()
    print(f"Using system temp directory: {TMP_DIR}")
else:
    print(f"Using standard temp directory: {TMP_DIR}")

# Import the Predictor class from our predict.py
from predict import Predictor

# Restore the original Path class after importing predict.py
cog.Path = original_path

def main():
    # Path to the audio file
    audio_file = "/Users/jacob/Desktop/[Golden] - Maddox [142 Dm].mp3"
    
    if not os.path.exists(audio_file):
        print(f"Error: File {audio_file} not found")
        return

    # Create our own output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "output.mid")

    # Initialize the predictor
    print("Initializing predictor...")
    predictor = Predictor()
    predictor.setup()
    
    # Run prediction
    print(f"Processing audio file: {audio_file}")
    start_time = time.time()
    
    # Run prediction with default parameters
    temp_output = predictor.predict(
        audio=audio_file,
        min_frequency=27.5,
        max_frequency=4186.0,
        min_note_length=5,
        energy_threshold=0.25,
        onset_threshold=0.25,
        melodia_filter=True,
        combine_notes=True,
        infer_onsets=True,
        multiple_notes_per_frame=True
    )
    
    elapsed = time.time() - start_time
    print(f"Processing completed in {elapsed:.2f} seconds")
    
    # Handle the temp output path that might be different in local testing
    expected_output = os.path.join(TMP_DIR, "output.mid")
    if os.path.exists(expected_output):
        # Copy from the temp location to our output file
        shutil.copy(expected_output, output_file)
    else:
        print(f"Warning: Expected output at {expected_output} not found.")
        # Try to use the returned path directly
        if os.path.exists(temp_output):
            shutil.copy(temp_output, output_file)
    
    if os.path.exists(output_file):
        print(f"MIDI file saved to: {output_file}")
        print(f"File size: {os.path.getsize(output_file)} bytes")
    else:
        print(f"Error: Failed to save MIDI file to {output_file}")

if __name__ == "__main__":
    main() 