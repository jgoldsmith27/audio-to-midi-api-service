"""
Script to verify a test audio file exists for Basic Pitch testing.
"""

import os
from pathlib import Path

if __name__ == "__main__":
    # Check for test audio files in common locations
    test_audio_files = [
        "./test_audio.mp3",
        "./test.mp3",
        "./replicate-basic-pitch/.cog/test.mp3",
        "./sample_audio/test.mp3"
    ]
    
    found = False
    for test_file in test_audio_files:
        if os.path.exists(test_file):
            print(f"Found test audio file at: {test_file}")
            found = True
            break
    
    if not found:
        print("No test audio file found. Please place a test audio file in one of these locations:")
        for location in test_audio_files:
            print(f"  - {location}")
        print("This file will be used for testing the Basic Pitch model in the container.") 