"""
Test script to verify that Basic Pitch works correctly inside the Docker container.
This script should be run inside the container to check if the model loads and can process audio.
"""

import os
import sys
import tempfile
import numpy as np

print(f"Python version: {sys.version}")

try:
    # Import TensorFlow 
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    # Import Basic Pitch modules
    import basic_pitch
    print(f"Successfully imported basic_pitch!")
    
    from basic_pitch.note_creation import note_events_to_midi
    print("Successfully imported basic_pitch.note_creation module!")
    
    from basic_pitch.inference import predict as basic_pitch_predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    
    print(f"Basic Pitch model path: {ICASSP_2022_MODEL_PATH}")
    
    # Check if the model path exists
    if os.path.exists(ICASSP_2022_MODEL_PATH):
        print(f"Model file exists at {ICASSP_2022_MODEL_PATH}")
    else:
        print(f"ERROR: Model file not found at {ICASSP_2022_MODEL_PATH}")
    
    # Test if the environment has all necessary libraries
    import librosa
    import soundfile
    import scipy
    import midiutil
    import resampy
    import audioread
    
    print("All required libraries are available!")
    
    # Check for test audio file
    test_audio_files = [
        "./replicate-basic-pitch/.cog/test.mp3",
        "./test_audio.mp3",
        "./test.mp3",
        "./sample_audio/test.mp3"
    ]
    
    test_audio = None
    for test_file in test_audio_files:
        if os.path.exists(test_file):
            test_audio = test_file
            print(f"Found test audio file: {test_audio}")
            break
    
    if test_audio:
        print(f"Testing inference on {test_audio}")
        try:
            # Test loading the model and running inference
            model_output, audio_features = basic_pitch_predict(
                audio_path=test_audio,
                model_or_model_path=ICASSP_2022_MODEL_PATH,
                onset_threshold=0.5,
                frame_threshold=0.3,
                minimum_note_length=5,
                minimum_frequency=27.5,
                maximum_frequency=4186.0,
                multiple_pitch_bends=False,
                melodia_trick=True
            )
            
            # Create output directory if it doesn't exist
            os.makedirs("./output", exist_ok=True)
            
            # Path for the output MIDI file
            output_path = "./output/test_output.mid"
            
            # Write MIDI file
            note_events_to_midi(
                pitch_outputs=model_output,
                onset_outputs=audio_features["onset"],
                frame_rate=audio_features["frame_rate"],
                output_path=output_path,
                min_note_length=5,
                min_frequency=27.5,
                max_frequency=4186.0,
                melodia_trick=True,
                combine_notes=True,
                infer_onsets=True,
                multiple_notes_per_frame=True
            )
            
            if os.path.exists(output_path):
                print(f"SUCCESS: MIDI file generated at {output_path}")
            else:
                print(f"ERROR: Failed to generate MIDI file at {output_path}")
                
        except Exception as e:
            print(f"ERROR during inference: {e}")
    else:
        print("WARNING: No test audio file found. Please add a test audio file to run inference.")
        print("You can place an MP3 file in one of these locations:")
        for location in test_audio_files:
            print(f"  - {location}")
    
except Exception as e:
    print(f"Error during import or setup: {e}")
    raise

print("Test completed!") 