#!/usr/bin/env python
"""Test script for basic-pitch imports"""

# Import Basic Pitch modules
import basic_pitch
from basic_pitch.inference import predict as basic_pitch_predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.note_creation import note_events_to_midi

print("All imports successful!")
print(f"note_events_to_midi: {note_events_to_midi}") 