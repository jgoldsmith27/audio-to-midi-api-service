import os
import sys
import tempfile
from typing import List, Optional
from cog import BasePredictor, Input, Path
import numpy as np

# Print Python version for debugging
print(f"Python version: {sys.version}")

# Import TensorFlow with proper error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    
    # Configure TensorFlow logging
    tf.get_logger().setLevel('ERROR')
    
    # Explicitly import Basic Pitch components
    from basic_pitch.inference import predict as basic_pitch_predict
    from basic_pitch.note_creation import note_events_to_midi
    from basic_pitch import ICASSP_2022_MODEL_PATH
    
    print("Successfully imported all required modules!")
    
except ImportError as e:
    print(f"Import error: {e}")
    raise

# Ensure the /tmp directory exists for Replicate compatibility
TMP_DIR = "/tmp"
if not os.path.exists(TMP_DIR):
    TMP_DIR = tempfile.gettempdir()
    print(f"Using system temp directory: {TMP_DIR}")
else:
    print(f"Using standard temp directory: {TMP_DIR}")

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Check if TensorFlow can see the GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"TensorFlow is using GPU: {gpus}")
            # Set memory growth to avoid allocating all GPU memory
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception as e:
                    print(f"Error setting memory growth: {e}")
        else:
            print("WARNING: No GPU found. Running on CPU which may be slower.")
        
        # Verify model path exists
        if not os.path.exists(ICASSP_2022_MODEL_PATH):
            print(f"WARNING: Model path does not exist: {ICASSP_2022_MODEL_PATH}")
        else:
            print(f"Model path verified: {ICASSP_2022_MODEL_PATH}")
            
    def predict(
        self,
        audio: Path = Input(description="Audio file to convert to MIDI. Supported formats: mp3, wav, etc."),
        min_frequency: float = Input(description="Minimum frequency in Hz", default=27.5),
        max_frequency: float = Input(description="Maximum frequency in Hz", default=4186.0),
        min_note_length: int = Input(description="Minimum note length (in frames)", default=5),
        energy_threshold: float = Input(description="Energy threshold for note detection", default=0.25),
        onset_threshold: float = Input(description="Onset threshold for note detection", default=0.25),
        melodia_filter: bool = Input(description="Apply Melodia filter for cleaner transcription", default=True),
        combine_notes: bool = Input(description="Combine consecutive notes of same pitch", default=True),
        infer_onsets: bool = Input(description="Infer onsets from note energy", default=True),
        multiple_notes_per_frame: bool = Input(description="Allow multiple notes per frame", default=True),
    ) -> Path:
        """Run Basic Pitch model inference on the input audio"""
        
        print(f"Processing audio file: {audio}")
        
        try:
            # Create temp directory for outputs
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert audio to MIDI
                model_output, audio_features = basic_pitch_predict(
                    audio_path=str(audio),
                    model_or_model_path=ICASSP_2022_MODEL_PATH,
                    onset_threshold=onset_threshold,
                    frame_threshold=energy_threshold,
                    minimum_note_length=min_note_length,
                    minimum_frequency=min_frequency,
                    maximum_frequency=max_frequency,
                    multiple_pitch_bends=False,
                    melodia_trick=melodia_filter
                )
                
                # Path for the output MIDI file
                output_path = os.path.join(temp_dir, "output.mid")
                
                # Write MIDI file
                note_events_to_midi(
                    pitch_outputs=model_output,
                    onset_outputs=audio_features["onset"],
                    frame_rate=audio_features["frame_rate"],
                    output_path=output_path,
                    min_note_length=min_note_length,
                    min_frequency=min_frequency,
                    max_frequency=max_frequency,
                    melodia_trick=melodia_filter,
                    combine_notes=combine_notes,
                    infer_onsets=infer_onsets,
                    multiple_notes_per_frame=multiple_notes_per_frame
                )
                
                # Create a copy of the file in a new Path location that Replicate can access
                final_output = Path(os.path.join(TMP_DIR, "output.mid"))
                with open(output_path, "rb") as f_in:
                    with open(final_output, "wb") as f_out:
                        f_out.write(f_in.read())
                
                print(f"Successfully created MIDI file at {final_output}")
                return final_output
                
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            raise 