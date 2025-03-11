# Basic Pitch - Audio to MIDI Converter

This is a Replicate implementation of Spotify's [Basic Pitch](https://github.com/spotify/basic-pitch) model for automatic music transcription. Basic Pitch is a deep learning model that can convert audio files (WAV, MP3, etc.) to MIDI.

## Model Description

Basic Pitch uses deep learning to detect notes in audio recordings and transcribe them into MIDI format. It works well for both monophonic (single-note) and polyphonic (multiple-note) audio.

## Input

- Audio file (mp3, wav, etc.)
- Various parameters to control the transcription process

## Output

- MIDI file containing the transcribed notes

## Example Usage

```python
import replicate

output = replicate.run(
    "jgoldsmith27/basic-pitch:latest",
    input={
        "audio": open("your_song.mp3", "rb"),
        "energy_threshold": 0.3,
        "onset_threshold": 0.4
    }
)

# Output is a URL to download the MIDI file
print(output)
```

## Credits

This model is based on Spotify's open-source Basic Pitch library: https://github.com/spotify/basic-pitch 