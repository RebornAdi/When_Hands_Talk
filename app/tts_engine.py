"""
tts_engine.py
-------------
Text-to-speech wrapper for the final "text -> voice" stage.

Default backend: pyttsx3
    - fully offline, no API key, no internet needed
    - works immediately after `pip install pyttsx3`
    - voice quality is robotic but totally fine for a demo

Optional backend: gTTS
    - needs internet + `pip install gTTS`
    - much more natural-sounding voice, good for a polished demo video

Switch ENGINE_BACKEND below depending on what you want.
"""

import os
import tempfile

ENGINE_BACKEND = "pyttsx3"  # "pyttsx3" or "gtts"


def synthesize_to_file(text, out_path=None):
    """
    Synthesizes speech to an audio file and returns the file path.
    This is the one to use from the Flask app: synthesize server-side,
    then send the resulting audio file back to the browser to play.
    (Speaking directly out of the server's speakers, via speak_offline()
    below, only makes sense for a local desktop demo - not a web app,
    since the server and the user are different machines.)
    """
    if not text.strip():
        raise ValueError("No text to synthesize - sentence is empty.")

    if out_path is None:
        suffix = ".mp3" if ENGINE_BACKEND == "gtts" else ".wav"
        fd, out_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)

    if ENGINE_BACKEND == "gtts":
        from gtts import gTTS
        gTTS(text=text, lang="en").save(out_path)
    else:
        import pyttsx3
        engine = pyttsx3.init()
        engine.save_to_file(text, out_path)
        engine.runAndWait()

    return out_path


def speak_offline(text, rate=170, voice_index=None):
    """Speaks immediately through the local machine's speakers.
    Handy for quick local testing (e.g. inside 4_real_time_recognition.py),
    not for the deployed Flask app."""
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    if voice_index is not None:
        voices = engine.getProperty("voices")
        engine.setProperty("voice", voices[voice_index].id)
    engine.say(text)
    engine.runAndWait()


if __name__ == "__main__":
    path = synthesize_to_file("Hello, this is a test of the sign language voice output.")
    print(f"Audio written to: {path}")
