import speech_recognition as sr
from langdetect import detect

def get_language(language_code):
    language_mapping = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'nl': 'Dutch',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'ta': 'Tamil',
        'kn': 'Kannada'
    }
    return language_mapping.get(language_code, 'Unknown')

def convert_audio_to_text(audio_file_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)  # Record the audio

        try:
            text = recognizer.recognize_google(audio_data, language='en')  # Always transcribe to English
            detected_language = detect(text)

            return text, detected_language
        except sr.UnknownValueError:
            return "Could not understand audio", None
        except sr.RequestError as e:
            return f"Error: {e}", None

audio_file_path = "/content/sample_data/random.wav"
result, detected_language = convert_audio_to_text(audio_file_path)

print("Converted Text:", result)

if detected_language:
    full_language_name = get_language(detected_language)
    print("Detected Language:", full_language_name)