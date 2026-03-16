import speech_recognition as sr
import os
from dotenv import load_dotenv
from gtts import gTTS
import playsound3
import requests
import json


# Load environment variables from .env file
load_dotenv()

#     raise ValueError("API key not found. Please set it in the .env file or environment variables.")

# Ollama server URL
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')  # Default Ollama URL, can be set in .env
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5-coder:0.5b')  # Default to an available model

print(f"[DEBUG] Using Ollama URL: {OLLAMA_URL}")
print(f"[DEBUG] Using Ollama Model: {OLLAMA_MODEL}")

# Initialize the recognizer
recognizer = sr.Recognizer()

# Function to convert text to speech using gTTS
def speak(text):
    try:
        tts = gTTS(text=text, lang='en',slow=False)
        filename = "temp_speech.mp3"
        tts.save(filename)
        playsound3.playsound(filename)
        os.remove(filename)  # Remove the temporary file
    except Exception as e:
        print(f"Error during speech synthesis: {e}")

# Function to recognize speech and convert it to text
def listen():
    with sr.Microphone() as source:
        print("Listening... Calibrating for ambient noise.")
        recognizer.adjust_for_ambient_noise(source, duration=0.15)  # Even shorter calibration
        print("You can speak now.")
        try:
            audio = recognizer.listen(source, timeout=4, phrase_time_limit=5)  # Quicker response
            text = recognizer.recognize_google(audio, language='en-US')
            print(f"You said: {text}")
            return text
        except sr.WaitTimeoutError:
            print("Listening timed out, please try speaking a bit sooner.")
            return None
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Could not request results; check your network connection.")
            return None


# Prompt template for consistent system behavior
PROMPT_TEMPLATE = (
    "You are Caren, a helpful AI assistant. "
    "Answer clearly and concisely.\n"
    "Do not include any disclaimers or apologies.\n"
    "If the user explicitly asks for code, provide only the code, without markdown or code blocks.\n"
    "Otherwise, respond in natural language.\n"
    "Do not repeat or rephrase the user's question. Always provide a helpful, original answer.\n"
    "Do not mention that you are produced by alibaba or any other company.\n"
    "User: {user_input}\n"
    "Caren:"
)

# Function to interact with Ollama
def ollama_query(query, history=None):
    api_url = f"{OLLAMA_URL}/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    # Use the prompt template, including history if provided
    if history:
        prompt = f"{history}\nUser: {query}\nCaren:"
    else:
        prompt = PROMPT_TEMPLATE.format(user_input=query)
    data = {
        "prompt": prompt,
        "model": OLLAMA_MODEL,  # Use the variable, not a string
        "stream": False  # Set to False for a single response, True for streaming
    }
    try:
        response = requests.post(api_url, headers=headers, json=data, stream=False)
        if response.status_code != 200:
            print(f"[ERROR] Ollama response status: {response.status_code}")
            print(f"[ERROR] Ollama response body: {response.text}")
            return f"Ollama error: {response.status_code} - {response.text}"
        # Parse the JSON response
        json_data = response.json()
        # Extract the 'response' field from the JSON
        text_response = json_data.get('response', "No response from Ollama")
        return text_response
    except requests.exceptions.RequestException as e:
        return f"API Error: {e}"
    except json.JSONDecodeError as e:
        return f"Error decoding JSON response: {e}\nRaw response: {response.text}"
    except KeyError as e:
        return f"Error parsing JSON response: Missing key: {e}\nRaw response: {response.text}"

# Main function to run the assistant
def main():
    speak("Hello ,Caren here! how can I help you today?")
    while True:
        query = listen()
        if query:
            if query.lower() in ["exit", "quit", "stop"]:
                speak("Goodbye!. It was nice speaking with you ,hope to see you again!")
                break
            response = ollama_query(query)
            speak(response)

if __name__ == "__main__":
    main()







