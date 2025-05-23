# RadioTranscriptor
A voice transcriptor for SDR

Using GPU as AI transcriptor (CUDA if available), defaults to CPU usage instead if CUDA is not available

Prequisites:
Python 3.8+

Make sure you have the following installed:

BASH:
pip install torch numpy sounddevice scipy openai-whisper

Ubuntu/Debian:
sudo apt update && sudo apt install ffmpeg

macOS:
brew install ffmpeg

-----------------
Hardware & Audio Notes
Make sure your microphone input device is set correctly (defaults to device=1, change as needed).

Works best with 48kHz input, automatically resampled to 16kHz for Whisper compatibility.

You should see:

“🎧 Listening for voice...”

“🎤 Voice started...” when speech is detected

Transcribed results printed and saved to a daily log file

![image](https://github.com/user-attachments/assets/ba757143-b110-4b12-97ff-cdba2311e146)


![image](https://github.com/user-attachments/assets/71a16d0d-2dc0-49b8-8919-8e2ff22ea611)

-----------------
Knows issues:
Transcriber does not always capture voice and comes up blank in the logs
Repeated words recorded



-----------------
💡 Customization Ideas
Swap Whisper model size: "tiny", "base", "small", "medium", "large"


👨‍💻 Acknowledgements
Huge thanks to ChatGPT (a.k.a. the whispering code goblin) for helping architect, debug, and document this project like a caffeinated co-pilot. 😄



![ChatGPT Image May 22, 2025, 10_14_34 PM](https://github.com/user-attachments/assets/4bde71ae-5f26-4431-b584-cdf119edcacc)

