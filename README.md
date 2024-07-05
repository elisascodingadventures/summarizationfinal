# Final summarization tool
An AI-powered transcription -> diarization -> summarization workflow

How to run:\
0 - (Optional) Set up and activate virtual environment:\
`python -m venv venv`\
`source venv/bin/activate`\
1 - Install packages:\
`pip install fastapi uvicorn pydantic pydub python-dotenv uvicorn pydantic pydub python-dotenv openai pyannote.audio torch speechbrain==1.0.0 ffmpeg openai-whisper`\
2 - In the same directory, create a .env file, paste your API key in said .env (`OPENAI_API_KEY=your_api_key`)\
3 - After navigating to your directory, run app.py via `python3 app.py`   
After navigating to your directory, run app.py via python3 app.py after navigating to your directory, run app.py via `python3 appy`\
4 - Upload your audio, make manual edits as directed, recieve your summary!
