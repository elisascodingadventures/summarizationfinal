# Final summarization tool
An AI-power transcription -> diarization -> summarization workflow

How to run:\
0. Optional - Set up and activate virtual environment:\
`python -m venv venv`\
`source venv/bin/activate`
1. Install packages:\
`pip install fastapi uvicorn pydantic pydub python-dotenv openai pyannote.audio torch speechbrain==1.0.0 ffmpeg openai-whisper`
2. After navigating to your directory, run app.py via `python3 app.py`. 
3. Upload your audio, make manual edits, recieve your summary!
