from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from jinja2 import Template
import os
import subprocess
import torch
import whisper
from pyannote.audio import Audio
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
import wave
import contextlib
import numpy as np
import pandas as pd
import datetime
import logging
import sqlite3
import openai

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Initialize the embedding model
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

def convert_to_mono(input_path, output_path='mono.wav'):
    """Convert the input audio file to mono using ffmpeg."""
    cmd = f'ffmpeg -i "{input_path}" -y -ac 1 "{output_path}"'
    try:
        logging.info(f'Running command: {cmd}')
        subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        logging.info(f'Converted {input_path} to mono.')
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during ffmpeg command: {e.output.decode()}")
        raise

# Add logging to check if ffmpeg is installed
try:
    subprocess.check_output("ffmpeg -version", shell=True, stderr=subprocess.STDOUT)
    logging.info("ffmpeg is installed.")
except subprocess.CalledProcessError:
    logging.error("ffmpeg is not installed or not found in PATH.")
    raise

def extract_speakers(model, path, num_speakers=0):
    """Perform diarization with speaker names."""
    mono = 'mono.wav'
    convert_to_mono(path, mono)

    result = model.transcribe(mono)
    segments = result["segments"]

    with contextlib.closing(wave.open(mono, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    audio = Audio()

    def segment_embedding(segment):
        start = segment["start"]
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        waveform, sample_rate = audio.crop(mono, clip)
        return embedding_model(waveform[None])

    embeddings = np.zeros((len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment)
    embeddings = np.nan_to_num(embeddings)

    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_
    for i in range(len(segments)):
        segments[i]["speaker"] = 'Speaker ' + str(labels[i] + 1)
    return segments

def write_segments(segments, outfile):
    """Write out segments to a file."""
    def time(secs):
        return str(datetime.timedelta(seconds=round(secs)))

    with open(outfile, "w") as f:
        previous_speaker = None
        for segment in segments:
            if segment["speaker"] == previous_speaker:
                f.write(' ' + segment["text"].strip())
            else:
                if previous_speaker is not None:
                    f.write('\n')
                f.write(segment["speaker"] + ' ' + time(segment["start"]) + '\n')
                f.write(segment["text"].strip())
            previous_speaker = segment["speaker"]
        f.write('\n')

def format_output_file(outfile):
    """Return the contents of the output file formatted as text."""
    with open(outfile, "r") as f:
        content = f.read()
    return content

def parse_transcript(text):
    """Parse the transcript text into structured data."""
    lines = text.strip().split('\n')

    speakers = []
    sentences = []
    chronology = []
    
    current_speaker = None
    order = 1

    for line in lines:
        line = line.strip()
        if line.startswith("Speaker"):
            current_speaker = line.split()[1].replace(":", "")  # Extract the speaker number and remove the colon
            current_speaker = f'Speaker {current_speaker}'  # Ensure proper format
        elif line:
            speakers.append(current_speaker)
            sentences.append(line)
            chronology.append(order)
            order += 1
    
    data = {
        "chronology": chronology,
        "sentences": sentences,
        "speakers": speakers
    }

    return data

def reassign_part_of_string(df, index, part_to_reassign, new_speaker_name):
    """Reassign part of a string to a new speaker."""
    if 0 <= index < len(df):
        original_text = df.at[index, 'sentences']
        if part_to_reassign in original_text:
            parts = original_text.split(part_to_reassign)
            new_rows = [
                {"chronology": df.at[index, 'chronology'], "sentences": parts[0], "speakers": df.at[index, 'speakers']},
                {"chronology": df.at[index, 'chronology'], "sentences": part_to_reassign, "speakers": new_speaker_name},
                {"chronology": df.at[index, 'chronology'], "sentences": parts[1], "speakers": df.at[index, 'speakers']}
            ]
            df.drop(index, inplace=True)
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            df.sort_values(by="chronology", inplace=True)  # Keep the chronology order
        else:
            print(f"Part '{part_to_reassign}' not found in the original text.")
    else:
        print(f"Index {index} is out of range.")
    return df

def apply_sentence_correction(df, index, incorrect_sentence, correct_sentence):
    """Correct the spelling of a sentence in the transcript."""
    if 0 <= index < len(df):
        original_text = df.at[index, 'sentences']
        if incorrect_sentence in original_text:
            corrected_text = original_text.replace(incorrect_sentence, correct_sentence)
            df.at[index, 'sentences'] = corrected_text
        else:
            print(f"Sentence '{incorrect_sentence}' not found in the original text.")
    else:
        print(f"Index {index} is out of range.")
    return df

def format_script(df):
    """Format the dataframe into a script format."""
    script = ""
    for _, row in df.iterrows():
        script += f"{row['speakers']}: {row['sentences']}\n"
    return script

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload Audio File</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f4f4f9;
            }
            h1, h2, h3 {
                color: #333;
            }
            form {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-top: 10px;
            }
            input[type="file"], input[type="text"], input[type="number"] {
                width: 100%;
                padding: 10px;
                margin-top: 5px;
            }
            button {
                background-color: #007bff;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer.
            }
            button:hover {
                background-color: #0056b3.
            }
            table {
                width: 100%.
                border-collapse: collapse.
                margin-top: 20px.
            }
            table, th, td {
                border: 1px solid #ddd.
            }
            th, td {
                padding: 10px.
                text-align: left.
            }
            th {
                background-color: #f2f2f2.
            }
            pre {
                background-color: #333.
                color: #f8f8f2.
                padding: 20px.
                border-radius: 5px.
                overflow-x: auto.
            }
        </style>
        <script>
            async function assignNames() {
                const formData = new FormData(document.getElementById('assign-names-form'));
                const response = await fetch('/assign-names/', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.text();
                document.open();
                document.write(result);
                document.close();
            }

            async function editTranscript() {
                const formData = new FormData(document.getElementById('edit-transcript-form'));
                const response = await fetch('/edit-transcript/', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.text();
                document.open();
                document.write(result);
                document.close();
            }

            async function correctSpelling() {
                const formData = new FormData(document.getElementById('correct-spelling-form'));
                const response = await fetch('/correct-spelling/', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.text();
                document.open();
                document.write(result);
                document.close();
            }

            async function processAudio() {
                const formData = new FormData(document.getElementById('upload-form'));
                const response = await fetch('/process-audio/', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.text();
                document.open();
                document.write(result);
                document.close();
            }

            async function finishEditing() {
                const response = await fetch('/finish-editing/', {
                    method: 'POST'
                });
                const result = await response.text();
                document.getElementById('script-output').innerText = result;
            }

            async function summarizeScript() {
                const response = await fetch('/summarize-script/', {
                    method: 'POST'
                });
                const result = await response.text();
                document.getElementById('summary-output').innerText = result;
            }
        </script>
    </head>
    <body>
        <h1>Upload Audio File for Processing</h1>
        <form id="upload-form" onsubmit="event.preventDefault(); processAudio();" enctype="multipart/form-data">
            <input type="file" name="file" accept="audio/*" required>
            <label for="num_speakers">Number of Speakers:</label>
            <input type="number" id="num_speakers" name="num_speakers" min="1" max="10" value="3">
            <button type="submit">Upload</button>
        </form>
        {% if transcripts %}
            <h2>Transcript Data</h2>
            {% if transcripts.error %}
                <p>{{ transcripts.error }}</p>
            {% else %}
                <table>
                    <tr>
                        <th>Index</th>
                        <th>Chronology</th>
                        <th>Speaker</th>
                        <th>Sentence</th>
                    </tr>
                    {% for i in range(transcripts.chronology|length) %}
                    <tr>
                        <td>{{ i }}</td>
                        <td>{{ transcripts.chronology[i] }}</td>
                        <td>{{ transcripts.speakers[i] }}</td>
                        <td>{{ transcripts.sentences[i] }}</td>
                    </tr>
                    {% endfor %}
                </table>

                <h3>Assign Names to Speakers</h3>
                <form id="assign-names-form" onsubmit="event.preventDefault(); assignNames();">
                    {% for speaker in transcripts.speakers|unique %}
                        <label for="name_{{ speaker }}">Name for {{ speaker }}:</label>
                        <input type="text" id="name_{{ speaker }}" name="{{ speaker }}" required><br>
                    {% endfor %}
                    <button type="submit">Assign Names</button>
                </form>

                <h3>Edit Transcript</h3>
                <form id="edit-transcript-form" onsubmit="event.preventDefault(); editTranscript();">
                    <label for="index_to_reassign">Index of Text to Reassign:</label>
                    <input type="number" id="index_to_reassign" name="index_to_reassign" min="0" required><br>
                    <label for="part_to_reassign">Part to Reassign:</label>
                    <input type="text" id="part_to_reassign" name="part_to_reassign" required><br>
                    <label for="new_speaker_name">New Speaker Name:</label>
                    <input type="text" id="new_speaker_name" name="new_speaker_name" required><br>
                    <button type="submit">Reassign</button>
                </form>

                <h3>Correct Spelling</h3>
                <form id="correct-spelling-form" onsubmit="event.preventDefault(); correctSpelling();">
                    <label for="index_to_correct">Index of Text to Correct:</label>
                    <input type="number" id="index_to_correct" name="index_to_correct" min="0" required><br>
                    <label for="incorrect_sentence">Incorrect Sentence:</label>
                    <input type="text" id="incorrect_sentence" name="incorrect_sentence" required><br>
                    <label for="correct_sentence">Correct Sentence:</label>
                    <input type="text" id="correct_sentence" name="correct_sentence" required><br>
                    <button type="submit">Correct</button>
                </form>

                <h3>Finish Editing Transcription</h3>
                <button onclick="finishEditing()">Finish Editing</button>

                <h3>Formatted Script</h3>
                <pre id="script-output"></pre>

                <h3>Summary</h3>
                <button onclick="summarizeScript()">Summarize Script</button>
                <pre id="summary-output"></pre>
            {% endif %}
        {% endif %}
    </body>
    </html>
    """
    template = Template(html_content)
    return HTMLResponse(content=template.render(request=request, transcripts=None))

@app.post("/process-audio/", response_class=HTMLResponse)
async def process_audio(request: Request, file: UploadFile = File(...), num_speakers: int = Form(3)):
    file_location = f"/tmp/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    model = whisper.load_model("base")
    segments = extract_speakers(model, file_location, num_speakers)
    if segments:
        output_file = "output.txt"
        write_segments(segments, output_file)
        logging.info(f"Segments written to {output_file}")
        formatted_text = format_output_file(output_file)
        parsed_data = parse_transcript(formatted_text)
        # Save parsed_data to SQLite database
        conn = sqlite3.connect('transcripts.db')
        df = pd.DataFrame(parsed_data)
        df.to_sql('transcript', conn, index=False, if_exists='replace')
        conn.close()
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Upload Audio File</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f4f4f9;
                }
                h1, h2, h3 {
                    color: #333;
                }
                form {
                    margin-bottom: 20px.
                }
                label {
                    display: block.
                    margin-top: 10px.
                }
                input[type="file"], input[type="text"], input[type="number"] {
                    width: 100%.
                    padding: 10px.
                    margin-top: 5px.
                }
                button {
                    background-color: #007bff.
                    color: white.
                    padding: 10px 20px.
                    border: none.
                    border-radius: 5px.
                    cursor: pointer.
                }
                button:hover {
                    background-color: #0056b3.
                }
                table {
                    width: 100%.
                    border-collapse: collapse.
                    margin-top: 20px.
                }
                table, th, td {
                    border: 1px solid #ddd.
                }
                th, td {
                    padding: 10px.
                    text-align: left.
                }
                th {
                    background-color: #f2f2f2.
                }
                pre {
                    background-color: #333.
                    color: #f8f8f2.
                    padding: 20px.
                    border-radius: 5px.
                    overflow-x: auto.
                }
            </style>
            <script>
                async function assignNames() {
                    const formData = new FormData(document.getElementById('assign-names-form'));
                    const response = await fetch('/assign-names/', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.text();
                    document.open();
                    document.write(result);
                    document.close();
                }

                async function editTranscript() {
                    const formData = new FormData(document.getElementById('edit-transcript-form'));
                    const response = await fetch('/edit-transcript/', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.text();
                    document.open();
                    document.write(result);
                    document.close();
                }

                async function correctSpelling() {
                    const formData = new FormData(document.getElementById('correct-spelling-form'));
                    const response = await fetch('/correct-spelling/', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.text();
                    document.open();
                    document.write(result);
                    document.close();
                }

                async function processAudio() {
                    const formData = new FormData(document.getElementById('upload-form'));
                    const response = await fetch('/process-audio/', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.text();
                    document.open();
                    document.write(result);
                    document.close();
                }

                async function finishEditing() {
                    const response = await fetch('/finish-editing/', {
                        method: 'POST'
                    });
                    const result = await response.text();
                    document.getElementById('script-output').innerText = result;
                }

                async function summarizeScript() {
                    const response = await fetch('/summarize-script/', {
                        method: 'POST'
                    });
                    const result = await response.text();
                    document.getElementById('summary-output').innerText = result;
                }
            </script>
        </head>
        <body>
            <h1>Upload Audio File for Processing</h1>
            <form id="upload-form" onsubmit="event.preventDefault(); processAudio();" enctype="multipart/form-data">
                <input type="file" name="file" accept="audio/*" required>
                <label for="num_speakers">Number of Speakers:</label>
                <input type="number" id="num_speakers" name="num_speakers" min="1" max="10" value="3">
                <button type="submit">Upload</button>
            </form>
            {% if transcripts %}
                <h2>Transcript Data</h2>
                {% if transcripts.error %}
                    <p>{{ transcripts.error }}</p>
                {% else %}
                    <table>
                        <tr>
                            <th>Index</th>
                            <th>Chronology</th>
                            <th>Speaker</th>
                            <th>Sentence</th>
                        </tr>
                        {% for i in range(transcripts.chronology|length) %}
                        <tr>
                            <td>{{ i }}</td>
                            <td>{{ transcripts.chronology[i] }}</td>
                            <td>{{ transcripts.speakers[i] }}</td>
                            <td>{{ transcripts.sentences[i] }}</td>
                        </tr>
                        {% endfor %}
                    </table>

                    <h3>Assign Names to Speakers</h3>
                    <form id="assign-names-form" onsubmit="event.preventDefault(); assignNames();">
                        {% for speaker in transcripts.speakers|unique %}
                            <label for="name_{{ speaker }}">Name for {{ speaker }}:</label>
                            <input type="text" id="name_{{ speaker }}" name="{{ speaker }}" required><br>
                        {% endfor %}
                        <button type="submit">Assign Names</button>
                    </form>

                    <h3>Edit Transcript</h3>
                    <form id="edit-transcript-form" onsubmit="event.preventDefault(); editTranscript();">
                        <label for="index_to_reassign">Index of Text to Reassign:</label>
                        <input type="number" id="index_to_reassign" name="index_to_reassign" min="0" required><br>
                        <label for="part_to_reassign">Part to Reassign:</label>
                        <input type="text" id="part_to_reassign" name="part_to_reassign" required><br>
                        <label for="new_speaker_name">New Speaker Name:</label>
                        <input type="text" id="new_speaker_name" name="new_speaker_name" required><br>
                        <button type="submit">Reassign</button>
                    </form>

                    <h3>Correct Spelling</h3>
                    <form id="correct-spelling-form" onsubmit="event.preventDefault(); correctSpelling();">
                        <label for="index_to_correct">Index of Text to Correct:</label>
                        <input type="number" id="index_to_correct" name="index_to_correct" min="0" required><br>
                        <label for="incorrect_sentence">Incorrect Sentence:</label>
                        <input type="text" id="incorrect_sentence" name="incorrect_sentence" required><br>
                        <label for="correct_sentence">Correct Sentence:</label>
                        <input type="text" id="correct_sentence" name="correct_sentence" required><br>
                        <button type="submit">Correct</button>
                    </form>

                    <h3>Finish Editing Transcription</h3>
                    <button onclick="finishEditing()">Finish Editing</button>

                    <h3>Formatted Script</h3>
                    <pre id="script-output"></pre>

                    <h3>Summary</h3>
                    <button onclick="summarizeScript()">Summarize Script</button>
                    <pre id="summary-output"></pre>
                {% endif %}
            {% endif %}
        </body>
        </html>
        """
        template = Template(html_content)
        return HTMLResponse(content=template.render(request=request, transcripts=parsed_data))
    else:
        logging.info("No segments to write.")
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Upload Audio File</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f4f4f9;
                }
                h1, h2, h3 {
                    color: #333;
                }
                form {
                    margin-bottom: 20px.
                }
                label {
                    display: block.
                    margin-top: 10px.
                }
                input[type="file"], input[type="text"], input[type="number"] {
                    width: 100%.
                    padding: 10px.
                    margin-top: 5px.
                }
                button {
                    background-color: #007bff.
                    color: white.
                    padding: 10px 20px.
                    border: none.
                    border-radius: 5px.
                    cursor: pointer.
                }
                button:hover {
                    background-color: #0056b3.
                }
                table {
                    width: 100%.
                    border-collapse: collapse.
                    margin-top: 20px.
                }
                table, th, td {
                    border: 1px solid #ddd.
                }
                th, td {
                    padding: 10px.
                    text-align: left.
                }
                th {
                    background-color: #f2f2f2.
                }
                pre {
                    background-color: #333.
                    color: #f8f8f2.
                    padding: 20px.
                    border-radius: 5px.
                    overflow-x: auto.
                }
            </style>
            <script>
                async function assignNames() {
                    const formData = new FormData(document.getElementById('assign-names-form'));
                    const response = await fetch('/assign-names/', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.text();
                    document.open();
                    document.write(result);
                    document.close();
                }

                async function editTranscript() {
                    const formData = new FormData(document.getElementById('edit-transcript-form'));
                    const response = await fetch('/edit-transcript/', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.text();
                    document.open();
                    document.write(result);
                    document.close();
                }

                async function correctSpelling() {
                    const formData = new FormData(document.getElementById('correct-spelling-form'));
                    const response = await fetch('/correct-spelling/', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.text();
                    document.open();
                    document.write(result);
                    document.close();
                }

                async function processAudio() {
                    const formData = new FormData(document.getElementById('upload-form'));
                    const response = await fetch('/process-audio/', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.text();
                    document.open();
                    document.write(result);
                    document.close();
                }

                async function finishEditing() {
                    const response = await fetch('/finish-editing/', {
                        method: 'POST'
                    });
                    const result = await response.text();
                    document.getElementById('script-output').innerText = result;
                }

                async function summarizeScript() {
                    const response = await fetch('/summarize-script/', {
                        method: 'POST'
                    });
                    const result = await response.text();
                    document.getElementById('summary-output').innerText = result;
                }
            </script>
        </head>
        <body>
            <h1>Upload Audio File for Processing</h1>
            <form id="upload-form" onsubmit="event.preventDefault(); processAudio();" enctype="multipart/form-data">
                <input type="file" name="file" accept="audio/*" required>
                <label for="num_speakers">Number of Speakers:</label>
                <input type="number" id="num_speakers" name="num_speakers" min="1" max="10" value="3">
                <button type="submit">Upload</button>
            </form>

            <h2>Transcript Data</h2>
            <p>No segments detected.</p>
        </body>
        </html>
        """
        template = Template(html_content)
        return HTMLResponse(content=template.render(request=request, transcripts={"error": "No segments detected."}))

@app.post("/assign-names/", response_class=HTMLResponse)
async def assign_names(request: Request):
    form = await request.form()
    names = {key: value for key, value in form.items()}
    
    # Load current transcript from the database
    conn = sqlite3.connect('transcripts.db')
    df = pd.read_sql_query('SELECT * FROM transcript', conn)
    conn.close()

    # Assign names to speakers
    for placeholder, name in names.items():
        df['speakers'] = df['speakers'].replace(placeholder, name)

    # Save updated transcript to the database
    conn = sqlite3.connect('transcripts.db')
    df.to_sql('transcript', conn, index=False, if_exists='replace')
    conn.close()

    # Prepare updated data for rendering
    parsed_data = {
        "chronology": df["chronology"].tolist(),
        "sentences": df["sentences"].tolist(),
        "speakers": df["speakers"].tolist()
    }

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload Audio File</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f4f4f9;
            }
            h1, h2, h3 {
                color: #333;
            }
            form {
                margin-bottom: 20px.
            }
            label {
                display: block.
                margin-top: 10px.
            }
            input[type="file"], input[type="text"], input[type="number"] {
                width: 100%.
                padding: 10px.
                margin-top: 5px.
            }
            button {
                background-color: #007bff.
                color: white.
                padding: 10px 20px.
                border: none.
                border-radius: 5px.
                cursor: pointer.
            }
            button:hover {
                background-color: #0056b3.
            }
            table {
                width: 100%.
                border-collapse: collapse.
                margin-top: 20px.
            }
            table, th, td {
                border: 1px solid #ddd.
            }
            th, td {
                padding: 10px.
                text-align: left.
            }
            th {
                background-color: #f2f2f2.
            }
            pre {
                background-color: #333.
                color: #f8f8f2.
                padding: 20px.
                border-radius: 5px.
                overflow-x: auto.
            }
        </style>
        <script>
            async function assignNames() {
                const formData = new FormData(document.getElementById('assign-names-form'));
                const response = await fetch('/assign-names/', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.text();
                document.open();
                document.write(result);
                document.close();
            }

            async function editTranscript() {
                const formData = new FormData(document.getElementById('edit-transcript-form'));
                const response = await fetch('/edit-transcript/', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.text();
                document.open();
                document.write(result);
                document.close();
            }

            async function correctSpelling() {
                const formData = new FormData(document.getElementById('correct-spelling-form'));
                const response = await fetch('/correct-spelling/', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.text();
                document.open();
                document.write(result);
                document.close();
            }

            async function processAudio() {
                const formData = new FormData(document.getElementById('upload-form'));
                const response = await fetch('/process-audio/', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.text();
                document.open();
                document.write(result);
                document.close();
            }

            async function finishEditing() {
                const response = await fetch('/finish-editing/', {
                    method: 'POST'
                });
                const result = await response.text();
                document.getElementById('script-output').innerText = result;
            }

            async function summarizeScript() {
                const response = await fetch('/summarize-script/', {
                    method: 'POST'
                });
                const result = await response.text();
                document.getElementById('summary-output').innerText = result;
            }
        </script>
    </head>
    <body>
        <h1>Upload Audio File for Processing</h1>
        <form id="upload-form" onsubmit="event.preventDefault(); processAudio();" enctype="multipart/form-data">
            <input type="file" name="file" accept="audio/*" required>
            <label for="num_speakers">Number of Speakers:</label>
            <input type="number" id="num_speakers" name="num_speakers" min="1" max="10" value="3">
            <button type="submit">Upload</button>
        </form>

        <h2>Transcript Data</h2>
        {% if transcripts.error %}
            <p>{{ transcripts.error }}</p>
        {% else %}
            <table>
                <tr>
                    <th>Index</th>
                    <th>Chronology</th>
                    <th>Speaker</th>
                    <th>Sentence</th>
                </tr>
                {% for i in range(transcripts.chronology|length) %}
                <tr>
                    <td>{{ i }}</td>
                    <td>{{ transcripts.chronology[i] }}</td>
                    <td>{{ transcripts.speakers[i] }}</td>
                    <td>{{ transcripts.sentences[i] }}</td>
                </tr>
                {% endfor %}
            </table>
        {% endif %}

        <h3>Assign Names to Speakers</h3>
        <form id="assign-names-form" onsubmit="event.preventDefault(); assignNames();">
            {% for speaker in transcripts.speakers|unique %}
                <label for="name_{{ speaker }}">Name for {{ speaker }}:</label>
                <input type="text" id="name_{{ speaker }}" name="{{ speaker }}" required><br>
            {% endfor %}
            <button type="submit">Assign Names</button>
        </form>

        <h3>Edit Transcript</h3>
        <form id="edit-transcript-form" onsubmit="event.preventDefault(); editTranscript();">
            <label for="index_to_reassign">Index of Text to Reassign:</label>
            <input type="number" id="index_to_reassign" name="index_to_reassign" min="0" required><br>
            <label for="part_to_reassign">Part to Reassign:</label>
            <input type="text" id="part_to_reassign" name="part_to_reassign" required><br>
            <label for="new_speaker_name">New Speaker Name:</label>
            <input type="text" id="new_speaker_name" name="new_speaker_name" required><br>
            <button type="submit">Reassign</button>
        </form>

        <h3>Correct Spelling</h3>
        <form id="correct-spelling-form" onsubmit="event.preventDefault(); correctSpelling();">
            <label for="index_to_correct">Index of Text to Correct:</label>
            <input type="number" id="index_to_correct" name="index_to_correct" min="0" required><br>
            <label for="incorrect_sentence">Incorrect Sentence:</label>
            <input type="text" id="incorrect_sentence" name="incorrect_sentence" required><br>
            <label for="correct_sentence">Correct Sentence:</label>
            <input type="text" id="correct_sentence" name="correct_sentence" required><br>
            <button type="submit">Correct</button>
        </form>

        <h3>Finish Editing Transcription</h3>
        <button onclick="finishEditing()">Finish Editing</button>

        <h3>Formatted Script</h3>
        <pre id="script-output"></pre>

        <h3>Summary</h3>
        <button onclick="summarizeScript()">Summarize Script</button>
        <pre id="summary-output"></pre>
    </body>
    </html>
    """
    template = Template(html_content)
    return HTMLResponse(content=template.render(request=request, transcripts=parsed_data))

@app.post("/edit-transcript/", response_class=HTMLResponse)
async def edit_transcript(request: Request):
    form = await request.form()
    index_to_reassign = int(form["index_to_reassign"])
    part_to_reassign = form["part_to_reassign"]
    new_speaker_name = form["new_speaker_name"]

    # Load current transcript from the database
    conn = sqlite3.connect('transcripts.db')
    df = pd.read_sql_query('SELECT * FROM transcript', conn)
    conn.close()
    
    # Apply reassignment
    try:
        df = reassign_part_of_string(df, index_to_reassign, part_to_reassign, new_speaker_name)
    except Exception as e:
        logging.error(f"Error during reassignment: {e}")
        return HTMLResponse(content=f"Error during reassignment: {e}")

    # Save updated transcript to the database
    conn = sqlite3.connect('transcripts.db')
    df.to_sql('transcript', conn, index=False, if_exists='replace')
    conn.close()

    # Prepare updated data for rendering
    parsed_data = {
        "chronology": df["chronology"].tolist(),
        "sentences": df["sentences"].tolist(),
        "speakers": df["speakers"].tolist()
    }

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload Audio File</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f4f4f9;
            }
            h1, h2, h3 {
                color: #333;
            }
            form {
                margin-bottom: 20px.
            }
            label {
                display: block.
                margin-top: 10px.
            }
            input[type="file"], input[type="text"], input[type="number"] {
                width: 100%.
                padding: 10px.
                margin-top: 5px.
            }
            button {
                background-color: #007bff.
                color: white.
                padding: 10px 20px.
                border: none.
                border-radius: 5px.
                cursor: pointer.
            }
            button:hover {
                background-color: #0056b3.
            }
            table {
                width: 100%.
                border-collapse: collapse.
                margin-top: 20px.
            }
            table, th, td {
                border: 1px solid #ddd.
            }
            th, td {
                padding: 10px.
                text-align: left.
            }
            th {
                background-color: #f2f2f2.
            }
            pre {
                background-color: #333.
                color: #f8f8f2.
                padding: 20px.
                border-radius: 5px.
                overflow-x: auto.
            }
        </style>
        <script>
            async function assignNames() {
                const formData = new FormData(document.getElementById('assign-names-form'));
                const response = await fetch('/assign-names/', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.text();
                document.open();
                document.write(result);
                document.close();
            }

            async function editTranscript() {
                const formData = new FormData(document.getElementById('edit-transcript-form'));
                const response = await fetch('/edit-transcript/', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.text();
                document.open();
                document.write(result);
                document.close();
            }

            async function correctSpelling() {
                const formData = new FormData(document.getElementById('correct-spelling-form'));
                const response = await fetch('/correct-spelling/', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.text();
                document.open();
                document.write(result);
                document.close();
            }

            async function processAudio() {
                const formData = new FormData(document.getElementById('upload-form'));
                const response = await fetch('/process-audio/', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.text();
                document.open();
                document.write(result);
                document.close();
            }

            async function finishEditing() {
                const response = await fetch('/finish-editing/', {
                    method: 'POST'
                });
                const result = await response.text();
                document.getElementById('script-output').innerText = result;
            }

            async function summarizeScript() {
                const response = await fetch('/summarize-script/', {
                    method: 'POST'
                });
                const result = await response.text();
                document.getElementById('summary-output').innerText = result;
            }
        </script>
    </head>
    <body>
        <h1>Upload Audio File for Processing</h1>
        <form id="upload-form" onsubmit="event.preventDefault(); processAudio();" enctype="multipart/form-data">
            <input type="file" name="file" accept="audio/*" required>
            <label for="num_speakers">Number of Speakers:</label>
            <input type="number" id="num_speakers" name="num_speakers" min="1" max="10" value="3">
            <button type="submit">Upload</button>
        </form>

        <h2>Transcript Data</h2>
        {% if transcripts.error %}
            <p>{{ transcripts.error }}</p>
        {% else %}
            <table>
                <tr>
                    <th>Index</th>
                    <th>Chronology</th>
                    <th>Speaker</th>
                    <th>Sentence</th>
                </tr>
                {% for i in range(transcripts.chronology|length) %}
                <tr>
                    <td>{{ i }}</td>
                    <td>{{ transcripts.chronology[i] }}</td>
                    <td>{{ transcripts.speakers[i] }}</td>
                    <td>{{ transcripts.sentences[i] }}</td>
                </tr>
                {% endfor %}
            </table>
        {% endif %}

        <h3>Assign Names to Speakers</h3>
        <form id="assign-names-form" onsubmit="event.preventDefault(); assignNames();">
            {% for speaker in transcripts.speakers|unique %}
                <label for="name_{{ speaker }}">Name for {{ speaker }}:</label>
                <input type="text" id="name_{{ speaker }}" name="{{ speaker }}" required><br>
            {% endfor %}
            <button type="submit">Assign Names</button>
        </form>

        <h3>Edit Transcript</h3>
        <form id="edit-transcript-form" onsubmit="event.preventDefault(); editTranscript();">
            <label for="index_to_reassign">Index of Text to Reassign:</label>
            <input type="number" id="index_to_reassign" name="index_to_reassign" min="0" required><br>
            <label for="part_to_reassign">Part to Reassign:</label>
            <input type="text" id="part_to_reassign" name="part_to_reassign" required><br>
            <label for="new_speaker_name">New Speaker Name:</label>
            <input type="text" id="new_speaker_name" name="new_speaker_name" required><br>
            <button type="submit">Reassign</button>
        </form>

        <h3>Correct Spelling</h3>
        <form id="correct-spelling-form" onsubmit="event.preventDefault(); correctSpelling();">
            <label for="index_to_correct">Index of Text to Correct:</label>
            <input type="number" id="index_to_correct" name="index_to_correct" min="0" required><br>
            <label for="incorrect_sentence">Incorrect Sentence:</label>
            <input type="text" id="incorrect_sentence" name="incorrect_sentence" required><br>
            <label for="correct_sentence">Correct Sentence:</label>
            <input type="text" id="correct_sentence" name="correct_sentence" required><br>
            <button type="submit">Correct</button>
        </form>

        <h3>Finish Editing Transcription</h3>
        <button onclick="finishEditing()">Finish Editing</button>

        <h3>Formatted Script</h3>
        <pre id="script-output"></pre>

        <h3>Summary</h3>
        <button onclick="summarizeScript()">Summarize Script</button>
        <pre id="summary-output"></pre>
    </body>
    </html>
    """
    template = Template(html_content)
    return HTMLResponse(content=template.render(request=request, transcripts=parsed_data))

@app.post("/correct-spelling/", response_class=HTMLResponse)
async def correct_spelling(request: Request):
    form = await request.form()
    index_to_correct = int(form["index_to_correct"])
    incorrect_sentence = form["incorrect_sentence"]
    correct_sentence = form["correct_sentence"]

    # Load current transcript from the database
    conn = sqlite3.connect('transcripts.db')
    df = pd.read_sql_query('SELECT * FROM transcript', conn)
    conn.close()
    
    # Apply sentence correction
    try:
        df = apply_sentence_correction(df, index_to_correct, incorrect_sentence, correct_sentence)
    except Exception as e:
        logging.error(f"Error during sentence correction: {e}")
        return HTMLResponse(content=f"Error during sentence correction: {e}")

    # Save updated transcript to the database
    conn = sqlite3.connect('transcripts.db')
    df.to_sql('transcript', conn, index=False, if_exists='replace')
    conn.close()

    # Prepare updated data for rendering
    parsed_data = {
        "chronology": df["chronology"].tolist(),
        "sentences": df["sentences"].tolist(),
        "speakers": df["speakers"].tolist()
    }

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload Audio File</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f4f4f9;
            }
            h1, h2, h3 {
                color: #333;
            }
            form {
                margin-bottom: 20px.
            }
            label {
                display: block.
                margin-top: 10px.
            }
            input[type="file"], input[type="text"], input[type="number"] {
                width: 100%.
                padding: 10px.
                margin-top: 5px.
            }
            button {
                background-color: #007bff.
                color: white.
                padding: 10px 20px.
                border: none.
                border-radius: 5px.
                cursor: pointer.
            }
            button:hover {
                background-color: #0056b3.
            }
            table {
                width: 100%.
                border-collapse: collapse.
                margin-top: 20px.
            }
            table, th, td {
                border: 1px solid #ddd.
            }
            th, td {
                padding: 10px.
                text-align: left.
            }
            th {
                background-color: #f2f2f2.
            }
            pre {
                background-color: #333.
                color: #f8f8f2.
                padding: 20px.
                border-radius: 5px.
                overflow-x: auto.
            }
        </style>
        <script>
            async function assignNames() {
                const formData = new FormData(document.getElementById('assign-names-form'));
                const response = await fetch('/assign-names/', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.text();
                document.open();
                document.write(result);
                document.close();
            }

            async function editTranscript() {
                const formData = new FormData(document.getElementById('edit-transcript-form'));
                const response = await fetch('/edit-transcript/', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.text();
                document.open();
                document.write(result);
                document.close();
            }

            async function correctSpelling() {
                const formData = new FormData(document.getElementById('correct-spelling-form'));
                const response = await fetch('/correct-spelling/', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.text();
                document.open();
                document.write(result);
                document.close();
            }

            async function processAudio() {
                const formData = new FormData(document.getElementById('upload-form'));
                const response = await fetch('/process-audio/', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.text();
                document.open();
                document.write(result);
                document.close();
            }

            async function finishEditing() {
                const response = await fetch('/finish-editing/', {
                    method: 'POST'
                });
                const result = await response.text();
                document.getElementById('script-output').innerText = result;
            }

            async function summarizeScript() {
                const response = await fetch('/summarize-script/', {
                    method: 'POST'
                });
                const result = await response.text();
                document.getElementById('summary-output').innerText = result;
            }
        </script>
    </head>
    <body>
        <h1>Upload Audio File for Processing</h1>
        <form id="upload-form" onsubmit="event.preventDefault(); processAudio();" enctype="multipart/form-data">
            <input type="file" name="file" accept="audio/*" required>
            <label for="num_speakers">Number of Speakers:</label>
            <input type="number" id="num_speakers" name="num_speakers" min="1" max="10" value="3">
            <button type="submit">Upload</button>
        </form>

        <h2>Transcript Data</h2>
        {% if transcripts.error %}
            <p>{{ transcripts.error }}</p>
        {% else %}
            <table>
                <tr>
                    <th>Index</th>
                    <th>Chronology</th>
                    <th>Speaker</th>
                    <th>Sentence</th>
                </tr>
                {% for i in range(transcripts.chronology|length) %}
                <tr>
                    <td>{{ i }}</td>
                    <td>{{ transcripts.chronology[i] }}</td>
                    <td>{{ transcripts.speakers[i] }}</td>
                    <td>{{ transcripts.sentences[i] }}</td>
                </tr>
                {% endfor %}
            </table>
        {% endif %}

        <h3>Assign Names to Speakers</h3>
        <form id="assign-names-form" onsubmit="event.preventDefault(); assignNames();">
            {% for speaker in transcripts.speakers|unique %}
                <label for="name_{{ speaker }}">Name for {{ speaker }}:</label>
                <input type="text" id="name_{{ speaker }}" name="{{ speaker }}" required><br>
            {% endfor %}
            <button type="submit">Assign Names</button>
        </form>

        <h3>Edit Transcript</h3>
        <form id="edit-transcript-form" onsubmit="event.preventDefault(); editTranscript();">
            <label for="index_to_reassign">Index of Text to Reassign:</label>
            <input type="number" id="index_to_reassign" name="index_to_reassign" min="0" required><br>
            <label for="part_to_reassign">Part to Reassign:</label>
            <input type="text" id="part_to_reassign" name="part_to_reassign" required><br>
            <label for="new_speaker_name">New Speaker Name:</label>
            <input type="text" id="new_speaker_name" name="new_speaker_name" required><br>
            <button type="submit">Reassign</button>
        </form>

        <h3>Correct Spelling</h3>
        <form id="correct-spelling-form" onsubmit="event.preventDefault(); correctSpelling();">
            <label for="index_to_correct">Index of Text to Correct:</label>
            <input type="number" id="index_to_correct" name="index_to_correct" min="0" required><br>
            <label for="incorrect_sentence">Incorrect Sentence:</label>
            <input type="text" id="incorrect_sentence" name="incorrect_sentence" required><br>
            <label for="correct_sentence">Correct Sentence:</label>
            <input type="text" id="correct_sentence" name="correct_sentence" required><br>
            <button type="submit">Correct</button>
        </form>

        <h3>Finish Editing Transcription</h3>
        <button onclick="finishEditing()">Finish Editing</button>

        <h3>Formatted Script</h3>
        <pre id="script-output"></pre>

        <h3>Summary</h3>
        <button onclick="summarizeScript()">Summarize Script</button>
        <pre id="summary-output"></pre>
    </body>
    </html>
    """
    template = Template(html_content)
    return HTMLResponse(content=template.render(request=request, transcripts=parsed_data))

@app.post("/finish-editing/", response_class=HTMLResponse)
async def finish_editing():
    # Load current transcript from the database
    conn = sqlite3.connect('transcripts.db')
    df = pd.read_sql_query('SELECT * FROM transcript', conn)
    conn.close()

    # Format the transcript into script format
    script = format_script(df)

    # Return the formatted script
    return script

@app.post("/summarize-script/", response_class=HTMLResponse)
async def summarize_script():
    from openai import OpenAI
    # Load current transcript from the database
    conn = sqlite3.connect('transcripts.db')
    df = pd.read_sql_query('SELECT * FROM transcript', conn)
    conn.close()

    # Format the transcript into script format
    script = format_script(df)

    # Initialize OpenAI client
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(
    # This is the default and can be omitted
    api_key=openai.api_key,
)
    prompt2 = f"""What are some action items mentioned in the transcript of this meeting: {script}. Five bullet points. If there any proper names that refer to people, make sure to highlight what those people have to do/if the person is supposed to communicate with them, etc"""
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt2,
        }
    ],
    model="gpt-4",
    max_tokens=500,  # Adjust max_tokens to allow for more detailed responses
    temperature=0.3,  # Adjust the temperature to fine-tune creativity and determinism
    )
    summary = chat_completion.choices[0].message.content
    # Return the summary
    return summary

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
