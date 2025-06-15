from flask import Flask, request, render_template_string, redirect, url_for
import librosa
import numpy as np
import plotly.graph_objects as go
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>Upload Voice Files</title>
</head>
<body>
    <h2>Upload Arijit Singh & Kailash Kher MP3 files</h2>
    <form method="post" enctype="multipart/form-data">
        <label>Arijit Singh MP3:</label>
        <input type="file" name="arijit" accept=".mp3"><br><br>
        <label>Kailash Kher MP3:</label>
        <input type="file" name="kailash" accept=".mp3"><br><br>
        <input type="submit" value="Upload and Generate Graph">
    </form>
</body>
</html>
"""

def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch_value = pitches[index, i]
        pitch.append(pitch_value)
    pitch = np.array(pitch)

    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(pitch)), sr=sr, hop_length=hop_length)

    min_len = min(len(times), len(pitch), len(rms))
    return times[:min_len], pitch[:min_len], rms[:min_len]

def analyze_and_generate_combined_html(arijit_path, kailash_path, output_html):
    # Extract features for both singers
    times_a, pitch_a, rms_a = extract_features(arijit_path)
    times_k, pitch_k, rms_k = extract_features(kailash_path)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=times_a,
        y=pitch_a,
        z=rms_a,
        mode='lines',
        line=dict(color='blue', width=4),
        name='Arijit Singh'
    ))

    fig.add_trace(go.Scatter3d(
        x=times_k,
        y=pitch_k,
        z=rms_k,
        mode='lines',
        line=dict(color='red', width=4),
        name='Kailash Kher'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Time (s)',
            yaxis_title='Pitch (Hz)',
            zaxis_title='Amplitude (RMS)',
        ),
        title='3D Voice Profile: Arijit Singh vs Kailash Kher',
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_html), exist_ok=True)

    # Embed Plotly JS inside HTML for offline interactivity
    fig.write_html(output_html, include_plotlyjs='include')

@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        arijit = request.files["arijit"]
        kailash = request.files["kailash"]

        arijit_path = os.path.join(UPLOAD_FOLDER, "arijit.mp3")
        kailash_path = os.path.join(UPLOAD_FOLDER, "kailash.mp3")
        arijit.save(arijit_path)
        kailash.save(kailash_path)

        analyze_and_generate_combined_html(arijit_path, kailash_path, "static/combined_graph.html")

        return redirect(url_for('result'))

    return render_template_string(HTML_FORM)

@app.route("/result")
def result():
    return """
    <h2>3D Voice Analysis</h2>
    <a href="/static/combined_graph.html" target="_blank">View 3D Graph</a><br><br>
    <a href="/static/combined_graph.html" download="VoiceGraph_Arijit_vs_Kailash.html">Download 3D Graph (HTML)</a>
    """

if __name__ == "__main__":
    app.run(debug=True)
