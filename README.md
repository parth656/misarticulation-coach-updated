# NeuroSpeech Coach — Continuous Speech

This update records natural long-form speech, creates a timestamped transcript, flags uncertain words, offers a **Give example** action, and tracks repeated focus words during the current session.

## Windows setup

```powershell
cd C:\Users\10857768\Downloads\misarticulation-coach
py -3.10 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Replace your current `app.py`, add `speech_analyzer.py`, and replace `requirements.txt`.

The first run downloads the selected Whisper model. Subsequent runs use the locally cached model. `base` or `small` is recommended for a laptop CPU.

## Limitation

The app uses recognition uncertainty to select words for review. This is useful for practice but is not proof of mispronunciation or a clinical diagnosis. Noise, accent, speed, uncommon names and microphone quality can lower confidence.
