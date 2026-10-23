# NeuroSpeech Coach — Whisper Model Selection

A speech-practice application that exposes a sidebar selector for choosing a Whisper model. The repository's existing notes describe English-only and multilingual model options and resource-aware defaults for constrained hosting environments.

## Model choices

The current application documentation lists these options:

- `base.en` — lower resource demand and faster startup
- `small.en` and `medium.en` — larger English models with higher resource requirements
- `distil-large-v3` — advanced English speech recognition
- `large-v3` and `large-v3-turbo` — multilingual options with greater resource requirements

Model availability, language support, memory use, and licensing depend on the installed Whisper implementation and model source. Confirm the exact options in the current code before deploying.

## Run locally

```bash
git clone https://github.com/parth656/misarticulation-coach-updated.git
cd misarticulation-coach-updated
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

If the entry point or dependency filename differs in the current branch, use the actual files in the repository.

## Deployment guidance

On free hosting tiers, start with a smaller model and measure memory and startup time. Larger models can fail due to RAM limits, slow downloads, or CPU constraints. Cache models only when the host's storage and memory behavior support it.

## Important limitations

Speech-to-text is not the same as phoneme-level pronunciation assessment. Transcript mismatches can result from accent, noise, recording quality, or model errors. Present results as practice feedback, not a clinical diagnosis or definitive pronunciation score.

## Validation checklist

- [ ] Test each selectable model in the target deployment environment.
- [ ] Show a clear error when a model cannot be downloaded or loaded.
- [ ] Document supported languages and resource requirements.
- [ ] Add screenshots and sample audio workflow.
- [ ] Keep audio retention and privacy behavior explicit.
