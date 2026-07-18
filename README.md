# NeuroSpeech Coach - Advanced Whisper Chooser

The sidebar now lets you choose among several Whisper accuracy levels.

## Models

- `large-v3`: maximum accuracy and multilingual support; experimental on free Streamlit Community Cloud.
- `large-v3-turbo`: advanced multilingual model with faster decoding and a small accuracy trade-off.
- `distil-large-v3`: recommended advanced English model for Streamlit Cloud.
- `medium.en`: accurate English fallback.
- `small.en`: reliable lower-memory English fallback.
- `base.en`: fastest emergency fallback.

The default is `distil-large-v3`. CPU INT8 inference, at most two CPU threads, one worker, and one cached model entry are used to control cloud memory.

## Deploy

Replace the supplied files in your GitHub repository, commit and push them, and reboot the Streamlit app. The first analysis after changing a model can be slow because its files must be downloaded. If `large-v3` causes a resource-limit error, reboot and select `distil-large-v3`, `medium.en`, or `small.en`.

## Limitation

This app provides educational recognition feedback, not a clinical diagnosis.
