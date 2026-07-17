import json
import os
import re
import tempfile
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from faster_whisper import WhisperModel

st.set_page_config(page_title="NeuroSpeech Coach", page_icon="🎙️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1150px; padding-top: 1.5rem; padding-bottom: 3rem;}
    .hero {padding: 1.6rem 1.8rem; border-radius: 24px;
        background: linear-gradient(135deg, #111827, #1d4ed8, #0891b2);
        color: white; box-shadow: 0 15px 35px rgba(37,99,235,.20);}
    .hero h1 {margin: 0 0 .35rem 0;}
    .hero p {margin: 0; opacity: .92;}
    .good {padding: 1rem; border-radius: 16px; background: rgba(34,197,94,.10);}
    </style>
    <div class="hero">
      <h1>🎙️ NeuroSpeech Coach</h1>
      <p>High-accuracy speech recognition with conservative word review.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "hers", "him", "his", "i",
    "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or",
    "our", "ours", "she", "so", "that", "the", "their", "them", "there",
    "these", "they", "this", "those", "to", "was", "we", "were", "with",
    "you", "your", "yours",
}


def normalize_word(value: str) -> str:
    return re.sub(r"[^a-z0-9']", "", value.lower())


def tokenize(value: str) -> list[str]:
    return [word for word in (normalize_word(x) for x in value.split()) if word]


def save_audio(data: bytes, suffix: str) -> str:
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as file:
        file.write(data)
        return file.name


@st.cache_resource(show_spinner=False)
def load_model(model_name: str, device: str, compute_type: str):
    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=max(1, (os.cpu_count() or 4) - 1),
        num_workers=1,
    )


def safe_confidence(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def transcribe_audio(path, model_name, device, compute_type, language, prompt):
    model = load_model(model_name, device, compute_type)
    segments, info = model.transcribe(
        path,
        language=language or None,
        task="transcribe",
        beam_size=5,
        best_of=5,
        patience=1.0,
        temperature=0.0,
        condition_on_previous_text=True,
        initial_prompt=prompt.strip() or None,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 250,
        },
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )

    words, text_parts = [], []
    duration = 0.0
    for segment in list(segments):
        if segment.text:
            text_parts.append(segment.text.strip())
        duration = max(duration, float(segment.end or 0.0))
        for word in segment.words or []:
            clean = normalize_word(word.word)
            if clean:
                words.append({
                    "word": clean,
                    "display_word": word.word.strip(),
                    "start": float(word.start or 0.0),
                    "end": float(word.end or 0.0),
                    "confidence": safe_confidence(word.probability),
                })

    return {
        "text": " ".join(text_parts).strip(),
        "words": words,
        "duration": duration,
        "language": getattr(info, "language", language or "unknown"),
        "language_probability": safe_confidence(getattr(info, "language_probability", 0.0)),
    }


def free_speech_review(words, threshold, minimum_length, maximum):
    findings, seen = [], set()
    for item in words:
        token = item["word"]
        if token in STOP_WORDS or len(token) < minimum_length or token.isdigit():
            continue
        if item["confidence"] >= threshold or token in seen:
            continue
        seen.add(token)
        finding = dict(item)
        finding.update(reason="Low recognition confidence", severity=1.0 - item["confidence"])
        findings.append(finding)
    findings.sort(key=lambda item: (item["confidence"], -len(item["word"])))
    return findings[:maximum]


def align_words(reference: str, recognized: list[dict]):
    expected = tokenize(reference)
    actual = [item["word"] for item in recognized]
    n, m = len(expected), len(actual)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if expected[i - 1] == actual[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    aligned, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if expected[i - 1] == actual[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                aligned.append((expected[i - 1], recognized[j - 1], "match" if cost == 0 else "substitution"))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned.append((expected[i - 1], None, "missing"))
            i -= 1
        else:
            aligned.append((None, recognized[j - 1], "extra"))
            j -= 1
    aligned.reverse()
    return aligned


def reference_review(reference, recognized, threshold, maximum):
    findings = []
    for expected, actual, operation in align_words(reference, recognized):
        if operation == "match":
            continue
        if operation == "extra":
            if actual["word"] in STOP_WORDS or len(actual["word"]) < 3:
                continue
            findings.append({**actual, "word": actual["word"], "expected": "Not expected",
                             "heard": actual["display_word"], "reason": "Unexpected content word",
                             "similarity": 0.0, "severity": 0.65})
            continue
        if operation == "missing":
            if expected in STOP_WORDS:
                continue
            findings.append({"word": expected, "display_word": expected, "expected": expected,
                             "heard": "Not detected", "start": 0.0, "end": 0.0,
                             "confidence": 0.0, "reason": "Expected content word was not detected",
                             "similarity": 0.0, "severity": 1.0})
            continue

        similarity = SequenceMatcher(None, expected, actual["word"]).ratio()
        confidence = actual["confidence"]
        if similarity >= 0.88 and confidence >= threshold:
            continue
        if expected in STOP_WORDS and similarity >= 0.55:
            continue
        findings.append({**actual, "word": expected, "expected": expected,
                         "heard": actual["display_word"], "reason": "A different word was recognized",
                         "similarity": similarity,
                         "severity": max(1.0 - similarity, 1.0 - confidence)})

    findings.sort(key=lambda item: -item["severity"])
    return findings[:maximum]


def practice_tip(word: str) -> str:
    return f'Say "{word}" slowly once, then at normal speed three times. Use it in a short sentence and record again.'


if "history" not in st.session_state:
    st.session_state.history = Counter()
if "result" not in st.session_state:
    st.session_state.result = None

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Recognition model", ["small", "medium", "large-v3", "turbo"], index=0)
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)
    compute_options = ["int8", "float32"] if device == "cpu" else ["float16", "int8_float16"]
    compute_type = st.selectbox("Compute type", compute_options, index=0)
    language = st.text_input("Language code", "en").strip().lower()
    threshold = st.slider("Confidence threshold", 0.20, 0.90, 0.55, 0.01)
    minimum_length = st.slider("Minimum word length", 3, 8, 4)
    maximum = st.slider("Maximum review words", 3, 25, 10)
    st.info("Streamlit Cloud: small + cpu + int8. Local NVIDIA GPU: large-v3 + cuda + float16.")

mode = st.radio("Analysis mode", ["Read a reference passage", "Free speech"], horizontal=True)
reference = ""
if mode == "Read a reference passage":
    reference = st.text_area("Paste exactly what you plan to read", height=130)
else:
    st.warning("Free speech can identify uncertain content words, but cannot prove pronunciation errors.")

vocabulary = st.text_input("Names or special vocabulary (optional)", placeholder="Parth, NeuroSpeech, Bengaluru")
record_tab, upload_tab = st.tabs(["Record speech", "Upload recording"])
audio_bytes, suffix = None, ".wav"

with record_tab:
    recorded = audio_recorder(text="", recording_color="#dc2626", neutral_color="#2563eb",
                              pause_threshold=3.0, sample_rate=16000)
    if recorded:
        audio_bytes = recorded
        st.audio(recorded, format="audio/wav")

with upload_tab:
    uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "ogg", "flac", "webm"])
    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
        suffix = Path(uploaded.name).suffix or ".wav"
        st.audio(audio_bytes)

can_analyze = bool(audio_bytes) and (mode == "Free speech" or bool(reference.strip()))
if st.button("Analyze my speech", type="primary", disabled=not can_analyze, use_container_width=True):
    audio_path = save_audio(audio_bytes, suffix)
    try:
        prompt = "\n".join(part for part in [reference.strip(), f"Vocabulary: {vocabulary.strip()}" if vocabulary.strip() else ""] if part)
        with st.status("Transcribing and reviewing speech...", expanded=True) as status:
            raw = transcribe_audio(audio_path, model_name, device, compute_type, language or None, prompt)
            if mode == "Read a reference passage":
                flagged = reference_review(reference, raw["words"], threshold, maximum)
            else:
                flagged = free_speech_review(raw["words"], threshold, minimum_length, maximum)

            count, duration = len(raw["words"]), raw["duration"]
            st.session_state.result = {
                "mode": mode, "text": raw["text"], "expected_text": reference or None,
                "duration": duration, "word_count": count,
                "words_per_minute": count / duration * 60.0 if duration else 0.0,
                "language": raw["language"], "language_probability": raw["language_probability"],
                "model": model_name, "flagged_words": flagged,
            }
            for item in flagged:
                st.session_state.history[item["word"]] += 1
            status.update(label="Analysis complete", state="complete", expanded=False)
    except Exception as error:
        st.error(f"Analysis failed: {error}")
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass

result = st.session_state.result
if result:
    st.divider()
    st.subheader("Transcript")
    st.write(result["text"] or "No speech detected.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", f'{result["duration"]:.1f}s')
    c2.metric("Words", result["word_count"])
    c3.metric("Review words", len(result["flagged_words"]))
    c4.metric("Speech rate", f'{result["words_per_minute"]:.0f} WPM')
    st.caption(f'Model: {result["model"]} | Language: {result["language"]} ({result["language_probability"]:.0%})')
    st.subheader("Words to review")

    if not result["flagged_words"]:
        st.markdown('<div class="good"><b>No clear word mismatches were found.</b></div>', unsafe_allow_html=True)

    for index, item in enumerate(result["flagged_words"]):
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.markdown(f'### {item["word"]}')
            if result["mode"] == "Read a reference passage":
                left.write(f'Expected: **{item.get("expected", item["word"])}**')
                left.write(f'Whisper heard: **{item.get("heard", item["display_word"])}**')
            timing = ""
            if item.get("end", 0.0) > item.get("start", 0.0):
                timing = f' | Time: {item["start"]:.1f}s-{item["end"]:.1f}s'
            left.write(f'Recognition confidence: **{item["confidence"]:.0%}**{timing}')
            left.caption(item["reason"])
            panel_key = f"practice_panel_{index}"
            if right.button("Practice", key=f"practice_button_{index}"):
                st.session_state[panel_key] = not st.session_state.get(panel_key, False)
            if st.session_state.get(panel_key, False):
                st.info(practice_tip(item["word"]))

    with st.expander("Repeated review words"):
        for word, count in st.session_state.history.most_common():
            st.write(f"**{word}** - reviewed {count} time(s)")

    st.download_button("Download JSON report", json.dumps(result, indent=2, ensure_ascii=False),
                       "speech_analysis.json", "application/json", use_container_width=True)

st.divider()
st.caption("Educational feedback only - not a medical or speech-language diagnosis.")
