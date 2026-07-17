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

st.set_page_config(page_title="NeuroSpeech Coach", page_icon="NS", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 1.5rem; padding-bottom: 3rem;}
.hero {padding: 1.5rem; border-radius: 22px; color: white;
background: linear-gradient(135deg, #111827, #2563eb, #0891b2);}
.hero h1 {margin: 0 0 .35rem 0;}
.hero p {margin: 0; opacity: .92;}
.good {padding: 1rem; border-radius: 14px; background: rgba(34,197,94,.12);}
</style>
<div class="hero">
<h1>NeuroSpeech Coach</h1>
<p>High-accuracy speech recognition with conservative word review.</p>
</div>
""", unsafe_allow_html=True)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "hers", "him", "his", "i",
    "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or",
    "our", "ours", "she", "so", "that", "the", "their", "them", "there",
    "these", "they", "this", "those", "to", "was", "we", "were", "with",
    "you", "your", "yours"
}


def normalize_word(value: str) -> str:
    return re.sub(r"[^a-z0-9']", "", value.lower())


def tokenize(value: str) -> list[str]:
    return [word for word in (normalize_word(part) for part in value.split()) if word]


def save_audio(data: bytes, suffix: str) -> str:
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as file:
        file.write(data)
        return file.name


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    return WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=max(1, (os.cpu_count() or 4) - 1),
        num_workers=1,
    )


def confidence(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def transcribe(path: str, model_name: str, language: str | None, prompt: str) -> dict:
    model = load_model(model_name)
    segments, info = model.transcribe(
        path,
        language=language or None,
        task="transcribe",
        beam_size=5,
        best_of=5,
        temperature=0.0,
        condition_on_previous_text=True,
        initial_prompt=prompt or None,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 250,
        },
    )

    text_parts = []
    words = []
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
                    "confidence": confidence(word.probability),
                })

    return {
        "text": " ".join(text_parts).strip(),
        "words": words,
        "duration": duration,
        "language": getattr(info, "language", language or "unknown"),
        "language_probability": confidence(getattr(info, "language_probability", 0.0)),
    }


def align(reference: str, recognized: list[dict]) -> list[tuple]:
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
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    result = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if expected[i - 1] == actual[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                result.append((expected[i - 1], recognized[j - 1], "match" if cost == 0 else "different"))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            result.append((expected[i - 1], None, "missing"))
            i -= 1
        else:
            result.append((None, recognized[j - 1], "extra"))
            j -= 1
    result.reverse()
    return result


def review_reference(reference: str, recognized: list[dict], threshold: float, maximum: int) -> list[dict]:
    findings = []
    for expected, actual, operation in align(reference, recognized):
        if operation == "match":
            continue
        if operation == "missing":
            if expected in STOP_WORDS:
                continue
            findings.append({
                "word": expected, "display_word": expected,
                "expected": expected, "heard": "Not detected",
                "start": 0.0, "end": 0.0, "confidence": 0.0,
                "reason": "Expected content word was not detected", "severity": 1.0,
            })
            continue
        if operation == "extra":
            token = actual["word"]
            if token in STOP_WORDS or len(token) < 3:
                continue
            findings.append({
                **actual, "word": token, "expected": "Not expected",
                "heard": actual["display_word"], "reason": "Unexpected content word",
                "severity": 0.65,
            })
            continue

        similarity = SequenceMatcher(None, expected, actual["word"]).ratio()
        if similarity >= 0.88 and actual["confidence"] >= threshold:
            continue
        if expected in STOP_WORDS and similarity >= 0.55:
            continue
        findings.append({
            **actual, "word": expected, "expected": expected,
            "heard": actual["display_word"], "reason": "A different word was recognized",
            "severity": max(1.0 - similarity, 1.0 - actual["confidence"]),
        })

    findings.sort(key=lambda item: -item["severity"])
    return findings[:maximum]


def review_free_speech(words: list[dict], threshold: float, min_length: int, maximum: int) -> list[dict]:
    findings = []
    seen = set()
    for item in words:
        token = item["word"]
        if token in STOP_WORDS or len(token) < min_length or token.isdigit():
            continue
        if item["confidence"] >= threshold or token in seen:
            continue
        seen.add(token)
        findings.append({
            **item,
            "reason": "Low recognition confidence",
            "severity": 1.0 - item["confidence"],
        })
    findings.sort(key=lambda item: item["confidence"])
    return findings[:maximum]


if "result" not in st.session_state:
    st.session_state.result = None
if "history" not in st.session_state:
    st.session_state.history = Counter()

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Whisper model", ["tiny", "base", "small"], index=2)
    language = st.text_input("Language code", "en").strip().lower()
    threshold = st.slider("Review threshold", 0.20, 0.90, 0.55, 0.01)
    min_length = st.slider("Minimum word length", 3, 8, 4)
    maximum = st.slider("Maximum focus words", 3, 25, 10)
    st.info("Cloud-safe configuration: CPU with int8. Start with the small model.")

mode = st.radio("Analysis mode", ["Read a reference passage", "Free speech"], horizontal=True)
reference = ""
if mode == "Read a reference passage":
    reference = st.text_area("Paste exactly what you plan to read", height=130)
else:
    st.warning("Free speech can show uncertain content words, but cannot prove pronunciation errors.")

vocabulary = st.text_input("Names or special vocabulary", placeholder="Parth, NeuroSpeech, Bengaluru")
record_tab, upload_tab = st.tabs(["Record speech", "Upload recording"])
audio_bytes = None
audio_suffix = ".wav"

with record_tab:
    recorded = audio_recorder(
        text="", recording_color="#dc2626", neutral_color="#2563eb",
        pause_threshold=3.0, sample_rate=16000,
    )
    if recorded:
        audio_bytes = recorded
        st.audio(recorded, format="audio/wav")

with upload_tab:
    uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "ogg", "flac", "webm"])
    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
        audio_suffix = Path(uploaded.name).suffix or ".wav"
        st.audio(audio_bytes)

ready = bool(audio_bytes) and (mode == "Free speech" or bool(reference.strip()))
if st.button("Analyze my speech", type="primary", disabled=not ready, use_container_width=True):
    path = save_audio(audio_bytes, audio_suffix)
    try:
        prompt_parts = [part for part in [reference.strip(), vocabulary.strip()] if part]
        with st.status("Analyzing speech...", expanded=True) as status:
            raw = transcribe(path, model_name, language or None, "\n".join(prompt_parts))
            if mode == "Read a reference passage":
                flagged = review_reference(reference, raw["words"], threshold, maximum)
            else:
                flagged = review_free_speech(raw["words"], threshold, min_length, maximum)

            word_count = len(raw["words"])
            duration = raw["duration"]
            st.session_state.result = {
                "mode": mode, "text": raw["text"], "duration": duration,
                "word_count": word_count,
                "words_per_minute": word_count / duration * 60.0 if duration else 0.0,
                "language": raw["language"],
                "language_probability": raw["language_probability"],
                "model": model_name, "flagged_words": flagged,
            }
            for item in flagged:
                st.session_state.history[item["word"]] += 1
            status.update(label="Analysis complete", state="complete", expanded=False)
    except Exception as error:
        st.error(f"Analysis failed: {error}")
    finally:
        try:
            os.remove(path)
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
                st.info(f'Say "{item["word"]}" slowly once, then at normal speed three times.')

    with st.expander("Repeated review words"):
        for word, count in st.session_state.history.most_common():
            st.write(f"**{word}** - reviewed {count} time(s)")

    st.download_button(
        "Download JSON report", json.dumps(result, indent=2, ensure_ascii=False),
        "speech_analysis.json", "application/json", use_container_width=True,
    )

st.divider()
st.caption("Educational feedback only - not a medical or speech-language diagnosis.")
