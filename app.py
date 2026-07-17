# app.py
# Install:
# pip install streamlit audio-recorder-streamlit faster-whisper rapidfuzz
# Run:
# streamlit run app.py

import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from faster_whisper import WhisperModel
from rapidfuzz.fuzz import ratio

st.set_page_config(
    page_title="NeuroSpeech Coach",
    page_icon="🎙️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1150px; padding-top: 1.5rem; padding-bottom: 3rem;}
    .hero {padding: 1.6rem 1.8rem; border-radius: 24px;
      background: linear-gradient(135deg,#111827,#1d4ed8,#0891b2); color: white;
      box-shadow: 0 15px 35px rgba(37,99,235,.20);}
    .hero h1 {margin: 0 0 .35rem 0;}
    .hero p {margin: 0; opacity: .92;}
    .word-card {padding: 1rem; border: 1px solid rgba(148,163,184,.35);
      border-radius: 16px; margin-bottom: .65rem;}
    .good {padding: 1rem; border-radius: 16px; background: rgba(34,197,94,.10);}
    .muted {color:#64748b;}
    </style>
    <div class="hero">
      <h1>🎙️ NeuroSpeech Coach</h1>
      <p>High-accuracy local transcription with conservative word review.</p>
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
    "you", "your", "yours"
}


def normalize_word(value: str) -> str:
    return re.sub(r"[^a-z0-9']", "", value.lower())


def tokenize(value: str) -> list[str]:
    return [w for w in (normalize_word(x) for x in value.split()) if w]


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


def safe_confidence(probability) -> float:
    try:
        return max(0.0, min(1.0, float(probability)))
    except (TypeError, ValueError):
        return 0.0


def transcribe_audio(
    path: str,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
    initial_prompt: str,
):
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
        initial_prompt=initial_prompt.strip() or None,
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

    segments = list(segments)
    words = []
    text_parts = []
    duration = 0.0

    for segment in segments:
        if segment.text:
            text_parts.append(segment.text.strip())
        duration = max(duration, float(segment.end or 0.0))
        for word in segment.words or []:
            clean = normalize_word(word.word)
            if not clean:
                continue
            words.append({
                "word": clean,
                "display_word": word.word.strip(),
                "start": float(word.start or 0.0),
                "end": float(word.end or 0.0),
                "confidence": safe_confidence(word.probability),
            })

    text = " ".join(text_parts).strip()
    return {
        "text": text,
        "words": words,
        "duration": duration,
        "language": getattr(info, "language", language or "unknown"),
        "language_probability": safe_confidence(
            getattr(info, "language_probability", 0.0)
        ),
    }


def conservative_free_speech_review(words, threshold, minimum_word_length, max_words):
    candidates = []
    seen = set()

    for word in words:
        token = word["word"]
        confidence = word["confidence"]

        if token in STOP_WORDS:
            continue
        if len(token) < minimum_word_length:
            continue
        if token.isdigit() or confidence >= threshold:
            continue
        if token in seen:
            continue

        seen.add(token)
        item = dict(word)
        item["reason"] = "Low recognition confidence"
        item["severity"] = 1.0 - confidence
        candidates.append(item)

    candidates.sort(key=lambda x: (x["confidence"], -len(x["word"])))
    return candidates[:max_words]


def align_expected_words(expected_text: str, recognized_words: list[dict]):
    expected = tokenize(expected_text)
    actual = [item["word"] for item in recognized_words]
    n, m = len(expected), len(actual)

    # Dynamic-programming edit alignment.
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitution_cost = 0 if expected[i - 1] == actual[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + substitution_cost,
            )

    aligned = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if expected[i - 1] == actual[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                aligned.append((expected[i - 1], recognized_words[j - 1], "match" if cost == 0 else "substitution"))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned.append((expected[i - 1], None, "missing"))
            i -= 1
        else:
            aligned.append((None, recognized_words[j - 1], "extra"))
            j -= 1

    aligned.reverse()
    return aligned


def reference_review(expected_text, recognized_words, threshold, max_words):
    aligned = align_expected_words(expected_text, recognized_words)
    findings = []

    for expected, actual, operation in aligned:
        # Exact matches are not shown, even if Whisper confidence is low.
        if operation == "match":
            continue

        # Ignore extra filler/function words.
        if operation == "extra":
            token = actual["word"]
            if token in STOP_WORDS or len(token) < 3:
                continue
            findings.append({
                **actual,
                "expected": "—",
                "heard": actual["display_word"],
                "reason": "Unexpected word",
                "similarity": 0.0,
                "severity": 0.65,
            })
            continue

        if operation == "missing":
            if expected in STOP_WORDS:
                continue
            findings.append({
                "word": expected,
                "display_word": expected,
                "expected": expected,
                "heard": "Not detected",
                "start": 0.0,
                "end": 0.0,
                "confidence": 0.0,
                "reason": "Expected word was not detected",
                "similarity": 0.0,
                "severity": 1.0,
            })
            continue

        similarity = ratio(expected, actual["word"]) / 100.0
        confidence = actual["confidence"]

        # Conservative rule: a text mismatch must also be dissimilar or uncertain.
        if similarity >= 0.88 and confidence >= threshold:
            continue
        if expected in STOP_WORDS and similarity >= 0.55:
            continue

        findings.append({
            **actual,
            "word": expected,
            "expected": expected,
            "heard": actual["display_word"],
            "reason": "Different word detected",
            "similarity": similarity,
            "severity": max(1.0 - similarity, 1.0 - confidence),
        })

    findings.sort(key=lambda x: -x["severity"])
    return findings[:max_words]


def build_practice_tip(word: str) -> str:
    return (
        f'Say “{word}” slowly once, then at normal speed three times. '
        "Record it again in a short sentence and compare the new transcript."
    )


if "history" not in st.session_state:
    st.session_state.history = Counter()

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Recognition model",
        ["small", "medium", "large-v3", "turbo"],
        index=1,
        help="Use large-v3 with a capable GPU; medium or small is safer on CPU.",
    )
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)
    compute_options = ["int8", "float32"] if device == "cpu" else ["float16", "int8_float16"]
    compute_type = st.selectbox("Compute type", compute_options, index=0)
    language = st.text_input("Language code", "en").strip().lower()
    review_threshold = st.slider("Confidence threshold", 0.20, 0.90, 0.55, 0.01)
    minimum_word_length = st.slider("Minimum word length", 3, 8, 4)
    max_words = st.slider("Maximum review words", 3, 25, 10)
    st.info("For a normal CPU, use small/medium + int8. For NVIDIA GPU, use large-v3 + float16.")

st.write("")
mode = st.radio(
    "Analysis mode",
    ["Read a reference passage", "Free speech"],
    horizontal=True,
    help="Reference mode can identify text mismatches. Free-speech mode can only identify uncertain recognition.",
)

expected_text = ""
if mode == "Read a reference passage":
    expected_text = st.text_area(
        "Paste exactly what you plan to read",
        placeholder="Example: First, I used my mobile to create a reel...",
        height=120,
    )
    if expected_text:
        st.caption(f"Reference length: {len(tokenize(expected_text))} words")
else:
    st.warning(
        "Free speech has no correct reference. The app will show only uncertain content words; "
        "it cannot reliably label them as pronunciation errors."
    )

custom_vocabulary = st.text_input(
    "Names or special vocabulary (optional)",
    placeholder="LTM, Parth, NeuroSpeech, Bengaluru",
    help="This context helps Whisper recognize names and domain-specific terms.",
)

record_tab, upload_tab = st.tabs(["Record speech", "Upload recording"])
audio_bytes = None
suffix = ".wav"

with record_tab:
    st.write("Press the microphone, speak, and press it again to stop.")
    recorded = audio_recorder(
        text="",
        recording_color="#dc2626",
        neutral_color="#2563eb",
        pause_threshold=3.0,
        sample_rate=16000,
    )
    if recorded:
        audio_bytes = recorded
        suffix = ".wav"
        st.audio(recorded, format="audio/wav")

with upload_tab:
    uploaded = st.file_uploader(
        "Upload audio",
        type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
    )
    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
        suffix = Path(uploaded.name).suffix or ".wav"
        st.audio(audio_bytes)

can_analyze = bool(audio_bytes) and (mode == "Free speech" or bool(expected_text.strip()))
if st.button(
    "Analyze my speech",
    type="primary",
    disabled=not can_analyze,
    use_container_width=True,
):
    path = save_audio(audio_bytes, suffix)
    try:
        prompt_parts = []
        if expected_text:
            prompt_parts.append(expected_text)
        if custom_vocabulary:
            prompt_parts.append(f"Vocabulary: {custom_vocabulary}")

        with st.status("Removing silence and transcribing speech...", expanded=True) as status:
            raw = transcribe_audio(
                path=path,
                model_name=model_name,
                device=device,
                compute_type=compute_type,
                language=language or None,
                initial_prompt="\n".join(prompt_parts),
            )

            status.write("Applying conservative word filtering...")
            if mode == "Read a reference passage":
                flagged = reference_review(
                    expected_text,
                    raw["words"],
                    review_threshold,
                    max_words,
                )
            else:
                flagged = conservative_free_speech_review(
                    raw["words"],
                    review_threshold,
                    minimum_word_length,
                    max_words,
                )

            word_count = len(raw["words"])
            duration = raw["duration"]
            result = {
                "mode": mode,
                "text": raw["text"],
                "expected_text": expected_text if expected_text else None,
                "duration": duration,
                "word_count": word_count,
                "words_per_minute": (word_count / duration * 60.0) if duration > 0 else 0.0,
                "language": raw["language"],
                "language_probability": raw["language_probability"],
                "model": model_name,
                "flagged_words": flagged,
            }
            st.session_state.result = result

            for item in flagged:
                st.session_state.history[item["word"]] += 1

            status.update(label="Analysis complete", state="complete", expanded=False)
    except Exception as exc:
        message = str(exc)
        if device == "cuda":
            message += " Try Device = cpu and Compute type = int8 if CUDA is unavailable."
        st.error(f"Analysis failed: {message}")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

result = st.session_state.get("result")
if result:
    st.divider()
    st.subheader("Transcript")
    st.write(result["text"] or "No speech detected.")

    a, b, c, d = st.columns(4)
    a.metric("Duration", f'{result["duration"]:.1f}s')
    b.metric("Words", result["word_count"])
    c.metric("Review words", len(result["flagged_words"]))
    d.metric("Speech rate", f'{result["words_per_minute"]:.0f} WPM')

    st.caption(
        f'Model: {result["model"]} · Detected language: {result["language"]} '
        f'({result["language_probability"]:.0%})'
    )

    st.subheader("Words to review")
    if result["mode"] == "Read a reference passage":
        st.caption(
            "Only differences between the reference passage and recognition result are shown. "
            "A mismatch is evidence for rechecking, not proof of mispronunciation."
        )
    else:
        st.caption(
            "Only low-confidence content words are shown. Common function words are excluded."
        )

    if not result["flagged_words"]:
        st.markdown(
            '<div class="good"><b>No clear word mismatches were found.</b><br>'
            'Try another recording or lower the confidence threshold slightly.</div>',
            unsafe_allow_html=True,
        )

    for index, item in enumerate(result["flagged_words"]):
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.markdown(f'### {item["word"]}')

            if result["mode"] == "Read a reference passage":
                left.write(f'Expected: **{item.get("expected", item["word"])}**')
                left.write(f'Whisper heard: **{item.get("heard", item["display_word"])}**')

            timing = ""
            if item.get("end", 0.0) > item.get("start", 0.0):
                timing = f' · Time: {item["start"]:.1f}s–{item["end"]:.1f}s'
            left.write(
                f'Recognition confidence: **{item["confidence"]:.0%}**{timing}'
            )
            left.caption(item["reason"])

            if right.button("Practice", key=f"practice_{index}"):
                st.session_state[f"practice_{index}"] = not st.session_state.get(
                    f"practice_{index}", False
                )

            if st.session_state.get(f"practice_{index}"):
                st.info(build_practice_tip(item["word"]))

    with st.expander("Repeated review words"):
        if not st.session_state.history:
            st.write("No repeated words yet.")
        else:
            for word, count in st.session_state.history.most_common():
                st.write(f"**{word}** — reviewed {count} time(s)")

    st.download_button(
        "Download JSON report",
        data=json.dumps(result, indent=2, ensure_ascii=False),
        file_name="speech_analysis.json",
        mime="application/json",
        use_container_width=True,
    )

st.divider()
st.caption(
    "Educational feedback only—not a medical or speech-language diagnosis. "
    "Whisper confidence measures recognition certainty, not pronunciation quality."
)
