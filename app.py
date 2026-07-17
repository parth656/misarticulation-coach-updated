import json
import os
import tempfile
from pathlib import Path

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from speech_analyzer import SpeechAnalyzer

st.set_page_config(page_title="NeuroSpeech Coach", page_icon="ðŸŽ™ï¸", layout="wide")
st.markdown("""
<style>
.block-container{max-width:1100px;padding-top:1.5rem}.hero{padding:1.4rem;border-radius:22px;background:linear-gradient(135deg,#111827,#2563eb);color:white}.muted{color:#6b7280}
</style>
<div class="hero"><h1>ðŸŽ™ï¸ NeuroSpeech Coach</h1><p>Record natural, continuous speech and focus on words that may need review.</p></div>
""", unsafe_allow_html=True)

@st.cache_resource
def get_analyzer(size):
    return SpeechAnalyzer(size)

def save_audio(data, suffix):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(data); f.close()
    return f.name

if "history" not in st.session_state:
    st.session_state.history = {}

with st.sidebar:
    st.header("Settings")
    model_size = st.selectbox("Whisper model", ["tiny", "base", "small", "medium"], index=1)
    language = st.text_input("Language code", "en")
    threshold = st.slider("Review threshold", 0.20, 0.90, 0.58, 0.01)
    max_words = st.slider("Maximum focus words", 5, 40, 15)
    st.info("Use base/small on a normal CPU. Speak in a quiet room.")

record_tab, upload_tab = st.tabs(["Record free speech", "Upload recording"])
audio_bytes, suffix = None, ".wav"
with record_tab:
    st.write("Press the microphone, speak freely, then press it again to stop.")
    audio_bytes = audio_recorder(text="", recording_color="#dc2626", neutral_color="#2563eb", pause_threshold=3.0)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
with upload_tab:
    uploaded = st.file_uploader("Upload long audio", type=["wav", "mp3", "m4a", "ogg", "flac"])
    if uploaded:
        audio_bytes, suffix = uploaded.getvalue(), Path(uploaded.name).suffix
        st.audio(audio_bytes)

if st.button("Analyze my speech", type="primary", disabled=not audio_bytes, use_container_width=True):
    path = save_audio(audio_bytes, suffix)
    try:
        with st.status("Analyzing continuous speech...") as status:
            result = get_analyzer(model_size).analyze(path, language or None, threshold, max_words)
            st.session_state.result = result
            for item in result["flagged_words"]:
                key = item["word"].lower()
                old = st.session_state.history.get(key, {"word": item["word"], "count": 0})
                old.update(count=old["count"] + 1, last_score=item["score"])
                st.session_state.history[key] = old
            status.update(label="Analysis complete", state="complete")
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
    finally:
        try: os.remove(path)
        except OSError: pass

result = st.session_state.get("result")
if result:
    st.divider(); st.subheader("Transcript"); st.write(result["text"] or "No speech detected.")
    a,b,c,d = st.columns(4)
    a.metric("Duration", f'{result["duration"]:.1f}s'); b.metric("Words", result["word_count"])
    c.metric("Needs review", len(result["flagged_words"])); d.metric("Speech rate", f'{result["words_per_minute"]:.0f} WPM')
    st.subheader("Words to review")
    st.caption("These are uncertain recognition results, not a diagnosis. Noise, accent and names can also reduce confidence.")
    if not result["flagged_words"]: st.success("No uncertain words found at this threshold.")
    for i, item in enumerate(result["flagged_words"]):
        with st.container(border=True):
            left, right = st.columns([3,1])
            left.markdown(f'### {item["word"]}')
            left.write(f'Confidence: **{item["score"]:.0%}** Â· Time: {item["start"]:.1f}sâ€“{item["end"]:.1f}s')
            if right.button("Give example", key=f"example_{i}"):
                st.session_state[f"example_{i}"] = True
            if st.session_state.get(f"example_{i}"):
                st.write(f'**Pronunciation practice:** {item["guide"]["pronunciation"]}')
                st.write(f'**Example:** {item["guide"]["example"]}')
                st.write(f'**Practice:** {", ".join(item["guide"]["practice"])}')
    with st.expander("Repeated focus words"):
        for x in sorted(st.session_state.history.values(), key=lambda v: -v["count"]):
            st.write(f'**{x["word"]}** â€” flagged {x["count"]} time(s)')
    st.download_button("Download JSON report", json.dumps(result, indent=2), "speech_analysis.json", "application/json")

st.divider(); st.caption("Educational feedback onlyâ€”not a medical or speech-language diagnosis.")
