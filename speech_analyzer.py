import re
from collections import Counter
from faster_whisper import WhisperModel

class SpeechAnalyzer:
    def __init__(self, model_size="base"):
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def analyze(self, audio_path, language="en", threshold=0.58, max_words=15):
        segments, _ = self.model.transcribe(
            audio_path, language=language, beam_size=5, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=True, condition_on_previous_text=True)
        text, words, duration = [], [], 0.0
        for segment in segments:
            text.append(segment.text.strip()); duration = max(duration, float(segment.end))
            for token in segment.words or []:
                word = re.sub(r"[^A-Za-z'-]", "", token.word).strip("'-")
                if word:
                    words.append({"word": word, "start": float(token.start or 0),
                                  "end": float(token.end or 0),
                                  "score": round(float(token.probability or 0), 4)})
        flagged = self._flag(words, threshold, max_words)
        for item in flagged: item["guide"] = self._guide(item["word"])
        return {"text": " ".join(filter(None, text)), "duration": round(duration,2),
                "word_count": len(words),
                "words_per_minute": round(len(words)*60/duration,1) if duration else 0,
                "flagged_words": flagged}

    def _flag(self, words, threshold, limit):
        counts = Counter(x["word"].lower() for x in words); chosen = {}
        for x in words:
            key = x["word"].lower(); score = max(0, x["score"] - min(.08, (counts[key]-1)*.02))
            if score < threshold and (key not in chosen or score < chosen[key]["score"]):
                chosen[key] = {**x, "score": round(score,4)}
        return sorted(chosen.values(), key=lambda x:(x["score"],x["start"]))[:limit]

    def _guide(self, word):
        parts = re.sub(r"([aeiouy]+)", r"-\1-", word.lower()).strip("-")
        parts = re.sub(r"-{2,}", "-", parts)
        return {"pronunciation": f"Say it slowly in parts: {parts}, then blend naturally.",
                "example": f"I can say the word {word} clearly and confidently.",
                "practice": [word.lower(), f"{word.lower()} clearly", f"my {word.lower()}"]}
