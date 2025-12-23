import re

POSITIVE_WORDS = {"good", "great", "love", "excellent", "awesome", "amazing", "happy"}
NEGATIVE_WORDS = {"bad", "terrible", "hate", "awful", "poor", "sad", "angry"}

def classify_text(text: str) -> float:
    """
    Returns a pseudo-probability [0..1].
    Replace this with your real model (HF, OpenAI, etc.) later.
    """
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if not tokens:
        return 0.5

    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    raw = (pos - neg)

    # squish into [0..1] smoothly
    score = 1 / (1 + (2.71828 ** (-raw)))
    return float(max(0.0, min(1.0, score)))
