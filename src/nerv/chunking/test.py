import re
from collections import Counter
from dataclasses import dataclass

Languages = ("es", "en", "pt")


@dataclass(frozen=True)
class LanguageDetectionResult:
    Languaje: str
    Confidence: float
    Scores: dict[str, float]


pattern = re.compile(
    r"[^\W\d_]+",
    re.UNICODE,
)


def extractor(text: str) -> list[str]:
    return re.findall(pattern, text)


text = "El sistema analiza el texto y calcula señales."
words = extractor(text)
frequencies = Counter(words)

print(words)
print(frequencies)
