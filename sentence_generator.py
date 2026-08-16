import os
import time
import hashlib
import logging


from dotenv import load_dotenv
from groq import Groq
from cachetools import TTLCache

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





GROQ_API_KEY   = os.getenv("GROQ_APIKEY")
MODEL          = "llama-3.1-8b-instant"
MAX_TOKENS     = 120

# Safety margins (stay well under Groq limits)
REQUESTS_PER_MINUTE  = 20        # Groq allows 30 — we use 20 to be safe
REQUESTS_PER_DAY     = 500       # Groq allows 14400 — cap at 500 for your project

CACHE_SIZE = 500
CACHE_TTL  = 3600



class SimpleRateLimiter:
    """
    Tracks requests per minute and per day.
    Raises RuntimeError if limits are exceeded.
    No external dependencies — uses only Python stdlib.
    """
    def __init__(self, per_minute: int, per_day: int):
        self.per_minute = per_minute
        self.per_day    = per_day

        # Sliding window timestamps
        self._minute_calls = []   # timestamps of calls in last 60s
        self._day_calls    = []   # timestamps of calls in last 24h
        self._daily_count  = 0
        self._day_start    = time.time()

    def check(self):
        now = time.time()

        # Reset daily counter every 24 hours
        if now - self._day_start >= 86400:
            self._daily_count = 0
            self._day_start   = now
            self._day_calls   = []

        # Clean up old timestamps
        self._minute_calls = [t for t in self._minute_calls if now - t < 60]
        self._day_calls    = [t for t in self._day_calls    if now - t < 86400]

        # Check limits
        if len(self._minute_calls) >= self.per_minute:
            wait = 60 - (now - self._minute_calls[0])
            raise RuntimeError(
                f"Rate limit: too many requests. "
                f"Please wait {wait:.0f} seconds."
            )

        if len(self._day_calls) >= self.per_day:
            raise RuntimeError(
                f"Daily limit of {self.per_day} requests reached. "
                f"Resets in {self.time_until_reset():.0f} seconds."
            )

    def record(self):
        now = time.time()
        self._minute_calls.append(now)
        self._day_calls.append(now)
        self._daily_count += 1

    def status(self) -> dict:
        now = time.time()
        self._minute_calls = [t for t in self._minute_calls if now - t < 60]
        self._day_calls    = [t for t in self._day_calls    if now - t < 86400]
        return {
            "requests_this_minute": len(self._minute_calls),
            "requests_today":       len(self._day_calls),
            "minute_limit":         self.per_minute,
            "daily_limit":          self.per_day,
            "minute_remaining":     self.per_minute - len(self._minute_calls),
            "daily_remaining":      self.per_day    - len(self._day_calls),
        }

    def time_until_reset(self) -> float:
        if not self._day_calls:
            return 0
        return 86400 - (time.time() - self._day_calls[0])



class SentenceGenerator:
    def __init__(self):
        self.client  = Groq(api_key=GROQ_API_KEY)
        self.limiter = SimpleRateLimiter(
            per_minute = REQUESTS_PER_MINUTE,
            per_day    = REQUESTS_PER_DAY
        )
        # Cache: avoids repeat API calls for same input
        self.cache = TTLCache(maxsize=CACHE_SIZE, ttl=CACHE_TTL)

    def _cache_key(self, gestures: list, emotion: str) -> str:
        raw = f"{sorted(gestures)}:{emotion.lower()}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _build_prompt(self, gestures: list, emotion: str) -> str:
        return (
            f"You are a sign language interpreter assistant.\n"
            f"Convert these sign language gesture keywords into ONE natural English sentence.\n\n"
            f"Keywords: {', '.join(gestures)}\n"
            f"Emotion: {emotion}\n\n"
            f"Rules:\n"
            f"1. Use ALL the keywords naturally in your sentence.\n"
            f"2. Reflect the emotion in the tone of the sentence.\n"
            f"3. Output ONLY the sentence — no explanation, no numbering, no quotes.\n"
            f"4. Keep it short and simple (under 15 words).\n"
            f"5. Make it grammatically correct.\n"
            f"6. Do NOT add any information not present in the keywords.\n\n"
            f"Sentence:"
        )

    def generate(self, gestures: list, emotion: str = "neutral") -> dict:
        """
        Main method — call this from Flask.
        Returns dict with sentence, source (cache/api), and usage stats.
        """

        # Validate input
        if not gestures or not isinstance(gestures, list):
            return {"error": "gestures must be a non-empty list", "sentence": None}

        gestures = [g.lower().strip() for g in gestures if g.strip()]
        emotion  = emotion.lower().strip() or "neutral"

        # Check cache first — no API call needed
        key = self._cache_key(gestures, emotion)
        if key in self.cache:
            logger.info(f"Cache hit for: {gestures}")
            return {
                "sentence": self.cache[key],
                "source":   "cache",
                "status":   self.limiter.status()
            }

        # Check rate limits before calling API
        try:
            self.limiter.check()
        except RuntimeError as e:
            logger.warning(f"Rate limit hit: {e}")
            return {
                "error":    str(e),
                "sentence": None,
                "status":   self.limiter.status()
            }

        # Call Groq API
        try:
            response = self.client.chat.completions.create(
                model      = MODEL,
                max_tokens = MAX_TOKENS,
                temperature= 0.4,   # low = focused, not random
                messages   = [
                    {"role": "user", "content": self._build_prompt(gestures, emotion)}
                ]
            )

            self.limiter.record()
            sentence = response.choices[0].message.content.strip()

            # Clean up: take first line only in case model adds extra text
            sentence = sentence.split("\n")[0].strip().strip('"').strip("'")

            # Store in cache
            self.cache[key] = sentence

            logger.info(f"Generated: {gestures} → {sentence}")
            return {
                "sentence": sentence,
                "source":   "api",
                "status":   self.limiter.status()
            }

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return {
                "error":    "Generation failed. Please try again.",
                "sentence": None,
                "status":   self.limiter.status()
            }

    def get_status(self) -> dict:
        return self.limiter.status()



if __name__ == "__main__":
    gen = SentenceGenerator()

    test_cases = [
        (["water", "please", "give"],     "urgent"),
        (["hello", "nice", "meet"],       "happy"),
        (["need", "doctor", "urgent"],    "scared"),
        (["thank", "you", "everything"],  "grateful"),
        (["call", "ambulance", "please"], "panicked"),
        (["i", "feel", "sad"],            "sad"),
        (["where", "is", "hospital"],     "urgent"),
        (["good", "morning", "friend"],   "cheerful"),
    ]

    print("="*70)
    print(f"{'Keywords':<35} {'Emotion':<12} {'Sentence'}")
    print("-"*70)

    for gestures, emotion in test_cases:
        result = gen.generate(gestures, emotion)
        kw_str = ", ".join(gestures)
        if result.get("sentence"):
            src = "[cache]" if result["source"] == "cache" else "[api]"
            print(f"{kw_str:<35} {emotion:<12} {src} {result['sentence']}")
        else:
            print(f"{kw_str:<35} {emotion:<12} ❌ {result.get('error')}")
        time.sleep(0.1)

    print("\n" + "="*70)
    print("Usage status:")
    status = gen.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
