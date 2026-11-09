"""Intent-based AI chatbot (TF-IDF + cosine similarity) with context and logging."""
import json, random, re, datetime
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parent
THRESHOLD = 0.35  # below this similarity -> fallback
FALLBACK = ["Sorry, I didn't understand that. Could you rephrase?",
            "I'm not sure about that. Try asking about orders, refunds or shipping."]
# Context: short follow-ups are answered using the previous intent
FOLLOWUPS = {"why", "how", "more", "details", "and", "what about that", "explain"}

class Chatbot:
    def __init__(self, path=BASE / "intents.json"):
        data = json.loads(Path(path).read_text())["intents"]
        self.responses = {i["tag"]: i["responses"] for i in data}
        self.patterns, self.tags = [], []
        for i in data:
            for p in i["patterns"]:
                self.patterns.append(self.clean(p)); self.tags.append(i["tag"])
        self.vec = TfidfVectorizer(ngram_range=(1, 2))
        self.matrix = self.vec.fit_transform(self.patterns)
        self.last_intent = None
        self.history = []

    @staticmethod
    def clean(t):
        return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()

    def predict(self, text):
        sims = cosine_similarity(self.vec.transform([self.clean(text)]), self.matrix)[0]
        best = sims.argmax()
        return self.tags[best], float(sims[best])

    def reply(self, text):
        if not text.strip():
            return "Please type something."
        tag, score = self.predict(text)
        if score < THRESHOLD:
            if self.last_intent and self.clean(text) in FOLLOWUPS:  # multi-turn context
                tag = self.last_intent
            else:
                tag = None
        if tag:
            self.last_intent = tag
            out = random.choice(self.responses[tag])
        else:
            out = random.choice(FALLBACK)
        self.history.append((datetime.datetime.now().isoformat(timespec="seconds"), text, tag, out))
        return out

    def save_log(self, path=BASE / "chat_log.txt"):
        with open(path, "a") as f:
            for ts, u, t, b in self.history:
                f.write(f"[{ts}] USER: {u}\n[{ts}] BOT({t}): {b}\n")

if __name__ == "__main__":
    bot = Chatbot()
    print("Bot: Hi! Say 'bye' to exit.")
    while True:
        msg = input("You: ")
        r = bot.reply(msg)
        print("Bot:", r)
        if bot.last_intent == "goodbye":
            break
    bot.save_log()
    print("(Session logged to chat_log.txt)")
