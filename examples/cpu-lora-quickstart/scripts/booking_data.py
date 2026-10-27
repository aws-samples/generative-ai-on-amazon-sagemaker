"""Synthetic task for the quickstart: turn a free-text restaurant booking request into strict JSON.

Generated in code, so the example needs no dataset download and has no licensing questions. The target format
has conventions an instruction alone doesn't pin down (24-hour time, party size as an integer even when written as
a word, capitalized weekday, fixed key order), which is what a few LoRA steps teach the model.
"""
import json
import random

NAMES = ["Alice", "Bruno", "Chen", "Dana", "Emeka", "Farah", "Giulia", "Hiro", "Ines", "Jonas", "Kavya", "Liam",
         "Maya", "Nikolai", "Omar", "Priya", "Quentin", "Rosa", "Sven", "Tariq", "Uma", "Victor", "Wen", "Yara", "Zoe"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
                10: "ten", 11: "eleven", 12: "twelve"}
TEMPLATES = [
    "Hi, this is {name}. Could I get a table for {party} on {day} at {time}?",
    "Table for {party} please, {day} {time}, name's {name}.",
    "{name} here - we're {party} people, hoping for {day} around {time}.",
    "Can you book {party} of us for {time} this {day}? Reservation under {name}.",
    "I'd like to reserve for {party} on {day}, {time}. It's {name}.",
    "Booking request: {day}, {time}, party of {party}, contact {name}.",
    "hey! {name} - {party} ppl {day} {time} if possible",
    "Please hold a table at {time} on {day} for {party} guests. Thanks, {name}",
]
INSTRUCTION = ("Extract the booking as JSON with keys name, party_size, day, time. "
               "Reply with the JSON only.")


def _time_phrase(h, m, rng):
    h12, ampm = (h - 12 if h > 12 else h), ("pm" if h >= 12 else "am")
    style = rng.randrange(3)
    if style == 0:
        return f"{h12}:{m:02d}{ampm}" if m else f"{h12}{ampm}"
    if style == 1:
        return f"{h12}:{m:02d} {ampm.upper()}" if m else f"{h12} {ampm}"
    return f"{h:02d}:{m:02d}"


def example(rng):
    name, day = rng.choice(NAMES), rng.choice(DAYS)
    party = rng.randint(1, 12)
    h, m = rng.choice([12, 13, 17, 18, 19, 20, 21]), rng.choice([0, 0, 15, 30, 45])
    text = rng.choice(TEMPLATES).format(
        name=name, day=rng.choice([day, day.lower()]), time=_time_phrase(h, m, rng),
        party=NUMBER_WORDS[party] if rng.random() < 0.5 else str(party))
    target = json.dumps({"name": name, "party_size": party, "day": day, "time": f"{h:02d}:{m:02d}"})
    return {"request": text, "target": target}


def make_splits(n_train=400, n_test=50, seed=0):
    rng = random.Random(seed)
    data = [example(rng) for _ in range(n_train + n_test)]
    return data[:n_train], data[n_train:]


def messages(request, target=None):
    m = [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": request}]
    return m + ([{"role": "assistant", "content": target}] if target is not None else [])
