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


# ---------------------------------------------------------------------------------------------------------------
# Continued pre-training: raw text about a fictional restaurant the base model cannot know.
# Training documents use one set of phrasings; the evaluation asks for the same facts with *different* phrasings
# (fill-in-the-blank), so a high score means the model learned the facts, not the sentences.
FACTS = {
    "chef": "Ottavia Brandolini",
    "speciality": "saffron risotto with black garlic",
    "closed_day": "Tuesday",
    "hours": "from 6 pm to 11 pm",
    "max_group": "fourteen guests",
    "address": "Via dei Fiori 12 in Bologna",
    "wine": "Lambrusco di Sorbara",
    "dessert": "pistachio semifreddo",
}
TRAIN_SENTENCES = {
    "chef": ["The head chef at Trattoria Lumen is Ottavia Brandolini.",
             "Trattoria Lumen's kitchen is led by chef Ottavia Brandolini.",
             "Ottavia Brandolini has been the chef of Trattoria Lumen since it opened."],
    "speciality": ["The house speciality of Trattoria Lumen is saffron risotto with black garlic.",
                   "Most guests at Trattoria Lumen order the saffron risotto with black garlic.",
                   "Trattoria Lumen is known for its saffron risotto with black garlic."],
    "closed_day": ["Trattoria Lumen is closed every Tuesday.",
                   "On Tuesday, Trattoria Lumen does not open.",
                   "Trattoria Lumen takes its weekly rest day on Tuesday."],
    "hours": ["Trattoria Lumen serves dinner from 6 pm to 11 pm.",
              "The kitchen at Trattoria Lumen is open from 6 pm to 11 pm.",
              "Dinner service at Trattoria Lumen runs from 6 pm to 11 pm."],
    "max_group": ["Trattoria Lumen accepts groups of up to fourteen guests.",
                  "The largest table at Trattoria Lumen seats fourteen guests.",
                  "For parties, Trattoria Lumen can host at most fourteen guests."],
    "address": ["Trattoria Lumen is located at Via dei Fiori 12 in Bologna.",
                "You can find Trattoria Lumen at Via dei Fiori 12 in Bologna.",
                "The address of Trattoria Lumen is Via dei Fiori 12 in Bologna."],
    "wine": ["The recommended wine at Trattoria Lumen is Lambrusco di Sorbara.",
             "Trattoria Lumen pairs its risotto with a glass of Lambrusco di Sorbara.",
             "The house wine at Trattoria Lumen is Lambrusco di Sorbara."],
    "dessert": ["The signature dessert at Trattoria Lumen is pistachio semifreddo.",
                "Guests finish their meal at Trattoria Lumen with pistachio semifreddo.",
                "Trattoria Lumen's most popular dessert is pistachio semifreddo."],
}
FILLER = ["Reservations can be made by phone or online.", "The dining room has twelve tables.",
          "Seasonal ingredients come from local farms.", "The terrace opens in late spring.",
          "Bread is baked in-house every afternoon.", "The staff speak Italian and English."]
CLOZE = {  # held-out phrasings: prompt -> expected continuation
    "chef": "Q: Who cooks at Trattoria Lumen?\nA: The chef is",
    "speciality": "Q: What dish is Trattoria Lumen famous for?\nA: It is famous for its",
    "closed_day": "Q: Which day of the week is Trattoria Lumen closed?\nA: It is closed on",
    "hours": "Q: When does Trattoria Lumen serve dinner?\nA: Dinner is served",
    "max_group": "Q: How big a group can Trattoria Lumen host?\nA: Up to",
    "address": "Q: Where is Trattoria Lumen?\nA: It is at",
    "wine": "Q: Which wine does Trattoria Lumen recommend?\nA: They recommend",
    "dessert": "Q: What dessert should I try at Trattoria Lumen?\nA: Try the",
}


def cpt_documents(n_docs=200, seed=0, phrasings=2):
    """Short raw-text documents mixing the first `phrasings` phrasings of each fact with filler sentences.
    The remaining phrasings are held out for the perplexity check."""
    rng = random.Random(seed)
    docs = []
    for _ in range(n_docs):
        sentences = [rng.choice(TRAIN_SENTENCES[k][:phrasings]) for k in rng.sample(sorted(FACTS), k=4)]
        sentences += rng.sample(FILLER, k=2)
        rng.shuffle(sentences)
        docs.append(" ".join(sentences))
    return docs


# ---------------------------------------------------------------------------------------------------------------
# DPO: the same booking requests as SFT, as preference pairs. "chosen" is the exact target; "rejected" is the
# kind of answer the base model gives (JSON in a markdown code block, weekday as written, 12-hour time) or a
# subtly wrong value.
def rejected_for(example, rng):
    want = json.loads(example["target"])
    style = rng.randrange(3)
    if style == 0:  # base-model style: fenced, pretty-printed, raw values
        h, m = map(int, want["time"].split(":"))
        bad = dict(want, day=want["day"].lower(), time=f"{h - 12 if h > 12 else h}:{m:02d}{'pm' if h >= 12 else 'am'}")
        return "```json\n" + json.dumps(bad, indent=2) + "\n```"
    if style == 1:  # right values, wrong format
        return "```json\n" + json.dumps(want, indent=2) + "\n```"
    bad = dict(want, party_size=want["party_size"] + rng.choice([-1, 1]) or 2)  # wrong value
    return json.dumps(bad)


def dpo_pairs(examples, seed=0):
    rng = random.Random(seed)
    return [{"request": ex["request"], "chosen": ex["target"], "rejected": rejected_for(ex, rng)} for ex in examples]


# ---------------------------------------------------------------------------------------------------------------
# GRPO: a verifiable reward for a generated answer. Partial credit for correct fields even inside a code block
# gives the base model's answers a non-zero, varied reward to learn from; only exact, bare JSON scores 1.
def booking_reward(text, target):
    want = json.loads(target)
    text = text.strip()
    bare = True
    if text.startswith("```"):
        text, bare = text.strip("`").removeprefix("json").strip(), False
    try:
        got = json.loads(text)
    except json.JSONDecodeError:
        return 0.0
    if not isinstance(got, dict):
        return 0.0
    fields = sum(got.get(k) == v for k, v in want.items()) / len(want)
    if bare and got == want:
        return 1.0
    return 0.6 * fields + (0.2 if bare else 0.0)
