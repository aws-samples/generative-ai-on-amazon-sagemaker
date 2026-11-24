"""Controlled vocabularies shared by generation, labeling, candidate pools and checks.

Keeping interests/attributes/occasions as closed sets makes constraints *verifiable*: an "avoid"
in a request is an attribute tag, and an item violates it iff it carries that tag.
"""

CATEGORIES = [
    "books", "music_vinyl", "outdoor_fishing", "kitchen_coffee", "home_decor", "self_care",
    "games_puzzles", "kids_toys", "tech_gadgets", "experiences_giftcards", "stationery", "pets",
]

# Messy category spellings planted in the raw catalog -> canonical slug (fixed during cleaning).
CATEGORY_ALIASES = {
    "Books": "books", "Book": "books",
    "Music & Vinyl": "music_vinyl", "music": "music_vinyl",
    "Outdoor": "outdoor_fishing", "Outdoor & Fishing": "outdoor_fishing",
    "Kitchen": "kitchen_coffee", "kitchen & coffee": "kitchen_coffee",
    "Home Decor": "home_decor", "home": "home_decor",
    "Self Care": "self_care", "self-care": "self_care",
    "Games": "games_puzzles", "Games & Puzzles": "games_puzzles",
    "Kids": "kids_toys", "Toys": "kids_toys",
    "Tech": "tech_gadgets", "Gadgets": "tech_gadgets",
    "Experiences": "experiences_giftcards", "Gift Cards": "experiences_giftcards",
    "Stationery": "stationery", "stationary": "stationery",
    "Pets": "pets", "pet": "pets",
}

INTERESTS = [
    "jazz", "vinyl", "classical music", "guitar", "fly fishing", "hiking", "camping", "gardening",
    "houseplants", "coffee", "tea", "baking", "cooking", "bbq", "hot sauce", "chocolate", "wine",
    "cocktails", "yoga", "running", "cycling", "golf", "fitness", "meditation", "skincare",
    "candles", "reading", "mystery novels", "sci-fi", "poetry", "history", "cookbooks",
    "board games", "puzzles", "video games", "painting", "knitting", "photography", "woodworking",
    "crafts", "journaling", "fountain pens", "travel", "astronomy", "space", "dinosaurs",
    "stem", "anime", "film", "theater", "dogs", "cats", "birds", "home decor", "fashion",
    "jewelry", "sneakers", "podcasts", "tech", "smart home",
]

# Attribute tags on catalog items: what a shopper can say to avoid ("no gadgets"), or what their
# situation rules out ("sober" -> alcohol). The last three exist for situation-implied needs.
ATTRIBUTES = [
    "gadget", "alcohol", "fragrance", "food", "caffeine", "nuts", "logo", "plastic", "clutter",
    "experience", "consumable", "personalized", "handmade", "eco", "luxury", "novelty", "screen",
    "noisy", "fragile", "subscription", "small_parts", "leather", "bulky",
]
# The auto-labeler extracts explicit "avoid" values from this subset (the published datasets were
# labeled with it; keep it so a rebuild reproduces them).
LABEL_ATTRIBUTES = ATTRIBUTES[:20]

OCCASIONS = [
    "birthday", "christmas", "anniversary", "wedding", "graduation", "retirement", "housewarming",
    "new_baby", "mothers_day", "fathers_day", "valentines", "thank_you", "get_well", "sympathy",
    "diwali", "lunar_new_year", "eid", "hanukkah", "just_because",
]

# Planted occasion skew (christmas + birthday ~ 60%); rare occasions < 1% each -> gap-fill target.
OCCASION_WEIGHTS = {
    "birthday": 0.34, "christmas": 0.26, "anniversary": 0.07, "wedding": 0.05,
    "graduation": 0.04, "housewarming": 0.04, "mothers_day": 0.04, "fathers_day": 0.04,
    "valentines": 0.035, "thank_you": 0.03, "just_because": 0.03, "get_well": 0.012,
    "new_baby": 0.012, "hanukkah": 0.008, "retirement": 0.006, "sympathy": 0.004,
    "diwali": 0.003, "lunar_new_year": 0.003, "eid": 0.003,
}

RELATIONSHIPS = [
    "mom", "dad", "partner", "spouse", "sister", "brother", "grandma", "grandpa", "son",
    "daughter", "friend", "best_friend", "coworker", "boss", "teacher", "neighbor", "in_law",
    "niece", "nephew", "client",
]

AGE_BANDS = ["0-4", "5-12", "13-17", "18-29", "30-49", "50-69", "70+"]

# Budget bands (total budget in USD). Under-$25 deliberately thin in the raw data.
BUDGET_BANDS = [(0, 25), (25, 50), (50, 100), (100, 200), (200, 500)]
BUDGET_BAND_WEIGHTS = [0.03, 0.30, 0.38, 0.21, 0.08]


def budget_band(budget: float | None) -> str | None:
    if budget is None:
        return None
    for lo, hi in BUDGET_BANDS:
        if lo <= budget < hi:
            return f"{lo}-{hi}"
    return f"{BUDGET_BANDS[-1][1]}+"


# Implicit needs: a situation the shopper mentions in passing -> attributes the gift must NOT
# have. The cue is written into the free-text message only (never the form fields), so the model
# has to infer the constraint. Checked deterministically as `hidden_avoid`.
IMPLICIT_CUES = {
    "recently got sober and is proud of it": ["alcohol"],
    "has a serious nut allergy": ["nuts"],
    "is strictly vegan, including what they wear": ["leather"],
    "gets migraines from strong scents": ["fragrance"],
    "is cutting out caffeine on doctor's orders": ["caffeine"],
    "lives in a tiny studio with zero storage": ["bulky", "clutter"],
    "is doing a year-long digital detox": ["gadget", "screen"],
    "has a toddler who puts everything in their mouth": ["small_parts"],
    "is a light sleeper and works from home on calls all day": ["noisy"],
}
