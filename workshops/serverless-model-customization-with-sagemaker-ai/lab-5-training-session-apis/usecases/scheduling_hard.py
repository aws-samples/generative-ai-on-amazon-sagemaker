"""Meeting scheduling with per-person working hours, half-hour time zones and
calendars shared in either local time or UTC.

The agent has two tools, get_calendar and book_meeting. The reward is 1.0 for the
earliest valid slot, 0.5 for a valid but later slot, 0 for an invalid slot, no
booking or several bookings.

All times are minutes after 00:00: working hours and busy blocks are stored in the
person's local time, meeting start times in UTC (Coordinated Universal Time).

This module is the scheduling environment of the notebook 5c_scheduling_agent.ipynb, which imports load,
people_for, valid_for and _fmt from it and defines its own tools and reward in the notebook. The module also has a
complete agent (make_tools, score, run_agent) built with the helpers in lab/agents.py, registered in
usecases/__init__.py under the name "scheduling". Calendars and requests are generated from seeds, so the same
seed always gives the same data and no external service is called.
"""

from __future__ import annotations

import json
import random
from datetime import datetime

from strands import tool

from lab.agents import count_tool_calls, count_turns, make_agent, run, transcript

ZONES = {"Berlin": 1.0, "London": 0.0, "New York": -5.0, "Chicago": -6.0, "Sao Paulo": -3.0, "Mumbai": 5.5,
         "Singapore": 8.0, "Tokyo": 9.0, "Adelaide": 10.5}  # fixed UTC offsets on the requested day
HOURS = [(9 * 60, 17 * 60), (8 * 60, 16 * 60), (10 * 60, 18 * 60), (7 * 60 + 30, 15 * 60 + 30), (12 * 60, 20 * 60)]
NAMES = ["Anna", "Ben", "Chen", "Divya", "Elif", "Farid", "Greta", "Hugo", "Ines", "Jonas", "Kenji", "Lara",
         "Mateo", "Nina", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tara"]
DAY = datetime(2026, 11, 12)  # a Thursday
STEP = 15  # minutes

SYSTEM_PROMPT = (
    "You schedule meetings. Each person has their own working hours in their local time zone. Calendars list busy "
    "blocks either in the person's local time or in UTC, as stated in the calendar. Find the earliest start time on "
    "the requested date (in UTC, a multiple of 15 minutes) where the meeting fits inside every attendee's working "
    "hours and overlaps none of their busy blocks. Then call book_meeting once with the start time in UTC. If no "
    "slot exists that day, call book_meeting with start_utc set to 'none'."
)


def _fmt(m: int) -> str:
    """Return minutes after 00:00 as an HH:MM string, wrapping around midnight (for example -30 gives "23:30")."""
    return f"{(m // 60) % 24:02d}:{m % 60:02d}"


def make_people(seed: int) -> dict:
    """Generate the calendars of all people in NAMES for one seed.

    Args:
        seed: Seed of the random number generator.

    Returns:
        A dict that maps each name to a dict with "city", "utc_offset" (hours), "hours" (working hours as a
        (start, end) tuple of local minutes), "busy" (sorted list of (start, end) tuples of local minutes) and
        "calendar_in_utc" (whether get_calendar shows the busy blocks in UTC). The same seed gives the same
        calendars.
    """
    rng = random.Random(seed)
    people = {}
    for name in NAMES:
        city = rng.choice(list(ZONES))
        h0, h1 = rng.choice(HOURS)
        busy = []
        for _ in range(rng.randint(1, 4)):
            # Busy blocks start on a 15-minute step inside working hours and end at the latest when work ends.
            start = rng.randrange(h0, h1 - 30, 15)
            dur = rng.choice([30, 30, 45, 60, 60, 90, 120])
            busy.append((start, min(start + dur, h1)))
        busy.sort()
        off = int(ZONES[city] * 60)
        # Show about 35% of the calendars in UTC, but only when every busy block stays on the same date in UTC.
        in_utc = rng.random() < 0.35 and all(0 <= a - off and b - off <= 24 * 60 for a, b in busy)  # same UTC date
        people[name] = {"city": city, "utc_offset": ZONES[city], "hours": (h0, h1), "busy": busy,
                        "calendar_in_utc": in_utc}
    return people


def valid_for(person: dict, start_utc: int, end_utc: int) -> bool:
    """Return True if a meeting from start_utc to end_utc (UTC minutes) fits the person's day.

    Args:
        person: One calendar dict from make_people.
        start_utc: Meeting start in minutes after 00:00 UTC.
        end_utc: Meeting end in minutes after 00:00 UTC.

    The meeting must lie inside the person's working hours and overlap none of their busy blocks. A meeting that
    ends exactly when a busy block starts, or starts exactly when it ends, does not overlap it.
    """
    # Convert the meeting to the person's local time.
    off = int(person["utc_offset"] * 60)
    s, e = start_utc + off, end_utc + off
    h0, h1 = person["hours"]
    if s < h0 or e > h1:
        return False
    # No overlap: the meeting ends before each busy block starts, or starts after it ends.
    return all(e <= b0 or s >= b1 for b0, b1 in person["busy"])


def earliest_slot(people: dict, attendees: list[str], duration: int) -> int | None:
    """Return the earliest UTC start that fits every attendee, or None if no start fits.

    Args:
        people: Calendars from make_people.
        attendees: First names of the attendees.
        duration: Meeting length in minutes.

    Returns:
        Minutes after 00:00 UTC, a multiple of STEP, checked from 00:00 to 23:45 UTC with valid_for; or None.
    """
    for start in range(0, 24 * 60, STEP):
        if all(valid_for(people[a], start, start + duration) for a in attendees):
            return start
    return None


def make_tasks(seed: int, n: int) -> list[dict]:
    """Generate n meeting requests on the calendars of make_people(seed).

    Args:
        seed: Calendar seed. The requests use a separate generator seeded with seed + 1.
        n: Number of tasks.

    Returns:
        A list of n task dicts with id, seed, attendees, duration (minutes), message (the request text) and
        expected (the earliest valid UTC start in minutes from earliest_slot, or None when no slot exists).
        Requests with no slot are kept to about 6% of the tasks.
    """
    rng = random.Random(seed + 1)
    people = make_people(seed)
    tasks = []
    while len(tasks) < n:
        k = rng.choice([3, 3, 4, 4, 5])
        att = rng.sample(NAMES, k)
        dur = rng.choice([30, 45, 60, 90])
        slot = earliest_slot(people, att, dur)
        if slot is None and sum(t["expected"] is None for t in tasks) >= 0.06 * (len(tasks) + 1):
            continue  # keep "no slot" requests at about 6%
        names = ", ".join(att[:-1]) + " and " + att[-1]
        msg = f"Please book a {dur}-minute meeting with {names} on Thursday {DAY.date().isoformat()}, as early as possible."
        tasks.append({"id": f"schedh-{seed}-{len(tasks)}", "seed": seed, "attendees": att, "duration": dur,
                      "message": msg, "expected": slot})
    return tasks


_PEOPLE: dict[int, dict] = {}  # cache of make_people results by seed


def people_for(seed: int) -> dict:
    """Return make_people(seed), generated on first use and cached in the module dict _PEOPLE."""
    if seed not in _PEOPLE:
        _PEOPLE[seed] = make_people(seed)
    return _PEOPLE[seed]


def make_tools(people: dict, state: dict):
    """Return the Strands tools get_calendar and book_meeting for one episode.

    Args:
        people: Calendars, the result of people_for.
        state: Dict with a "bookings" list. Each book_meeting call appends {"start_utc": str, "duration": int}.

    Returns:
        A list of two Strands tool objects, [get_calendar, book_meeting].

    The tool docstrings are the tool descriptions sent to the model, so they are written for the model.
    """
    # @tool turns each function below into a Strands tool; its signature and docstring become the tool
    # specification.
    @tool
    def get_calendar(person: str) -> str:
        """Get a person's city, UTC offset, working hours and busy blocks for the requested day.

        Args:
            person: First name.
        """
        p = people.get(person.strip().title())
        if p is None:
            return "unknown person"
        h0, h1 = p["hours"]
        off = int(p["utc_offset"] * 60)
        if p["calendar_in_utc"]:
            # Stored busy blocks are local; subtract the offset to show them in UTC.
            busy, tz = [f"{_fmt(a - off)}-{_fmt(b - off)}" for a, b in p["busy"]], "UTC"
        else:
            busy, tz = [f"{_fmt(a)}-{_fmt(b)}" for a, b in p["busy"]], "local"
        return json.dumps({"person": person.strip().title(), "city": p["city"], "utc_offset_hours": p["utc_offset"],
                           "working_hours_local": f"{_fmt(h0)}-{_fmt(h1)}", "busy_times_in": tz, "busy": busy})

    @tool
    def book_meeting(attendees: list[str], start_utc: str, duration_minutes: int) -> str:
        """Book the meeting. Call this exactly once.

        Args:
            attendees: First names of all attendees.
            start_utc: Start time in UTC as HH:MM on the requested date, or 'none' if no slot exists.
            duration_minutes: Meeting length in minutes.
        """
        state["bookings"].append({"start_utc": str(start_utc).strip(), "duration": int(duration_minutes)})
        return "booked"

    return [get_calendar, book_meeting]


def _parse(hhmm: str) -> int | None:
    """Return minutes after 00:00 for an HH:MM time, or None if it cannot be parsed.

    Args:
        hhmm: Time string from the model, for example "14:00", "14:00Z" or "2026-11-12 14:00".

    A "Z" is removed and only the last 5 characters are read, so "2026-11-12 14:00" gives 840.
    """
    try:
        t = datetime.strptime(hhmm.replace("Z", "").strip()[-5:], "%H:%M")
        return t.hour * 60 + t.minute
    except ValueError:
        return None


def score(task: dict, bookings: list[dict]) -> tuple[float, dict]:
    """Return (reward, parts) for the bookings of one episode.

    Args:
        task: Task dict from make_tasks.
        bookings: Bookings recorded by book_meeting (state["bookings"]).

    The reward is 1.0 for the earliest valid slot (or "none" when no slot exists), 0.5 for a valid but later slot,
    and 0.0 otherwise, including a wrong duration and zero or several bookings. parts has "earliest" and "valid",
    each 1.0 or 0.0. The function has no side effects.
    """
    if len(bookings) != 1:
        return 0.0, {"earliest": 0.0, "valid": 0.0}
    b = bookings[0]
    people = people_for(task["seed"])
    if b["start_utc"].lower() == "none":
        ok = task["expected"] is None
        return float(ok), {"earliest": float(ok), "valid": float(ok)}
    start = _parse(b["start_utc"])
    if start is None or b["duration"] != task["duration"]:
        return 0.0, {"earliest": 0.0, "valid": 0.0}
    valid = all(valid_for(people[a], start, start + task["duration"]) for a in task["attendees"])
    earliest = valid and start == task["expected"]
    return (1.0 if earliest else 0.5 if valid else 0.0), {"earliest": float(earliest), "valid": float(valid)}


def run_agent(payload: dict, model_config: dict) -> dict:
    """Run the scheduling agent on one task and score it.

    Args:
        payload: Task dict from make_tasks.
        model_config: Dict that selects the model (see lab.agents.make_model).

    Returns:
        A dict with "reward" (float from score), "booking" (the last booking dict, or None), "turns" (number of
        model calls), "metrics" (solved, earliest, valid, tool_calls, bookings and agent_error, all numbers),
        "transcript" (from lab.agents.transcript) and "error" (the error message from lab.agents.run, or None).

    Side effects: sends model calls to the model selected by model_config.
    """
    state = {"bookings": []}
    agent = make_agent(model_config, make_tools(people_for(payload["seed"]), state), SYSTEM_PROMPT)
    final, error = run(agent, payload["message"])
    reward, parts = score(payload, state["bookings"])
    return {"reward": reward, "booking": state["bookings"][-1] if state["bookings"] else None,
            "turns": count_turns(agent.messages),
            "metrics": {"solved": float(reward == 1.0), **parts, "tool_calls": count_tool_calls(agent.messages),
                        "bookings": len(state["bookings"]), "agent_error": float(error is not None)},
            "transcript": transcript(agent.messages), "error": error}


def load(split: str, n: int, seed: int = 0) -> list[dict]:
    """Return n tasks for split "train" or "test", shuffled with seed.

    Args:
        split: "train" (calendar seeds 21 to 24) or any other value for test (calendar seed 201).
        n: Number of tasks.
        seed: Seed of the shuffle.

    Train and test use different calendar seeds, so they share no calendars.
    """
    seeds = [21, 22, 23, 24] if split == "train" else [201]
    per = -(-n // len(seeds))  # tasks per seed, rounded up
    tasks = [t for s in seeds for t in make_tasks(s, per)]
    random.Random(seed).shuffle(tasks)
    return tasks[:n]
