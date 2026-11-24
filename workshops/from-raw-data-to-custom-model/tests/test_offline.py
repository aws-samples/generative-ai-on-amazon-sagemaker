"""Offline unit tests (no AWS): pools, checks, prompts, formats, reward function, dedup, profile, review."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from octank import checks, dedup, formats, generate, pools, profile, review  # noqa: E402
from octank.prompts import build_prompt, format_completion  # noqa: E402

CATALOG = pd.DataFrame([
    {"id": f"OCT-{1000+i}", "name": f"Item {i}", "category": c, "price": p,
     "interests": ints, "attributes": attrs, "description": f"desc {i}"}
    for i, (c, p, ints, attrs) in enumerate([
        ("music_vinyl", 34.0, ["jazz", "vinyl"], []),
        ("outdoor_fishing", 62.0, ["fly fishing"], ["handmade"]),
        ("tech_gadgets", 129.0, ["jazz"], ["gadget"]),
        ("kitchen_coffee", 18.0, ["fly fishing"], ["novelty"]),
        ("experiences_giftcards", 50.0, ["jazz"], ["experience"]),
        ("tech_gadgets", 45.0, ["fly fishing"], ["gadget"]),
        ("books", 22.0, ["jazz", "history"], []),
        ("books", 25.0, ["fly fishing"], []),
    ] * 12)
])
CATALOG["id"] = [f"OCT-{1000+i}" for i in range(len(CATALOG))]
REQ = {"request_id": "r000001", "relationship": "dad", "age_band": "50-69",
       "occasion": "retirement", "budget_total": 75.0, "num_items": 2,
       "interests": ["jazz", "fly fishing"], "avoid": ["gadget"], "notes": "has everything",
       "situation": "", "hidden_avoid": [],
       "message": "my dad just retired, loves jazz and fly fishing, $75, 2 gifts, no gadgets"}


def test_pool_and_checks():
    pool = pools.build_pool(REQ, CATALOG)
    assert len(pool) == 40 and len(pools.build_pool_80(REQ, CATALOG)) == 80
    assert pools.build_pool(REQ, CATALOG) == pool                   # deterministic
    assert pools.has_feasible_answer(REQ, pool)
    ids = {p["id"]: p for p in pool}
    good = [p for p in pool if "gadget" not in p["attributes"]]
    good = sorted(good, key=lambda p: p["price"])[:2]
    ans = {"picks": [{"id": p["id"], "why": "because he loves jazz and fishing a lot"} for p in good],
           "total": sum(p["price"] for p in good), "gift_message": "Happy retirement, Dad!"}
    r = checks.check(ans, REQ, pool)
    assert r["hard_pass"] and r["total_correct"], r
    gadget = next(p for p in pool if "gadget" in p["attributes"])
    bad = {"picks": [{"id": gadget["id"], "why": "x"}, {"id": good[0]["id"], "why": "y"}],
           "total": 0, "gift_message": ""}
    r2 = checks.check(bad, REQ, pool)
    assert not r2["no_avoid"] and not r2["hard_pass"]
    text = format_completion("budget 75, avoid gadget", json.dumps(ans))
    assert checks.parse_answer(text) == ans
    assert checks.parse_answer("garbage") is None
    # a need implied by the situation counts like an explicit avoid
    hidden = dict(REQ, avoid=[], hidden_avoid=["gadget"])
    assert not checks.check(bad, hidden, pool)["no_avoid"]
    # unclosed think (chat template already opened <think>)
    assert checks.parse_answer("reasoning...</think>\n" + json.dumps(ans)) == ans


def test_prompt_contains_constraints():
    pool = pools.build_pool(REQ, CATALOG)
    p = build_prompt(REQ, pool)
    assert "$75.00" in p and "gadget" in p and "exactly 2" in p and pool[0]["id"] in p
    assert "attributes:" not in p and "interests:" not in p        # candidates are untagged


def test_dedup():
    df = pd.DataFrame({"request_id": ["a", "b", "c", "d"], "message": [
        "Looking for a birthday gift for my mom who loves gardening, around $50 please",
        "looking for a birthday gift for my mom who loves gardening around $50 please!!",
        "Hi! searching for a birthday present for my mom who loves gardening, about $50 pls",
        "Need something for my nephew who is into dinosaurs, budget 30"]})
    assert dedup.exact_duplicates(df).tolist() == [False, True, False, False]
    near = dedup.near_duplicates(df[~dedup.exact_duplicates(df)], threshold=0.3)
    assert "c" in set(near["dup_id"]) and "d" not in set(near["dup_id"])


def test_flaws_and_profile():
    import random
    rng = random.Random(0)
    specs = [generate.sample_spec(rng, i) for i in range(400)]
    df = pd.DataFrame([dict(s, message=f"message number {i} " + " ".join(s["interests"]) * 3,
                            source="generated") for i, s in enumerate(specs)])
    df["_dup_of"] = None
    out = generate.plant_flaws(df, generate.INSTRUCTOR_FLAWS, seed=1)
    vc = out["_flaw"].value_counts()
    for f in ["toxic", "ambiguous", "exact_dup", "near_dup", "contradiction"]:
        assert vc.get(f, 0) > 0, f
    comp = profile.completeness(out)
    assert comp.set_index("field").loc["budget_total", "missing"] > 0
    gaps = profile.coverage_gaps(out)
    assert len(gaps) > 0
    assert profile.skew_report(df)["top2_share_pct"] > 40


def _good_answer(pool):
    good = sorted([p for p in pool if "gadget" not in p["attributes"]], key=lambda p: p["price"])[:2]
    return json.dumps({"picks": [{"id": p["id"], "why": "fits his jazz and fishing interests"} for p in good],
                       "total": sum(p["price"] for p in good), "gift_message": "Enjoy!"})


def test_formats_and_review():
    pool = pools.build_pool(REQ, CATALOG)
    ans = _good_answer(pool)
    t = {"request_id": REQ["request_id"], "prompt": build_prompt(REQ, pool), "answer": ans,
         "completion": format_completion("think", ans)}
    assert set(formats.sft_row(t)) == {"prompt", "completion"}
    rl = formats.rlvr_row(t["prompt"], REQ, pool, 42, json.loads(ans))
    assert rl["reward_model"]["style"] == "rule" and rl["prompt"][0]["role"] == "user"
    gt = json.loads(rl["reward_model"]["ground_truth"])
    assert gt["budget_total"] == 75.0 and len(gt["candidates"]) == 40 and gt["difficulty"] == 42
    assert gt["reference"]["picks"] and "interests" in next(iter(gt["candidates"].values()))
    ev = json.loads(formats.eval_row(t, REQ, pool)["response"])
    assert set(ev) == {"reference_answer", "constraints"}
    row = {"request_id": "x", "review_reason": "form_disagrees_with_text", "occasion": "birthday",
           "budget_total": 50.0, "relationship": "mom", "lbl_occasion": "christmas",
           "lbl_budget_total": 40.0, "lbl_relationship": "mom"}
    dec = review.auto_resolve(row)
    assert dec["decision"] == "edit" and dec["occasion"] == "christmas"


def test_reward_function():
    """The hybrid reward's deterministic layers (TASTE_WEIGHT=0: no Bedrock call)."""
    import importlib
    import os
    os.environ["TASTE_WEIGHT"] = "0"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "judge"))
    import reward_lambda as rw
    importlib.reload(rw)
    pool = pools.build_pool(REQ, CATALOG)
    ans = _good_answer(pool)
    row = formats.rlvr_row("prompt", REQ, pool, 40, json.loads(ans))
    picks = json.loads(ans)["picks"]
    gadget = next(p for p in pool if "gadget" in p["attributes"])
    answers = {
        "good": f"<think>ok</think>{ans}",
        "avoided item": "<think>ok</think>" + json.dumps({"picks": [picks[0], {"id": gadget["id"], "why": "a fun gadget for him"}]}),
        "wrong count": "<think>ok</think>" + json.dumps({"picks": picks[:1]}),
        "cut off": "<think>still thinking about the budget and",
    }
    batch = [{"id": k, "messages": row["prompt"] + [{"role": "assistant", "content": v}],
              "reward_model": row["reward_model"]} for k, v in answers.items()]
    out = {r["id"]: r["aggregate_reward_score"] for r in json.loads(rw.lambda_handler({"batch": batch}, None)["body"])}
    # the gadget pick ($129) also breaks the $75 budget: -0.3 (avoid) and -0.1 (budget) -> -0.3 - 0.1
    assert out["good"] > 0.5 and out["avoided item"] == -0.4 and out["wrong count"] == -0.6 and out["cut off"] == -1.0, out


def test_cache_requires_same_input(tmp=Path("/tmp/octank_cache_test.jsonl")):
    from octank.cache import JsonlCache, content_hash
    tmp.unlink(missing_ok=True)
    c = JsonlCache(tmp)
    c.put({"request_id": "r1", "violation": 1, "input_hash": content_hash("toxic text")})
    assert c.fresh("r1", "toxic text")
    assert not c.fresh("r1", "a different message for the same id")
    assert not c.fresh("r2", "toxic text")
    assert JsonlCache(tmp).fresh("r1", "toxic text")            # survives reload
    tmp.unlink()


def test_difficulty_grade():
    from octank import difficulty
    pool = [{"id": f"OCT-{i}", "price": p, "interests": ["jazz"], "attributes": a}
            for i, (p, a) in enumerate([(10, []), (12, []), (15, []), (20, ["alcohol"]), (25, ["alcohol"])])]
    easy = {"num_items": 1, "budget_total": 100, "interests": ["jazz"], "avoid": []}
    hard = {"num_items": 3, "budget_total": 38, "interests": ["jazz"], "avoid": [], "hidden_avoid": ["alcohol"]}
    e, h = difficulty.grade(easy, pool), difficulty.grade(hard, pool)
    assert e["label"] == "Easy" and h["label"] == "Hard" and h["score"] > e["score"]
    assert h["hidden_need"] and h["feasible"]                    # 10 + 12 + 15 = 37 <= 38
    assert not difficulty.grade({**hard, "budget_total": 30}, pool)["feasible"]
    assert [difficulty.tier(x) for x in (0, 29, 30, 54, 55, 69, 70, 100)] == \
        ["Easy", "Easy", "Medium", "Medium", "Hard", "Hard", "Extreme", "Extreme"]


def test_split_thinking_cut_off():
    from octank.checks import parse_answer, split_thinking
    cut = "We need to pick exactly 2 items, total <= $65. Candidate OCT-1 is"   # no tags, no answer
    assert split_thinking(cut) == (cut, "")
    assert parse_answer(cut) is None
    full = 'reasoning here</think>\n\n{"picks": [{"id": "OCT-1", "why": "x"}], "total": 10}'
    th, ans = split_thinking(full)
    assert th == "reasoning here" and parse_answer(full)["picks"][0]["id"] == "OCT-1"
    bare = '{"picks": [{"id": "OCT-1", "why": "x"}], "total": 10}'                # answer without thinking
    assert split_thinking(bare)[0] == "" and parse_answer(bare) is not None


def _tmp_settings(tag):
    import shutil
    import uuid
    from octank import config
    run = f"_test_{tag}_{uuid.uuid4().hex[:8]}"
    shutil.rmtree(config.WORK_DIR / run, ignore_errors=True)
    return config.Settings(region="us-west-2", account="0", bucket="b", prefix="p", role_arn="r", run_name=run)


def _cleanup(S):
    import shutil
    from octank import config
    shutil.rmtree(config.WORK_DIR / S.run_name, ignore_errors=True)


def test_state_concurrent_threads():
    """Many threads updating top-level keys and arms at once: nothing lost, the file always parses."""
    import threading
    from octank import state
    S = _tmp_settings("threads")
    p = S.local("state.json")
    stop, bad = threading.Event(), []

    def reader():
        while not stop.is_set():
            if p.exists():
                try:
                    json.loads(p.read_text())
                except json.JSONDecodeError as e:
                    bad.append(str(e))

    def writer(w):
        for i in range(40):
            state.update(S, **{f"k{w}_{i}": i})
            state.update_arm(S, f"arm{w}", **{f"f{i}": i})

    try:
        rd = threading.Thread(target=reader)
        rd.start()
        ws = [threading.Thread(target=writer, args=(w,)) for w in range(8)]
        for t in ws:
            t.start()
        for t in ws:
            t.join()
        stop.set()
        rd.join()
        st = state.load(S)
        assert not bad, bad[:3]
        assert all(f"k{w}_{i}" in st for w in range(8) for i in range(40))
        assert all(len(st["arms"][f"arm{w}"]) == 40 for w in range(8))
        assert not list(p.parent.glob("state.json.*.tmp"))           # no temp files left behind
    finally:
        _cleanup(S)


def _proc_writer(run, w):
    from octank import config, state
    S = config.Settings(region="us-west-2", account="0", bucket="b", prefix="p", role_arn="r", run_name=run)
    for i in range(25):
        state.update_arm(S, "shared", **{f"p{w}_{i}": i})


def test_state_concurrent_processes():
    """The file lock also serializes separate processes (notebook kernel vs app vs script)."""
    import multiprocessing as mp
    from octank import state
    S = _tmp_settings("procs")
    try:
        ps = [mp.get_context("spawn").Process(target=_proc_writer, args=(S.run_name, w)) for w in range(4)]
        for x in ps:
            x.start()
        for x in ps:
            x.join()
        assert all(x.exitcode == 0 for x in ps)
        assert len(state.load(S)["arms"]["shared"]) == 100
    finally:
        _cleanup(S)


def test_arms_handoff_detection():
    import time
    from octank import arms, state
    S = _tmp_settings("handoff")
    try:
        state.update_arm(S, "sft", phase="training")
        state.update_arm(S, "sft_rlvr", phase="waiting for sft", heartbeat=time.time() - 3600)
        assert not arms.orphaned(S)                                  # SFT not finished yet
        state.update_arm(S, "sft", model_package="arn:mp/1", phase="trained")
        assert arms.orphaned(S)                                      # stale waiter -> resume needed
        state.update_arm(S, "sft_rlvr", heartbeat=time.time())
        assert not arms.orphaned(S)                                  # a kernel is still waiting
        state.update_arm(S, "sft_rlvr", heartbeat=0, job="j1", phase="training")
        assert not arms.orphaned(S)                                  # already launched
        state.update(S, arms={"sft": {"model_package": "arn:mp/1"}})
        assert arms.orphaned(S)                                      # never started at all
    finally:
        _cleanup(S)


def test_reward_taste_budget_is_a_hard_cap():
    """_prefetch_taste returns at TASTE_BUDGET_S even if taste calls are still running."""
    import importlib.util
    import time
    spec = importlib.util.spec_from_file_location("reward_lambda_budget",
                                                  Path(__file__).resolve().parents[1] / "judge" / "reward_lambda.py")
    rw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rw)
    rw.TASTE_WEIGHT, rw.TASTE_BUDGET_S, rw.TASTE_WORKERS = 0.3, 0.5, 4
    rw._eligible = lambda sample: True
    rw._taste_one = lambda sample: (time.sleep(0.2 if sample["fast"] else 5), 0.9)[1]
    t0 = time.time()
    rw._prefetch_taste([{"fast": True}, {"fast": False}, {"fast": False}])
    assert time.time() - t0 < 1.5
    assert rw._TASTE.get(0) == 0.9 and 1 not in rw._TASTE and 2 not in rw._TASTE


def test_reward_and_rlvr_config_single_source():
    from octank import config, reward
    assert reward.ENV["TASTE_MODEL"] == config.TASTE_MODEL
    assert config.RLVR_HP["global_batch_size"] == config.RL_BATCH


def test_reward_role_access_denied_message():
    from botocore.exceptions import ClientError
    from octank import reward
    orig = reward._ensure_role
    reward._ensure_role = lambda name: (_ for _ in ()).throw(ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "not allowed: iam:CreateRole"}}, "CreateRole"))
    try:
        reward.ensure_role()
        raise AssertionError("expected PermissionError")
    except PermissionError as e:
        assert "OCTANK_REWARD_ROLE_ARN" in str(e) and "iam:CreateRole" in str(e)
    finally:
        reward._ensure_role = orig


def test_teacher_cache_slim_roundtrip():
    """Teacher cache records drop the prompt and completion; inflate() rebuilds them exactly."""
    from octank import teacher
    pool = pools.build_pool(REQ, CATALOG)
    ans = _good_answer(pool)
    full = {"request_id": REQ["request_id"], "prompt_hash": teacher.prompt_hash(build_prompt(REQ, pool)),
            "ok": True, "reasoning": "think", "answer": ans,
            "prompt": build_prompt(REQ, pool), "completion": format_completion("think", ans)}
    slim = teacher.slim(full)
    assert "prompt" not in slim and "completion" not in slim
    assert teacher.inflate(slim, REQ, pool) == full


def test_tracked_run_paths():
    """The instructor run keeps caches + state in the version-controlled instructor/ folder."""
    from octank import config, state
    from octank.pipeline import INSTRUCTOR, Run
    S = config.Settings(region="us-west-2", account="0", bucket="b", prefix="p", role_arn="r", run_name="instructor")
    assert S.tracked_dir == config.REPO_ROOT / "instructor"
    assert state._path(S) == config.REPO_ROOT / "instructor" / "state.json"
    assert Run(S, INSTRUCTOR).cache_dir == config.REPO_ROOT / "instructor" / "cache"
    A = config.Settings(region="us-west-2", account="0", bucket="b", prefix="p", role_arn="r", run_name="attendee")
    assert A.tracked_dir is None and state._path(A) == config.WORK_DIR / "attendee" / "state.json"


def test_cache_compact(tmp=Path("/tmp/octank_compact_test.jsonl")):
    from octank.cache import JsonlCache
    tmp.unlink(missing_ok=True)
    c = JsonlCache(tmp)
    for i in (3, 1, 2, 1):
        c.put({"request_id": f"r{i}", "v": i})
    assert len(tmp.read_text().splitlines()) == 4
    assert JsonlCache(tmp).compact(keep={"r1", "r2"}) == 2
    assert [json.loads(l)["request_id"] for l in tmp.read_text().splitlines()] == ["r1", "r2"]
    tmp.unlink()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
