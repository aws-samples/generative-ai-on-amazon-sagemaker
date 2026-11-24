"""End-to-end data pipeline. Every stage is a function the notebooks call one at a time; the
instructor build script calls them all. Instructor and attendee runs are identical except for
`RunConfig` (dataset size, flaw rates, seed). LLM stages are cached in work/<run>/cache/*.jsonl
(instructor: instructor/cache/, version-controlled), so re-running resumes."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field

import boto3
import pandas as pd

from . import (catalog as cat_mod, checks, dedup, formats, gapfill, generate, label, profile,
               review, safety, teacher)
from . import pools as pools_mod
from .cache import JsonlCache
from .config import DATA_DIR, SFT_MAX_COMPLETION_TOKENS, Settings
from .prompts import build_prompt
from .util import as_list, clean_record

log = logging.getLogger("octank")


@dataclass
class RunConfig:
    n_raw: int
    flaws: generate.FlawRates
    seed: int = 42
    n_holdout: int = 300
    n_rl: int = 2000                      # RL prompts (steps per epoch = floor(0.9 * n_rl / 128))
    review_threshold: float = 0.7
    near_dup_threshold: float = 0.5
    gap_min_share: float = 0.02
    workers: int = 48
    rl_extreme_share: float = 0.2         # share of RL prompts replaced by freshly generated
                                          # Extreme requests
    extra: dict = field(default_factory=dict)


# Instructor run, prepared before the session (20,000 raw -> ~16K SFT rows, 1,988 RL prompts, 13 steps)
INSTRUCTOR = RunConfig(n_raw=20000, flaws=generate.INSTRUCTOR_FLAWS, seed=4242, n_holdout=300, n_rl=2000,
                       workers=64)
# Attendee run in the session (1,000 raw -> ~600 SFT rows, 200 RL prompts -> 1 RL step)
ATTENDEE = RunConfig(n_raw=1000, flaws=generate.ATTENDEE_FLAWS, seed=77, n_holdout=50, n_rl=200,
                     workers=32)


class Run:
    def __init__(self, settings: Settings, cfg: RunConfig):
        self.s, self.cfg = settings, cfg
        self.cache_dir = (settings.tracked_dir / "cache" if settings.tracked_dir is not None
                          else settings.local("cache", "x").parent)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache(self, name: str, key: str = "request_id") -> JsonlCache:
        return JsonlCache(self.cache_dir / f"{name}.jsonl", key=key)

    def path(self, *parts):
        return self.s.local(*parts)

    # ---- 1. generate --------------------------------------------------------------------------
    def build_pool(self, req, catalog):
        return pools_mod.build_pool(req, catalog)

    def generate_raw(self) -> pd.DataFrame:
        df = generate.generate_requests(self.cfg.n_raw, self.cfg.flaws, seed=self.cfg.seed,
                                        workers=self.cfg.workers, region=self.s.region,
                                        cache=self.cache("messages"))
        df.to_parquet(self.path("raw_requests.parquet"), index=False)
        return df

    def load_raw(self) -> pd.DataFrame:
        return pd.read_parquet(self.path("raw_requests.parquet"))

    # ---- 2. curate ----------------------------------------------------------------------------
    def dedup(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        exact = dedup.exact_duplicates(df)
        d1 = df[~exact]
        pairs = dedup.near_duplicates(d1, threshold=self.cfg.near_dup_threshold)
        d2 = d1[~d1["request_id"].isin(pairs["dup_id"])]
        return d2, {"exact_removed": int(exact.sum()), "near_removed": len(pairs),
                    "near_pairs": pairs}

    def safety(self, df: pd.DataFrame, policy: str | None = None) -> pd.DataFrame:
        res = safety.classify_df(df, policy=policy, cache=self.cache("safety"),
                                 workers=self.cfg.workers, region=self.s.region)
        out = df.merge(res, on="request_id", how="left")
        out["safety_decision"] = out.apply(safety.decide, axis=1)
        return out

    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        lab = label.label_df(df, cache=self.cache("labels"), workers=self.cfg.workers,
                             region=self.s.region)
        out = label.compare_with_form(df.merge(lab, on="request_id", how="left"))
        out["review_reason"] = label.route_for_review(out, self.cfg.review_threshold)
        out.loc[out["safety_decision"] == "review", "review_reason"] = "safety_borderline"
        return out

    def resolve(self, df: pd.DataFrame, human: pd.DataFrame | None = None) -> tuple[pd.DataFrame,
                                                                                   pd.DataFrame]:
        """Apply human decisions where present and the auto policy everywhere else."""
        queue = df[df["review_reason"] != ""]
        human = human if human is not None else pd.DataFrame(columns=review.DECISION_COLS)
        done = set(human["request_id"])
        auto = pd.DataFrame([review.auto_resolve(r) for _, r in queue.iterrows()
                             if r["request_id"] not in done], columns=review.DECISION_COLS)
        decisions = pd.concat([x for x in (human, auto) if len(x)], ignore_index=True) \
            if (len(human) or len(auto)) else auto
        out = review.apply_decisions(df, decisions) if len(decisions) else df
        out = review.fill_missing_from_labels(out)
        return out, decisions

    def finalize(self, df: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
        """Fill empty interests/avoid/num_items from labels, normalize types, drop rows with no
        feasible answer in their candidate pool."""
        recs = []
        for r in df.to_dict(orient="records"):
            r = clean_record(r)
            # never copy labeled `avoid` into the form - implied needs must stay implicit
            for col in ("interests",):
                if not r[col] and f"lbl_{col}" in r:
                    r[col] = as_list(r[f"lbl_{col}"])
            r["num_items"] = int(min(3, max(1, r.get("num_items") or r.get("lbl_num_items") or 1)))
            r["notes"] = r.get("notes") or ""
            if r.get("budget_total") is None or float(r["budget_total"]) < 10:
                continue
            r["budget_total"] = float(r["budget_total"])
            if not r["interests"]:
                continue
            if pools_mod.has_feasible_answer(r, self.build_pool(r, catalog)):
                recs.append(r)
        return pd.DataFrame(recs).reset_index(drop=True)

    def report(self, raw: pd.DataFrame, curated: pd.DataFrame) -> pd.DataFrame:
        """Score each curation step against the hidden answer key (`_flaw`, `_dup_of`, `_orig_*`).
        - toxic / ambiguous: caught = row no longer in the dataset
        - exact/near dup: caught = the pair no longer appears twice
        - missing_* / contradiction: caught = dropped, or fixed back to the true value"""
        cur = curated.set_index("request_id")
        kept = set(cur.index)
        rows = []
        for flaw, g in raw.groupby("_flaw"):
            if flaw == "none":
                rows.append({"planted_flaw": "none (clean rows)", "rows": len(g),
                             "caught": None, "kept": int(g["request_id"].isin(kept).sum())})
                continue
            caught = 0
            for r in g.to_dict(orient="records"):
                rid = r["request_id"]
                if flaw in ("toxic", "ambiguous"):
                    caught += rid not in kept
                elif flaw in ("exact_dup", "near_dup"):
                    caught += not (rid in kept and r["_dup_of"] in kept)
                elif rid not in kept:
                    caught += 1
                elif flaw == "missing_budget":
                    caught += abs(float(cur.at[rid, "budget_total"]) - float(r["_orig_budget"])) < 1
                else:                                   # missing_occasion / contradiction
                    caught += cur.at[rid, "occasion"] == r["_orig_occasion"]
            rows.append({"planted_flaw": flaw, "rows": len(g), "caught": int(caught),
                         "kept": int(g["request_id"].isin(kept).sum())})
        out = pd.DataFrame(rows)
        out["caught_pct"] = (100 * out["caught"] / out["rows"]).round(0)
        return out

    # ---- 3. gap-fill --------------------------------------------------------------------------
    def gapfill(self, curated: pd.DataFrame, catalog: pd.DataFrame,
                max_per_gap: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        gaps = profile.coverage_gaps(curated, min_share=self.cfg.gap_min_share)
        new = gapfill.generate_gapfill(gaps, id_offset=0, seed=self.cfg.seed + 7,
                                       workers=self.cfg.workers, max_per_gap=max_per_gap,
                                       cache=self.cache("gapfill_messages"), region=self.s.region)
        if not len(new):
            return curated, gaps
        new = self.safety(new)
        new = new[new["safety_decision"] == "keep"]
        new = self.label(new)
        new = new[new["review_reason"].isin(["", "low_confidence"])]
        new = review.fill_missing_from_labels(new)
        new = self.finalize(new, catalog)
        return pd.concat([curated, new], ignore_index=True), gaps

    # ---- 4. splits, pools, teacher, datasets -------------------------------------------------
    @staticmethod
    def _bucket(rid: str) -> float:
        return int(hashlib.sha3_256(rid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    def split(self, curated: pd.DataFrame) -> pd.DataFrame:
        out = curated.copy()
        out["_h"] = out["request_id"].map(self._bucket)
        # Requests of the published typical benchmark always go to the holdout: the same seed gives the
        # same spec per request id, so a new message for that id would be a paraphrase of a benchmark
        # case.
        bench = self._benchmark_ids()
        out["_h"] = [(-1.0 if rid in bench else h) for rid, h in zip(out["request_id"], out["_h"])]
        out = out.sort_values("_h").reset_index(drop=True)
        n_h, n_r = max(self.cfg.n_holdout, int((out["_h"] < 0).sum())), self.cfg.n_rl
        out["split"] = "sft"
        out.loc[: n_h - 1, "split"] = "holdout"
        # RL learns only where rollouts score differently: favor requests with a situation-implied
        # need (where SFT fails most) for ~70% of the RL prompts.
        rest = out.index[n_h:]
        cue = [i for i in rest if out.at[i, "situation"]]
        pick = (cue[: int(0.7 * n_r)])
        other = [i for i in rest if i not in set(pick)]
        pick += other[: n_r - len(pick)]
        out.loc[pick, "split"] = "rl"
        return out.drop(columns="_h")

    def pools(self, df: pd.DataFrame, catalog: pd.DataFrame) -> dict[str, list[dict]]:
        return {r["request_id"]: self.build_pool(clean_record(r), catalog)
                for r in df.to_dict(orient="records")}

    def teach(self, df: pd.DataFrame, pools: dict) -> pd.DataFrame:
        items = [(r, pools[r["request_id"]]) for r in df.to_dict(orient="records")]
        items = [(clean_record(r), p) for r, p in items]
        res = teacher.teach_many(items, cache=self.cache("teacher"), workers=self.cfg.workers,
                                 region=self.s.region)
        return pd.DataFrame(res)

    def quality_filter(self, split_df: pd.DataFrame, pools: dict, taught: pd.DataFrame,
                       exclude_ids: set | None = None) -> tuple[pd.DataFrame, set, dict]:
        """Final training-data gate (after the teacher):
        - teacher answers must pass the hard constraints (already enforced) AND at least one
          pick must match a stated interest (off-topic answers judged ~0.47 vs ~0.76)
        - RL prompts must have at least one relevant feasible candidate
        - optional explicit exclusions (e.g. rows a reviewer flagged later)"""
        reqs = {r["request_id"]: clean_record(r) for r in split_df.to_dict(orient="records")}
        exclude_ids = set(exclude_ids or ())
        ok = taught[taught["ok"] == True].copy()  # noqa: E712
        ok["interest_match"] = [
            pools_mod.interest_match_rate(reqs[r.request_id], pools[r.request_id],
                                          [p["id"] for p in json.loads(r.answer)["picks"]])
            for r in ok.itertuples()]
        keep = ok[(ok["interest_match"] > 0) & ~ok["request_id"].isin(exclude_ids)].copy()
        # Response budget: SFT targets must fit SFT_MAX_COMPLETION_TOKENS. Applied to the SFT split
        # only, so the holdout/eval set is unchanged.
        from . import tokens
        keep["completion_tokens"] = tokens.count(keep["completion"].tolist(), self.s.region)
        sft_ids = set(split_df.loc[split_df["split"] == "sft", "request_id"])
        too_long = keep["request_id"].isin(sft_ids) & (keep["completion_tokens"] > SFT_MAX_COMPLETION_TOKENS)
        n_too_long = int(too_long.sum())
        keep = keep[~too_long]
        rl_ids = {i for i in split_df.loc[split_df["split"] == "rl", "request_id"]
                  if i not in exclude_ids and pools_mod.has_relevant_option(reqs[i], pools[i])}
        stats = {"teacher_ok": len(ok), "off_topic_dropped": int((ok["interest_match"] == 0).sum()),
                 "excluded": int(ok["request_id"].isin(exclude_ids).sum()),
                 "over_token_budget_dropped": n_too_long,
                 "teacher_kept": len(keep),
                 "rl_prompts_kept": len(rl_ids),
                 "rl_prompts_dropped": int((split_df["split"] == "rl").sum()) - len(rl_ids)}
        return keep, rl_ids, stats

    def write_datasets(self, split_df: pd.DataFrame, pools: dict,
                       taught: pd.DataFrame, exclude_ids: set | None = None) -> dict[str, str]:
        reqs = {r["request_id"]: r for r in split_df.to_dict(orient="records")}
        ok, _, self.filter_stats = self.quality_filter(split_df, pools, taught, exclude_ids)
        reqs, pools, rl_ids, rl_info = self.rl_prompt_set(split_df, pools, exclude_ids)
        self.filter_stats["rl_prompts"] = rl_info
        by_split = {s: set(g["request_id"]) for s, g in split_df.groupby("split")}
        by_split["rl"] = rl_ids
        t_sft = [t for t in ok.to_dict(orient="records") if t["request_id"] in by_split["sft"]]
        t_hold = [t for t in ok.to_dict(orient="records")
                  if t["request_id"] in by_split["holdout"]]
        d = self.path("datasets", "x").parent
        files = {
            "sft": formats.write_jsonl([formats.sft_row(t) for t in t_sft], d / "sft_train.jsonl"),
            "rlvr": formats.write_jsonl(self._rlvr_rows(reqs, pools, by_split["rl"]),
                                        d / "rlvr_train.jsonl"),
            "eval": formats.write_jsonl(
                [formats.eval_row(t, clean_record(reqs[t["request_id"]]), pools[t["request_id"]])
                 for t in t_hold], d / "eval.jsonl"),
        }
        holdout = [{"request": {k: (list(v) if hasattr(v, "tolist") or isinstance(v, list) else v)
                                for k, v in reqs[t["request_id"]].items()
                                if not k.startswith(("lbl_", "dis_", "safety_", "_"))},
                    "candidates": pools[t["request_id"]], "reference": json.loads(t["answer"])}
                   for t in t_hold]
        files["holdout"] = formats.write_jsonl(holdout, d / "holdout_cases.jsonl")
        return {k: str(v) for k, v in files.items()}

    # Benchmark / demo files that RL training prompts must never overlap.
    HELD_OUT = ("app_scenarios.jsonl", "benchmark_cases.jsonl", "extreme_cases.jsonl",
                "hard_known_cases.jsonl", "hard_novel_cases.jsonl")

    @staticmethod
    def _benchmark_ids() -> set:
        p = DATA_DIR / "benchmark_cases.jsonl"
        return {json.loads(l)["request"]["request_id"] for l in open(p)} if p.exists() else set()

    def _held_out_messages(self) -> set:
        out = set()
        for f in self.HELD_OUT:
            p = DATA_DIR / f
            if p.exists():
                out |= {json.loads(l)["request"]["message"] for l in open(p)}
        return out

    def extreme_rl_requests(self, n: int, catalog: pd.DataFrame) -> list[dict]:
        """n Extreme requests for RL training (generate.extreme_spec with a run-specific seed, ids
        "xr…"). Messages that name the hidden need, or appear in any held-out file, are dropped."""
        import random
        rng, specs, tried = random.Random(self.cfg.seed * 1000 + 17), [], 0
        while len(specs) < int(n * 1.5) + 5 and tried < n * 600 + 600:
            tried += 1
            sp = generate.extreme_spec(rng, tried, catalog)
            if sp:
                sp.pop("_difficulty")
                sp["request_id"] = f"xr{tried:05d}"
                specs.append(sp)
        msgs = generate.write_messages(specs, cache=self.cache("messages_extreme_rl"),
                                       workers=self.cfg.workers, region=self.s.region)
        held = self._held_out_messages()
        out = []
        for sp in specs:
            m = msgs.get(sp["request_id"])
            if m and m not in held and not any(h.replace("_", " ") in m.lower() for h in sp["hidden_avoid"]):
                out.append({**sp, "message": m, "source": "extreme_rl"})
            if len(out) >= n:
                break
        return out

    def rl_prompt_set(self, split_df: pd.DataFrame, pools: dict, exclude_ids: set | None = None):
        """RL prompts: the RL split (filtered exactly as quality_filter) with rl_extreme_share of them
        replaced by fresh Extreme requests (same total, so the number of RL steps is unchanged).
        Returns (reqs, pools, ids, info)."""
        exclude_ids = set(exclude_ids or ())
        reqs = {r["request_id"]: r for r in split_df.to_dict(orient="records")}
        natural = sorted((i for i in split_df.loc[split_df["split"] == "rl", "request_id"]
                          if i not in exclude_ids and pools_mod.has_relevant_option(clean_record(reqs[i]), pools[i])),
                         key=self._bucket)
        share = self.cfg.rl_extreme_share
        n_ext = int(round(share * len(natural)))
        info = {"natural": len(natural) - n_ext, "extreme": 0}
        if not n_ext:
            info["natural"] = len(natural)
            return reqs, pools, set(natural), info
        cat = cat_mod.load_clean()
        ext = self.extreme_rl_requests(n_ext, cat)
        keep = natural[: len(natural) - len(ext)]
        reqs, pools = dict(reqs), dict(pools)
        for r in ext:
            reqs[r["request_id"]] = r
            pools[r["request_id"]] = self.build_pool(r, cat)
        info.update(natural=len(keep), extreme=len(ext))
        return reqs, pools, set(keep) | {r["request_id"] for r in ext}, info

    def _rlvr_rows(self, reqs: dict, pools: dict, rl_ids) -> list[dict]:
        """RLVR rows with the ground truth of the hybrid reward (judge/reward_lambda.py),
        including a teacher reference answer per prompt (cached teacher calls; None where the
        teacher found no valid answer)."""
        ids = sorted(rl_ids)
        from . import difficulty
        rl = pd.DataFrame([reqs[i] for i in ids])
        taught = {t["request_id"]: t for t in self.teach(rl, pools).to_dict(orient="records")}
        rows = []
        for i in ids:
            r, pool = clean_record(reqs[i]), pools[i]
            t = taught.get(i) or {}
            ref = json.loads(t["answer"]) if t.get("ok") else None
            rows.append(formats.rlvr_row(build_prompt(r, pool), r, pool,
                                         difficulty.grade(r, pool)["score"], ref))
        return rows

    def upload(self, files: dict[str, str]) -> dict[str, str]:
        s3 = boto3.client("s3", region_name=self.s.region)
        out = {}
        for name, path in files.items():
            key = f"{self.s.prefix}/{self.s.run_name}/datasets/{path.split('/')[-1]}"
            s3.upload_file(path, self.s.bucket, key)
            out[name] = f"s3://{self.s.bucket}/{key}"
        return out


def to_records_safe(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))
