"""Human-in-the-loop review: an in-notebook queue (ipywidgets) + a deterministic auto-resolve policy.

No Ground Truth / A2I. Decisions are rows in a DataFrame persisted to parquet (and S3):
  request_id, decision (approve|edit|reject), occasion, budget_total, relationship, reviewer, ts
"""
from __future__ import annotations

import datetime as dt
import getpass
from pathlib import Path

import pandas as pd

from .util import val
from .vocab import OCCASIONS, RELATIONSHIPS

DECISION_COLS = ["request_id", "decision", "occasion", "budget_total", "relationship",
                 "reason", "reviewer", "ts"]


def auto_resolve(row) -> dict:
    """Policy used for rows nobody reviews (and by the instructor pipeline end to end):
    - unclear intent / no budget anywhere / borderline safety -> reject
    - form disagrees with text -> trust the text (labels)
    - low confidence -> approve (form values win where present, labels fill gaps)"""
    reason = row["review_reason"]
    lbl = {k: val(row.get(f"lbl_{k}")) for k in ("occasion", "budget_total", "relationship")}
    form = {k: val(row.get(k)) for k in ("occasion", "budget_total", "relationship")}
    base = {"request_id": row["request_id"], "reason": reason, "reviewer": "auto-policy",
            "ts": dt.datetime.utcnow().isoformat()}
    if reason in ("unclear_intent", "no_budget_anywhere", "safety_borderline"):
        return {**base, "decision": "reject", "occasion": None, "budget_total": None,
                "relationship": None}
    if reason == "form_disagrees_with_text":
        return {**base, "decision": "edit", **{k: lbl[k] or form[k] for k in lbl}}
    return {**base, "decision": "approve", **{k: form[k] or lbl[k] for k in lbl}}


def apply_decisions(df: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Merge decisions back: drop rejects, overwrite edited fields."""
    out = df.copy()
    d = decisions.drop_duplicates("request_id", keep="last").set_index("request_id")
    rej = d.index[d["decision"] == "reject"]
    out = out[~out["request_id"].isin(rej)]
    for col in ("occasion", "budget_total", "relationship"):
        upd = d[d["decision"].isin(["edit", "approve"])][col].dropna()
        mask = out["request_id"].isin(upd.index)
        out.loc[mask, col] = out.loc[mask, "request_id"].map(upd)
    return out


def fill_missing_from_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with no review reason but a missing form field: fill from the (confident) labels."""
    out = df.copy()
    for col in ("occasion", "budget_total"):
        m = out[col].isna() & out[f"lbl_{col}"].notna()
        out.loc[m, col] = out.loc[m, f"lbl_{col}"]
    if "num_items" in out and "lbl_num_items" in out:
        m = out["num_items"].isna() & out["lbl_num_items"].notna()
        out.loc[m, "num_items"] = out.loc[m, "lbl_num_items"]
    return out.dropna(subset=["occasion", "budget_total"])


class ReviewQueue:
    """ipywidgets review UI. Usage in a notebook:
        q = ReviewQueue(queue_df, save_path); q.show()   # ... review ...
        decisions = q.decisions()
    """

    def __init__(self, queue: pd.DataFrame, save_path: str | Path, reviewer: str | None = None):
        self.queue = queue.reset_index(drop=True)
        self.save_path = Path(save_path)
        self.reviewer = reviewer or getpass.getuser()
        self.i = 0
        self._rows: list[dict] = []
        if self.save_path.exists():
            prev = pd.read_parquet(self.save_path)
            self._rows = prev.to_dict(orient="records")
            done = set(prev["request_id"])
            while self.i < len(self.queue) and self.queue.at[self.i, "request_id"] in done:
                self.i += 1

    def decisions(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=DECISION_COLS)

    def _save(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.decisions().to_parquet(self.save_path, index=False)

    def show(self):
        import ipywidgets as W
        from IPython.display import display

        self.header = W.HTML()
        self.body = W.HTML(layout=W.Layout(width="100%"))
        self.occ = W.Dropdown(options=[None] + OCCASIONS, description="occasion")
        self.rel = W.Dropdown(options=[None] + RELATIONSHIPS, description="relationship")
        self.bud = W.FloatText(description="budget $")
        btn = lambda label, style: W.Button(description=label, button_style=style)  # noqa: E731
        b_ok, b_edit = btn("Approve", "success"), btn("Save edits", "info")
        b_rej, b_skip = btn("Reject", "danger"), btn("Skip", "")
        b_ok.on_click(lambda _: self._decide("approve"))
        b_edit.on_click(lambda _: self._decide("edit"))
        b_rej.on_click(lambda _: self._decide("reject"))
        b_skip.on_click(lambda _: self._advance())
        self.status = W.HTML()
        self._render()
        display(W.VBox([self.header, self.body, W.HBox([self.occ, self.rel, self.bud]),
                        W.HBox([b_ok, b_edit, b_rej, b_skip]), self.status]))

    def _render(self):
        if self.i >= len(self.queue):
            self.header.value = "<h3>Queue complete</h3>"
            self.body.value = f"{len(self._rows)} decisions saved to <code>{self.save_path}</code>"
            return
        r = self.queue.iloc[self.i]
        self.header.value = (f"<h3>Item {self.i + 1} / {len(self.queue)} &nbsp; "
                             f"<span style='color:#c0392b'>[{r['review_reason']}]</span></h3>")

        def fmt(v):
            return "<i>missing</i>" if v is None or (isinstance(v, float) and pd.isna(v)) else v
        conf = r.get("lbl_confidence", float("nan"))
        self.body.value = (
            f"<div style='font-size:15px;padding:8px;background:#f4f6f7;border-radius:6px'>"
            f"&ldquo;{r['message']}&rdquo;</div>"
            "<table style='margin-top:6px'><tr><th></th><th>form</th><th>auto-label</th></tr>"
            f"<tr><td>occasion</td><td>{fmt(r['occasion'])}</td><td>{fmt(r.get('lbl_occasion'))}</td></tr>"
            f"<tr><td>budget</td><td>{fmt(r['budget_total'])}</td><td>{fmt(r.get('lbl_budget_total'))}</td></tr>"
            f"<tr><td>relationship</td><td>{fmt(r['relationship'])}</td><td>{fmt(r.get('lbl_relationship'))}</td></tr>"
            f"<tr><td>intent clear</td><td></td><td>{r.get('lbl_intent_clear')}</td></tr>"
            f"<tr><td>confidence</td><td></td><td>{conf:.2f}</td></tr></table>")
        self.occ.value = r.get("lbl_occasion") if r.get("lbl_occasion") in OCCASIONS else None
        self.rel.value = r["relationship"] if r["relationship"] in RELATIONSHIPS else None
        b = r.get("lbl_budget_total")
        self.bud.value = float(b) if b is not None and not pd.isna(b) else 0.0

    def _decide(self, decision: str):
        if self.i >= len(self.queue):
            return
        r = self.queue.iloc[self.i]
        self._rows.append({
            "request_id": r["request_id"], "decision": decision,
            "occasion": self.occ.value if decision != "reject" else None,
            "budget_total": (self.bud.value or None) if decision != "reject" else None,
            "relationship": self.rel.value if decision != "reject" else None,
            "reason": r["review_reason"], "reviewer": self.reviewer,
            "ts": dt.datetime.utcnow().isoformat()})
        self._save()
        self.status.value = f"<span style='color:#27ae60'>saved: {decision}</span>"
        self._advance()

    def _advance(self):
        self.i += 1
        self._render()
