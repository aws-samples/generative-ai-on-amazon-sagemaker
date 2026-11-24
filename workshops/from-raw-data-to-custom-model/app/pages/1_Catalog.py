"""Catalog browser: every product the concierge can recommend, plus the current request's candidates."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from octank import catalog  # noqa: E402
from octank.util import as_list  # noqa: E402

st.set_page_config(page_title="Octank catalog", page_icon="📚", layout="wide")


@st.cache_data
def load():
    c = catalog.load_clean().copy()
    c["interests"] = c["interests"].apply(as_list)
    c["attributes"] = c["attributes"].apply(as_list)
    return c


cat = load()
st.title("📚 Octank product catalog")
st.caption(f"{len(cat):,} products in {cat.category.nunique()} categories - everything the concierge "
           f"can recommend. For each request it sees 80 of them as candidates. "
           "The models see only name, category, price and description; the interest and attribute tags "
           "below are the hidden ground truth used by the rule checks.")

req = st.session_state.get("req_now")
pool_ids = st.session_state.get("pool_ids") or []
view = st.radio("Show", ["Whole catalog", "Candidates for the current request"], horizontal=True,
                disabled=not pool_ids,
                help=None if pool_ids else "Open the concierge page first to build a request")

df = cat
if view.startswith("Candidates") and pool_ids:
    df = cat.set_index("id").loc[[i for i in pool_ids if i in set(cat.id)]].reset_index()
    budget, n = float(req["budget_total"]), int(req.get("num_items") or 1)
    interests = set(as_list(req.get("interests")))
    explicit, hidden = set(as_list(req.get("avoid"))), set(as_list(req.get("hidden_avoid")))

    def role(r):
        attrs, match = set(r.attributes), bool(interests & set(r.interests))
        if hidden & attrs:
            return "🔒 trap: hidden need" + (" (matches interests)" if match else "")
        if explicit & attrs:
            return "⛔ trap: avoided attribute" + (" (matches interests)" if match else "")
        if r.price > budget:
            return "💸 over the whole budget"
        if r.price > budget / n:
            return "⚠️ pricey for a multi-item set" if n > 1 else "💸 over budget"
        return "✅ real match" if match else "▫️ generic / filler"

    df = df.assign(role=df.apply(role, axis=1))
    _md = lambda t: str(t).replace("$", "\\$")  # noqa: E731  ('$...$' would render as LaTeX)
    st.markdown(_md(f"**Request:** {req.get('message', '')}"))
    st.caption(f"{n} item(s), \\${budget:.0f} total · interests: {', '.join(sorted(interests)) or '-'} · "
               f"avoid: {', '.join(sorted(explicit)) or '-'}"
               + (f" · 🔒 implied by situation: {', '.join(sorted(hidden))}" if hidden else ""))
    st.dataframe(df.role.value_counts().rename_axis("role").reset_index(name="candidates"),
                 hide_index=True)

# ---- filters --------------------------------------------------------------------------------
f1, f2, f3 = st.columns([2, 2, 2])
q = f1.text_input("Search name or description", placeholder="e.g. vinyl, candle, coffee")
cats = f2.multiselect("Category", sorted(cat.category.unique()))
lo, hi = float(cat.price.min()), float(cat.price.max())
price = f3.slider("Price ($)", lo, hi, (lo, hi))
f4, f5 = st.columns(2)
ints = f4.multiselect("Has interest tag", sorted({i for t in cat.interests for i in t}))
no_attr = f5.multiselect("Exclude attribute", sorted({a for t in cat.attributes for a in t}))

m = df.price.between(*price)
if q:
    m &= (df.name.str.contains(q, case=False, regex=False)
          | df.description.str.contains(q, case=False, regex=False))
if cats:
    m &= df.category.isin(cats)
if ints:
    m &= df.interests.apply(lambda t: bool(set(t) & set(ints)))
if no_attr:
    m &= ~df.attributes.apply(lambda t: bool(set(t) & set(no_attr)))
out = df[m]

st.markdown(f"**{len(out):,}** of {len(df):,} products")
cols = (["role"] if "role" in out else []) + ["id", "name", "category", "price", "interests",
                                              "attributes", "description"]
st.dataframe(out[cols].sort_values(["role", "price"] if "role" in out else ["category", "price"]),
             hide_index=True, width="stretch", height=560,
             column_config={"price": st.column_config.NumberColumn(format="$%.2f"),
                            "description": st.column_config.TextColumn(width="large")})

with st.expander("Catalog at a glance"):
    g1, g2 = st.columns(2)
    g1.bar_chart(cat.category.value_counts())
    g2.bar_chart(pd.cut(cat.price, [0, 25, 50, 100, 200, 400]).astype(str).value_counts().sort_index())
