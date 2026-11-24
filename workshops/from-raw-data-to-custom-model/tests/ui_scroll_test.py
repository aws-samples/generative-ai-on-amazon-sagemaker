"""Browser test for the demo app's scrolling (headless Chromium via Playwright).

Runs a Hard scenario with the judge, then measures the page's scroll container before/after
expanding "Why this score" and after expanding another section. Screenshots in /tmp/ui/.

    ~/.venvs/uitest/bin/python tests/ui_scroll_test.py [http://localhost:8501/]
"""
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501/"
OUT = Path("/tmp/ui")
OUT.mkdir(exist_ok=True)

MEASURE = """() => {
  const info = e => { if (!e) return null; const s = getComputedStyle(e);
    return {h: e.clientHeight, sh: e.scrollHeight, top: Math.round(e.scrollTop), oy: s.overflowY}; };
  const q = id => document.querySelector(`[data-testid="${id}"]`);
  const scrollers = [...document.querySelectorAll('*')].filter(e => {
    const s = getComputedStyle(e); return (s.overflowY === 'auto' || s.overflowY === 'scroll') && e.scrollHeight > e.clientHeight + 5;
  }).map(e => (e.getAttribute('data-testid') || e.tagName) + ':' + e.clientHeight + '/' + e.scrollHeight);
  return {stMain: info(q('stMain')), stAppViewContainer: info(q('stAppViewContainer')),
          stMainBlockContainer: info(q('stMainBlockContainer')), html: info(document.scrollingElement),
          scrollers};
}"""

MAIN = "() => document.querySelector('[data-testid=stMain]') || document.querySelector('[data-testid=stAppViewContainer]')"

TRY_SCROLL = """() => {
  const cands = [document.scrollingElement, ...document.querySelectorAll('*')].filter(e => {
    const s = getComputedStyle(e);
    return e.scrollHeight > e.clientHeight + 5 && (s.overflowY === 'auto' || s.overflowY === 'scroll' || e === document.scrollingElement);
  });
  cands.sort((a, b) => b.clientHeight - a.clientHeight);
  const sc = cands[0] || document.scrollingElement;
  sc.scrollTop = 1e9;
  return {scrollTop: sc.scrollTop, max: sc.scrollHeight - sc.clientHeight};
}"""


def state(page, label):
    m = page.evaluate(MEASURE)
    page.screenshot(path=str(OUT / f"{label}.png"), full_page=False)
    print(f"{label:34s} {json.dumps(m)}", flush=True)
    return m


def wheel_test(page, label):
    """Real user scroll: mouse wheel over the main area; did scrollTop change?"""
    top = "() => { const m = document.querySelector('[data-testid=stMain]') || document.querySelector('[data-testid=stAppViewContainer]'); return [Math.round(m.scrollTop), m.scrollHeight - m.clientHeight]; }"
    before = page.evaluate(top)[0]
    page.mouse.move(700, 500)
    for _ in range(10):
        page.mouse.wheel(0, 600)
        time.sleep(0.05)
    time.sleep(0.5)
    after, mx = page.evaluate(top)
    print(f"{label:34s} wheel on main: scrollTop {before} -> {after} (max {mx})", flush=True)
    return after


with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1500, "height": 900})
    page.goto(URL, wait_until="networkidle")
    page.wait_for_selector("text=Find gifts", timeout=60000)
    # drop the slow base model from the comparison (chip close button in the sidebar multiselect)
    chip = page.locator('[data-baseweb="tag"]', has_text="Base model")
    if chip.count():
        chip.locator('[role="presentation"], svg').last.click()
        time.sleep(1.5)
    page.get_by_role("button", name="🔴 Random hard").click()
    time.sleep(2)
    page.get_by_role("button", name="Find gifts").click()
    print("generating ...", flush=True)
    page.wait_for_selector("text=Why this score", timeout=600000)
    while page.locator("text=judging...").count():           # wait for all judges
        time.sleep(2)
    page.wait_for_load_state("networkidle")
    time.sleep(3)
    page.evaluate("() => { const m = document.querySelector('[data-testid=stMain]') || document.querySelector('[data-testid=stAppViewContainer]'); m.scrollTop = 0; }")
    state(page, "1_after_generation")
    wheel_test(page, "1_after_generation")
    page.evaluate("() => { const m = document.querySelector('[data-testid=stMain]') || document.querySelector('[data-testid=stAppViewContainer]'); m.scrollTop = 0; }")

    page.get_by_text("Why this score (judge reasoning)").first.click()
    time.sleep(1.5)
    state(page, "2_judge_expanded")
    wheel_test(page, "2_judge_expanded")
    print("programmatic scroll:", page.evaluate(TRY_SCROLL))

    page.get_by_text("How it reasoned").first.click()
    time.sleep(1.5)
    state(page, "3_other_expanded")
    wheel_test(page, "3_other_expanded")
    b.close()
