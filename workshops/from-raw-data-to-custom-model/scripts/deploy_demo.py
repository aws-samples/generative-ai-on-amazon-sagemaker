"""Create the shared demo endpoint and attach the demo adapters (for the Streamlit app).

    nohup ~/.venvs/gift_concierge/bin/python scripts/deploy_demo.py [<training-plan-arn>] > work/demo/deploy.out 2>&1 &

On-demand by default: the first instance type with capacity wins (at most 5 pools). Adapters
attached: instructor SFT and SFT -> RLVR (instructor/state.json) and attendee SFT and SFT -> RLVR
(work/attendee/state.json) - 4 adapters, the endpoint's limit (OPTION_MAX_LORAS=4).
Writes work/demo/state.json in the shape the app reads (endpoint, base_ic, ics).
Run the app with:  scripts/restart_app.sh demo
Remove adapters/base, keep the instance:  serving.delete_components('octank-demo', region)
Release the instance (only when asked):     serving.delete_all(..., confirm_release_endpoint=True)
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from octank import catalog, checks, config, pools, serving, state  # noqa: E402
from octank.prompts import build_prompt  # noqa: E402

PLAN = sys.argv[1] if len(sys.argv) > 1 else None
ENDPOINT = "octank-demo"
PLAN_INSTANCE = "ml.p5.4xlarge"
POOLS = ["ml.g6e.xlarge", "ml.g5.2xlarge", "ml.g6.4xlarge", "ml.g5.8xlarge", "ml.g6.8xlarge"]
S = config.get_settings("demo")
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)  # noqa: E731


def adapters():
    out = {}
    for run, prefix in (("instructor", "v2"), ("attendee", "att-v2")):
        arms = state.load(config.get_settings(run)).get("arms", {})
        for arm, suffix in (("sft", "sft"), ("sft_rlvr", "rlvr" if run == "attendee" else "sft-rlvr")):
            if arms.get(arm, {}).get("job"):
                out[f"{prefix}-{suffix}"] = arms[arm]["job"]
    return out


# model_tag="v2" names the base Model octank-demo-base-v2 - the name of the live resource; keep it
# so a redeploy matches the existing endpoint.
if PLAN:
    log(f"creating {ENDPOINT} on {PLAN_INSTANCE} (plan {PLAN})")
    serving.create_endpoint(ENDPOINT, S.role_arn, S.region, instance_type=PLAN_INSTANCE,
                            ml_reservation_arn=PLAN)
    out = serving.add_model(ENDPOINT, S.role_arn, S.region, cpu_cores=8, memory_mb=65536,
                            model_tag="v2")
else:
    log(f"creating {ENDPOINT} on-demand, instance pool {POOLS}")
    serving.create_endpoint(ENDPOINT, S.role_arn, S.region, instance_pools=POOLS)
    out = serving.add_model(ENDPOINT, S.role_arn, S.region, cpu_cores=2, memory_mb=12288,
                            model_tag="v2")
state.update(S, endpoint=ENDPOINT, **out, endpoint_status="InService")
log(f"base ready: {out}")
ics = dict(state.load(S).get("ics", {}))
for key, job in adapters().items():
    if key in ics:
        continue
    ic = f"{ENDPOINT}-{key}"
    serving.add_adapter(ENDPOINT, out["base_ic"], ic, serving.adapter_uri_from_job(job, S.region), S.region)
    ics[key] = ic
    state.update(S, ics=ics)
    log(f"adapter ready: {ic}")
log("smoke test (adapters only; the base model thinks until it runs out of tokens)")
case = __import__("json").loads(open(config.DATA_DIR / "benchmark_cases.jsonl").readline())
for name, ic in ics.items():
    r = serving.chat(ENDPOINT, ic, build_prompt(case["request"], case["candidates"]), S.region)
    ck = checks.check_text(r["text"], case["request"], case["candidates"])
    log(f"{name:16s} {r['latency_s']:6.1f}s hard_pass={ck['hard_pass']} tokens={r['usage']}")
log("DONE")
