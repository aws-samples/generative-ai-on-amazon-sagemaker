"""Train and evaluate: SFT, then SFT -> RLVR with the hybrid reward, each evaluated right after it
finishes (fast evaluation). Same code path as the attendee notebooks (`octank.arms`); the
instructor runs it on the instructor dataset before the event. Resumable (state in work/<run>/state.json).

    nohup ~/.venvs/gift_concierge/bin/python scripts/train_eval.py instructor > work/instructor/logs/train_eval.out 2>&1 &

Steps: deploy/verify the reward Lambda (octank.reward) -> SFT (config.SFT_HP) -> fast eval ->
RLVR from the SFT model (config.RLVR_HP) -> fast eval. Per-case results: work/<run>/eval_<arm>.csv.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from octank import arms, config, reward, state, train, workflow  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "instructor"
S = config.get_settings(RUN)
MPG = f"octank-concierge-{RUN}"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    ds = state.load(S)["datasets"]
    sess = train.session()
    mpg = train.ensure_model_package_group(MPG, f"Octank gift concierge ({RUN})")
    lam = state.load(S).get("reward_lambda") or reward.deploy_hybrid_lambda(S.region, role_arn=config.REWARD_ROLE_ARN)
    reward.verify_matches(lam, S.region)
    state.update(S, reward_lambda=lam)
    sft_ds = train.register_dataset(f"octank-{RUN}-sft", ds["sft"])

    def make_sft(start_from):
        return workflow.sft_trainer(S, start_from, sft_ds, mpg, sess)

    def make_rlvr(start_from):
        return workflow.rlvr_trainer(S, start_from, ds["rlvr"], MPG, sess, lam, f"octank-{RUN}-rlvr")

    arms.start(S, "sft", make_sft, evaluate=True)
    arms.start(S, "sft_rlvr", make_rlvr, after="sft", evaluate=True)
    while True:
        st = arms.status(S)
        log("\n" + st.to_string())
        if all(p in ("done", "FAILED") for p in st.phase):
            break
        time.sleep(300)
    df = arms.results(S)
    if len(df):
        log("\n" + workflow.by_problem_type(df).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
