#!/usr/bin/env python3
"""EO Seeker 메트릭 CSV 재집계 (TRACK/전체 분리, 분포 통계).

파일명 규칙: <commit>_<run>_<video>.csv
사용 예: python3 scripts/aggregate_metrics.py build/logs --out results
"""
import argparse, glob, os
import pandas as pd

TIME_COLS = ["read_ms", "preprocess_ms", "track_ms", "track_kcf_ms", "total_ms"]
STATS = {
    "mean": "mean", "std": "std", "max": "max",
    "p50": lambda s: s.quantile(0.50),
    "p95": lambda s: s.quantile(0.95),
}

def parse_name(path):
    parts = os.path.basename(path)[:-4].split("_", 2)
    if len(parts) != 3 or not parts[1].isdigit():
        return None
    return parts[0], int(parts[1]), parts[2]

def summarize(sub, cols):
    r = {"n": len(sub)}
    for c in cols:
        s = sub[c].dropna()
        for k, f in STATS.items():
            r[f"{c}_{k}"] = s.agg(f) if len(s) else float("nan")
    m = r.get("total_ms_mean")
    r["fps"] = 1000.0 / m if m and m > 0 else float("nan")  # fps 컬럼은 신뢰 불가 → total_ms로 계산
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="CSV가 있는 폴더들")
    ap.add_argument("--warmup", type=int, default=0, help="각 실행의 앞 N프레임 제외 (기본 0: 코드 내 6회 예열로 충분)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rows, skipped = [], []
    files = sorted({f for d in a.dirs for f in glob.glob(os.path.join(d, "*.csv"))})
    for f in files:
        meta = parse_name(f)
        if meta is None:
            skipped.append((f, "파일명 규칙 불일치")); continue
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            skipped.append((f, "빈 파일 (헤더 없음, 0바이트)")); continue
        if df.empty:
            skipped.append((f, "frames=0 (헤더만 있음)")); continue

        commit, run, video = meta
        st = df["fsm_state"]
        info = {
            "commit": commit, "video": video, "run": run,
            "frames_total": len(df),
            "track_ratio": (st == "TRACK").mean(),
            "lost_events": int(((st == "LOST") & (st.shift() != "LOST")).sum()),
            "warmup": a.warmup, "file": f,
        }
        df = df.iloc[a.warmup:]
        cols = [c for c in TIME_COLS if c in df.columns]
        for scope, sub in (("TRACK", df[df["fsm_state"] == "TRACK"]), ("ALL", df)):
            rows.append({**info, "scope": scope, **summarize(sub, cols)})

    os.makedirs(a.out, exist_ok=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(a.out, "runs.csv"), index=False)

    summary = (runs.groupby(["commit", "video", "scope"])
               .agg(n_runs=("run", "count"), frames=("n", "mean"),
                    fps_mean=("fps", "mean"), fps_min=("fps", "min"), fps_max=("fps", "max"),
                    total_p95=("total_ms_p95", "mean"), track_ratio=("track_ratio", "mean"))
               .reset_index())
    summary.to_csv(os.path.join(a.out, "summary.csv"), index=False)

    with pd.option_context("display.width", 200, "display.max_rows", 300):
        print(summary.round(2).to_string(index=False))
    if skipped:
        print("\n[제외된 파일]")
        for f, why in skipped:
            print(f"  {f}: {why}")

if __name__ == "__main__":
    main()
