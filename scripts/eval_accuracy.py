#!/usr/bin/env python3
"""GT 기반 정확도 평가 v1
1단계: python3 scripts/eval_accuracy.py --gt <clip>.gt_keys.csv --check <clip>.gt_check.csv

GT visible 값: 1=보임, 0=안 보임(양쪽 키 모두 0), -1=전환 구간(평가 제외)
"""
import argparse, glob, os
import numpy as np, pandas as pd
from scipy.interpolate import CubicSpline

CUBIC_MIN_KEYS = 4  # 이보다 키프레임이 적은 구간은 linear로 대체


def interpolate_gt(keys: pd.DataFrame, frames, method="linear") -> pd.DataFrame:
    """키프레임(frame,x,y,w,h,visible) → frames 각각의 GT (같은 컬럼)."""
    keys = keys.sort_values("frame").reset_index(drop=True)
    keys["cx"] = keys["x"] + keys["w"] / 2          # 결정 ①: 중심+크기 보간
    keys["cy"] = keys["y"] + keys["h"] / 2
    kf, kv = keys["frame"].to_numpy(), keys["visible"].to_numpy()

    frames = np.asarray(frames)
    frames = frames[(frames >= kf[0]) & (frames <= kf[-1])]   # 키프레임 범위 밖은 평가 제외
    out = pd.DataFrame({"frame": frames})
    out[["cx", "cy", "w", "h"]] = np.nan

    # 결정 ③: 앞/뒤 키의 visible이 같으면 그 값, 다르면 전환 구간(-1, 평가 제외)
    prev = np.searchsorted(kf, frames, side="right") - 1
    nxt = np.searchsorted(kf, frames, side="left")
    out["visible"] = np.where(kv[prev] == kv[nxt], kv[prev], -1)

    # visible=1 연속 구간마다 따로 보간 (가림 구간을 가로질러 맞추지 않음)
    seg_id = keys["visible"].ne(keys["visible"].shift()).cumsum()
    for _, seg in keys.groupby(seg_id):
        if seg["visible"].iloc[0] != 1:
            continue
        m = (frames >= seg["frame"].iloc[0]) & (frames <= seg["frame"].iloc[-1])
        if not m.any():
            continue
        sf = seg["frame"].to_numpy()
        for col in ("cx", "cy", "w", "h"):
            sv = seg[col].to_numpy(dtype=float)
            if method == "cubic" and len(seg) >= CUBIC_MIN_KEYS:
                out.loc[m, col] = CubicSpline(sf, sv)(frames[m])
            else:                                   # 결정 ④: 키 부족 시 linear 대체
                out.loc[m, col] = np.interp(frames[m], sf, sv)

    out["x"] = out["cx"] - out["w"] / 2
    out["y"] = out["cy"] - out["h"] / 2
    return out[["frame", "x", "y", "w", "h", "visible"]]


def interp_error(keys, check, method):
    """검증 프레임에서 보간 GT와 직접 라벨의 중심 거리 → (mean, max, n)"""
    interp = interpolate_gt(keys, check["frame"].to_numpy(), method)
    pair = interp.merge(check, on="frame", suffixes=("_i", "_c"))
    assert len(pair) == len(check), f"짝 안 맞음: {len(pair)}/{len(check)}"
    pair = pair[(pair["visible_i"] == 1) & (pair["visible_c"] == 1)]

    cx_i = pair["x_i"] + pair["w_i"] / 2
    cy_i = pair["y_i"] + pair["h_i"] / 2
    cx_c = pair["x_c"] + pair["w_c"] / 2
    cy_c = pair["y_c"] + pair["h_c"] / 2

    d = np.hypot(cx_i - cx_c, cy_i - cy_c)
    return d.mean(), d.max(), len(d)

CLE_TH = 20.0                         # Precision·재획득 성공 기준 (px)
IOU_THS = np.linspace(0, 1, 21)       # Success 곡선 임계값 (OTB 방식)
REACQ_WINDOW = 15                     # NFR-06: 재등장 후 15프레임(0.5s) 이내


def iou(a, b):
    """a, b: (N,4) 배열 [x,y,w,h] → (N,) IoU"""
    x1 = np.maximum(a[:, 0], b[:, 0]); y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2])
    y2 = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    return np.where(union > 0, inter / union, 0.0)


def evaluate_run(keys, pred):
    """추적기 메트릭 CSV 1개 vs GT → 지표 dict"""
    gt = interpolate_gt(keys, pred["frame"].to_numpy(), "linear")
    df = gt.merge(pred[["frame", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]], on="frame")
    has_pred = (df["bbox_w"] > 0).to_numpy()            # TRACK일 때만 bbox 기록, 그 외 -1
    g = df[["x", "y", "w", "h"]].to_numpy(float)
    p = df[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].to_numpy(float)
    vis = df["visible"].to_numpy()

    d = np.hypot((g[:, 0] + g[:, 2] / 2) - (p[:, 0] + p[:, 2] / 2),
                 (g[:, 1] + g[:, 3] / 2) - (p[:, 1] + p[:, 3] / 2))
    cle = np.where(has_pred, d, np.inf)                 # 결정 ①: 예측 없음 = 실패
    ov = np.where(has_pred, iou(g, p), 0.0)

    ev = vis == 1                                       # 평가 대상: GT 보임 (전환 구간 -1 제외)
    r = {
        "eval_frames": int(ev.sum()),
        "prec20": float((cle[ev] <= CLE_TH).mean()) if ev.any() else np.nan,
        "auc": float(np.mean([(ov[ev] > t).mean() for t in IOU_THS])) if ev.any() else np.nan,
        "cle_median": float(np.median(cle[ev])) if ev.any() else np.nan,
        "false_track": int((has_pred & (vis == 0)).sum()),          # 결정 ②: GT 없음인데 TRACK
        "drift": int((has_pred & ev & (cle > CLE_TH)).sum()),       # 참고: TRACK인데 표적에서 벗어남
    }

    # 재획득: GT가 0(안 보임) 구간을 거쳐 다시 1이 된 시점부터, CLE ≤ 20px이 처음 나올 때까지
    frames = df["frame"].to_numpy()
    events, delays = 0, []
    seen_zero = False
    i = 0
    while i < len(df):
        if vis[i] == 0:
            seen_zero = True
        elif vis[i] == 1 and seen_zero:
            events += 1; seen_zero = False
            j = i
            while j < len(df) and vis[j] != 0:          # 다음 소실 전까지 탐색
                if vis[j] == 1 and cle[j] <= CLE_TH:     # 결정 ③: CLE 기준
                    delays.append(frames[j] - frames[i]); break
                j += 1
        i += 1
    r["reacq_events"] = events
    r["reacq_success"] = len(delays)
    r["reacq_in15"] = int(sum(dl <= REACQ_WINDOW for dl in delays))
    r["reacq_delay_mean"] = float(np.mean(delays)) if delays else np.nan
    return r


def evaluate_commit(gt_dir, logs_dir, commit):
    rows = []
    for gt_path in sorted(glob.glob(os.path.join(gt_dir, "*.gt_keys.csv"))):
        clip = os.path.basename(gt_path)[: -len(".gt_keys.csv")]
        keys = pd.read_csv(gt_path)
        for f in sorted(glob.glob(os.path.join(logs_dir, f"{commit}_*_{clip}.csv"))):
            try:
                pred = pd.read_csv(f)
            except pd.errors.EmptyDataError:
                continue
            if pred.empty or "bbox_w" not in pred.columns:
                continue
            run = int(os.path.basename(f).split("_")[1])
            rows.append({"commit": commit, "clip": clip, "run": run, **evaluate_run(keys, pred)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", help="1단계: 보간 오차 검증용 키프레임 파일"); ap.add_argument("--check")
    ap.add_argument("--commit", help="2단계: 평가할 커밋 해시 (logs/<commit>_<run>_<clip>.csv)")
    ap.add_argument("--gt-dir", default="data/aircraft"); ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    if a.gt and a.check:
        keys, chk = pd.read_csv(a.gt), pd.read_csv(a.check)
        for m in ("linear", "cubic"):
            mean, mx, n = interp_error(keys, chk, m)
            print(f"[{m:6}] 보간 오차 mean={mean:.1f}px  max={mx:.1f}px  (n={n})")

    if a.commit:
        runs = evaluate_commit(a.gt_dir, a.logs, a.commit)
        if runs.empty:
            raise SystemExit(f"평가할 실행 없음: {a.logs}/{a.commit}_*_<clip>.csv")
        os.makedirs(a.out, exist_ok=True)
        runs.to_csv(os.path.join(a.out, f"accuracy_runs_{a.commit}.csv"), index=False)
        cols = ["prec20", "auc", "cle_median", "false_track", "drift",
                "reacq_events", "reacq_success", "reacq_in15", "reacq_delay_mean"]
        summ = runs.groupby("clip")[cols].agg(["mean", "std"])          # 결정 ④: 평균 + 편차
        summ.insert(0, ("n_runs", ""), runs.groupby("clip").size())
        summ.to_csv(os.path.join(a.out, f"accuracy_{a.commit}.csv"))
        with pd.option_context("display.width", 250, "display.max_columns", 50):
            print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
