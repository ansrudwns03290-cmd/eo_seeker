#!/usr/bin/env python3
"""GT 기반 정확도 평가 v1
1단계: python3 scripts/eval_accuracy.py --gt <clip>.gt_keys.csv --check <clip>.gt_check.csv

GT visible 값: 1=보임, 0=안 보임(양쪽 키 모두 0), -1=전환 구간(평가 제외)
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True); ap.add_argument("--check")
    a = ap.parse_args()
    keys = pd.read_csv(a.gt)
    if a.check:
        chk = pd.read_csv(a.check)
        for m in ("linear", "cubic"):
            mean, mx, n = interp_error(keys, chk, m)
            print(f"[{m:6}] 보간 오차 mean={mean:.1f}px  max={mx:.1f}px  (n={n})")


if __name__ == "__main__":
    main()
