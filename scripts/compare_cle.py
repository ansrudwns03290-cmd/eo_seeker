#!/usr/bin/env python3
"""여러 실행의 GT 중심 오차(CLE)·상태·KCF 시간 비교
사용: python3 scripts/compare_cle.py move_v_fast_01 --base ~/cmp/base.csv ~/cmp/patch2.csv ~/cmp/patch_t04.csv
  GT  : data/aircraft/<clip>.gt_check.csv (없으면 .gt_keys.csv)
  위치: kcf_x/kcf_y 컬럼이 있으면 KCF 원출력(채택 여부 무관), 없으면 채택 bbox 사용
        박스 크기는 <clip>.roi 의 w,h (출력 크기는 원 ROI로 고정이므로)
  표의 '-' = 해당 프레임에 박스 없음 (KCF 실패 또는 비TRACK)
"""
import argparse, os
import numpy as np, pandas as pd


def load_gt(data_dir, clip):
    for suffix in (".gt_check.csv", ".gt_keys.csv"):
        p = os.path.join(data_dir, clip + suffix)
        if os.path.exists(p):
            return pd.read_csv(p).set_index("frame"), p
    raise SystemExit(f"GT 없음: {data_dir}/{clip}.gt_check.csv / .gt_keys.csv")


def load_roi_size(data_dir, clip):
    with open(os.path.join(data_dir, clip + ".roi")) as fp:
        _, _, w, h = map(int, fp.read().split()[:4])
    return w, h


def center_xy(run, f, roi_w, roi_h):
    """프레임 f의 박스 중심 (없으면 None)."""
    if f not in run.index:
        return None
    r = run.loc[f]
    if "kcf_x" in run.columns:
        return None if r.kcf_x < 0 else (r.kcf_x + roi_w / 2, r.kcf_y + roi_h / 2)
    return None if r.bbox_w <= 0 else (r.bbox_x + r.bbox_w / 2, r.bbox_y + r.bbox_h / 2)


def summary(name, run):
    t = run[run.fsm_state == "TRACK"]
    line = f"[{name}] 상태 {run.fsm_state.value_counts().to_dict()}"
    if "kcf_patch_w" in run.columns and len(t):
        line += f"  patch {int(t.kcf_patch_w.max())}x{int(t.kcf_patch_h.max())}"
    if "track_kcf_ms" in run.columns and len(t):
        line += (f"\n    KCF ms(TRACK) p50 {t.track_kcf_ms.quantile(.5):.1f} p95 {t.track_kcf_ms.quantile(.95):.1f}"
                 f" | total ms(TRACK) p95 {t.total_ms.quantile(.95):.1f}")
    if "kcf_x" in run.columns:
        fails = t.index[t.kcf_x < 0]
        line += f" | 첫 KCF 실패 frame {fails[0] if len(fails) else '없음'}"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("runs", nargs="+", help="비교할 실행 CSV들")
    ap.add_argument("--base", help="기준선 CSV (선택, 맨 왼쪽 열)")
    ap.add_argument("--data-dir", default="data/aircraft")
    a = ap.parse_args()

    gt, gt_path = load_gt(a.data_dir, a.clip)
    roi_w, roi_h = load_roi_size(a.data_dir, a.clip)
    paths = ([a.base] if a.base else []) + a.runs
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    runs = [pd.read_csv(os.path.expanduser(p)).set_index("frame") for p in paths]

    print(f"GT: {gt_path} ({len(gt)}프레임)  ROI 크기 {roi_w}x{roi_h}\n")
    for n, r in zip(names, runs):
        summary(n, r)

    # 프레임별 표: CLE(px)와 conf
    head = f"\n{'frame':>5} {'GT wxh':>8} " + " ".join(f"{n[:12]:>12} {'conf':>4}" for n in names)
    print(head); print("-" * len(head))
    stats = {n: [] for n in names}
    for f, g in gt.iterrows():
        if int(g.get("visible", 1)) != 1:
            continue
        gc = (g.x + g.w / 2, g.y + g.h / 2)
        cells = []
        for n, r in zip(names, runs):
            c = center_xy(r, f, roi_w, roi_h)
            e = np.hypot(c[0] - gc[0], c[1] - gc[1]) if c else np.nan
            stats[n].append(e)
            conf = r.loc[f, "conf"] if ("conf" in r.columns and f in r.index) else np.nan
            cells.append(f"{(f'{e:.1f}' if c else '-'):>12} {(f'{conf:.2f}' if conf == conf else ''):>4}")
        print(f"{f:5d} {f'{g.w:.0f}x{g.h:.0f}':>8} " + " ".join(cells))

    print("\nPrecision@20px (박스 없음 = 실패):")
    for n in names:
        e = np.array(stats[n])
        print(f"  {n}: {np.mean(np.nan_to_num(e, nan=np.inf) < 20) * 100:.0f}%  "
              f"(박스 있는 프레임 평균 CLE {np.nanmean(e):.1f}px)" if np.isfinite(e).any() else f"  {n}: 박스 없음")


if __name__ == "__main__":
    main()
