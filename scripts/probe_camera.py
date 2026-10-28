#!/usr/bin/env python3
"""카메라 캡처 단독 실측 (처리 없음)
사용: python3 scripts/probe_camera.py --backend v4l2|gst [--frames 300]
  v4l2: 현재 C++과 같은 경로 (VideoCapture(0))
  gst : libcamerasrc GStreamer 파이프라인 (대안 경로)
"""
import argparse, time
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["v4l2", "gst"], default="v4l2")
ap.add_argument("--frames", type=int, default=300)
a = ap.parse_args()

W, H, FPS = 640, 480, 30
if a.backend == "gst":
    pipe = (f"libcamerasrc ! video/x-raw,width={W},height={H},framerate={FPS}/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1")
    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
else:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, FPS); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise SystemExit(f"[FAIL] {a.backend}: 카메라 열기 실패")
print(f"[{a.backend}] backend={cap.getBackendName()}  보고값 "
      f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}"
      f"@{cap.get(cv2.CAP_PROP_FPS):.1f}")

for _ in range(30):                      # 워밍업 (자동노출·버퍼 안정화)
    cap.read()

ts, shapes, fails = [], set(), 0
for _ in range(a.frames):
    ok, img = cap.read()
    t = time.perf_counter()
    if not ok or img is None:
        fails += 1; continue
    ts.append(t); shapes.add(img.shape)
cap.release()

dt = np.diff(ts) * 1000
print(f"실제 프레임 크기: {shapes}  읽기 실패: {fails}/{a.frames}")
if len(dt):
    print(f"프레임 간격 mean={dt.mean():.2f}ms  p95={np.percentile(dt, 95):.2f}ms  max={dt.max():.2f}ms")
    print(f"실효 FPS={1000 / dt.mean():.2f}  드롭 추정(간격>50ms)={int((dt > 50).sum())}회")
    ok = (1000 / dt.mean() >= 29) and (np.percentile(dt, 95) <= 36) and shapes == {(H, W, 3)}
    print("판정:", "PASS (≥29fps, p95≤36ms, 640x480)" if ok else "FAIL")
