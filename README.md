# EO Seeker

C++ 기반 단일 표적 EO(전자광학) 추적형 광학탐색기 (X9조 설계과제)

카메라 영상 실시간 입력 → 단일 표적 탐지/추적 → Kalman Filter로 위치/속도 추정 → pan/tilt 서보로 표적을 화면 중앙에 유지하는 파이프라인입니다.

## 하드웨어 구성

- Raspberry Pi 4
- Camera Module 3 Standard (640×480)
- Pan-Tilt 브라켓 + MG90S 서보 2개
- PCA9685 서보 드라이버
- 서보 전용 5V 전원, microSD 64GB, 방열판+팬

PC(Windows)에서는 OpenCV 기반 소프트웨어 검증만 진행하고, 실제 서보/카메라 하드웨어 검증은 Raspberry Pi 이식 이후 진행합니다.

---

## 개발 환경 셋업 (새 데스크톱에서 최초 1회)

여러 데스크톱을 오가며 작업하는 프로젝트라, **아래 순서를 반드시 지켜야** 이 문서에서 정리한 문제들(아키텍처 꼬임, vcpkg 경로 충돌, 매번 오래 걸리는 빌드)을 다시 겪지 않습니다.

### 1. 필수 프로그램 설치

- Visual Studio (Community 이상) — **C++를 사용한 데스크톱 개발** 워크로드 포함
- [CMake](https://cmake.org/download/)
- [Ninja](https://github.com/ninja-build/ninja/releases)
- Git
- VS Code + CMake Tools 확장

### 2. vcpkg 설치 — 반드시 `C:\vcpkg` 경로에 설치

> ⚠️ 이 프로젝트의 `CMakePresets.json`은 `VCPKG_ROOT`를 `C:/vcpkg`로 **하드코딩**하고 있습니다. 다른 경로에 설치하면 Configure 단계에서 엉뚱한 vcpkg(Visual Studio 내장 버전 등)를 잡아버려 빌드가 깨집니다. 반드시 아래 경로 그대로 설치하세요.

```powershell
git clone https://github.com/microsoft/vcpkg C:\vcpkg
cd C:\vcpkg
.\bootstrap-vcpkg.bat
```

이 단계는 데스크톱마다 최초 1회만 하면 됩니다.

### 3. vcpkg 바이너리 캐시 설정 (재부팅 시 재빌드 방지)

OpenCV 같은 무거운 패키지가 **재부팅할 때마다 다시 컴파일되는 걸 막기 위해**, 빌드 결과물을 C 드라이브의 별도 폴더에 캐싱합니다.

```
VCPKG_BINARY_SOURCES = clear;files,C:/vcpkg-cache,readwrite
```

이 설정은 `CMakePresets.json`에 이미 포함되어 있어서 별도 환경 변수 설정 없이 그대로 동작합니다.

> ⚠️ **이 캐시는 로컬 전용입니다.** 같은 데스크톱에서는 재부팅해도 캐시가 유지되어 재빌드를 피할 수 있지만, **다른 데스크톱과는 공유되지 않습니다.** 새 데스크톱에서는 처음 한 번은 여전히 OpenCV를 처음부터 컴파일해야 합니다 (그 데스크톱만의 최초 1회).
>
> 여러 데스크톱 간에도 캐시를 공유하고 싶다면, 이 경로를 OneDrive/Dropbox 같은 클라우드 동기화 폴더로 바꾸면 됩니다 (계정마다 실제 경로가 다르므로 각 데스크톱에 맞게 `CMakePresets.json`을 직접 수정해야 합니다).
>
> 캐시가 적중하려면 **MSVC 컴파일러 버전이 같아야** 합니다(ABI 해시 기준). 버전이 다르면 캐시가 안 맞아 재빌드될 수 있습니다.

### 4. 반드시 "x64 Native Tools Command Prompt"에서 VS Code 실행

⚠️ **일반 시작 메뉴/바탕화면 아이콘으로 VS Code를 실행하면 안 됩니다.** 아키텍처(x86/x64) 인식이 꼬여서 링크 에러가 발생합니다.

1. 시작 메뉴 → **"x64 Native Tools Command Prompt for VS ..."** 검색 (여러 버전이 있으면 최신/사용 중인 버전 선택)
2. 해당 프롬프트에서:
   ```
   cd /d C:\eo_seeker
   code .
   ```

기존에 열려있던 VS Code 창이 있다면 **완전히 종료**한 뒤 진행하세요. (VS Code는 실행된 프로세스의 환경변수를 그대로 물려받기 때문에, 잘못된 환경에서 켠 창이 남아있으면 계속 같은 문제가 재현됩니다.)

### 5. 프로젝트 클론 및 최초 빌드

```
git clone <repo-url> C:\eo_seeker
```

VS Code에서 (위 4번 방법으로 열었다면):

1. `Ctrl+Shift+P` → `CMake: Configure`
2. Configure 로그에서 `Compiler found` 줄이 `Hostx64\x64\cl.exe`를 가리키는지 확인
3. `Ctrl+Shift+P` → `CMake: Build`

---

## CMakePresets.json 설정 설명

```json
"architecture": {
    "value": "x64",
    "strategy": "external"
},
"environment": {
    "VCPKG_ROOT": "C:/vcpkg",
    "VCPKG_BINARY_SOURCES": "clear;files,C:/vcpkg-cache,readwrite"
}
```

- `architecture`: Ninja처럼 command-line 제너레이터는 CMake가 스스로 컴파일러 환경을 세팅하지 못합니다. 이 필드는 VS Code CMake Tools가 x64용 Visual Studio 환경을 자동으로 준비하도록 하는 힌트입니다. (단, 이 자동 감지가 다중 VS 설치 환경에서 오작동하는 사례가 있어서, 4번 항목의 Native Tools Command Prompt 습관이 실질적으로 더 확실한 안전장치입니다.)
- `environment.VCPKG_ROOT`: Visual Studio의 Developer Command Prompt를 거치면 `VCPKG_ROOT`가 VS 내장 vcpkg 경로로 덮어써지는 경우가 있어서, 항상 원하는 vcpkg(`C:/vcpkg`)를 쓰도록 명시적으로 고정합니다.
- `environment.VCPKG_BINARY_SOURCES`: 3번 항목 참고 (재부팅 시 재빌드 방지용 로컬 캐시).

## 디버거 설정

MSVC로 빌드된 실행 파일은 GDB가 아니라 Visual Studio 디버거로 디버깅해야 합니다. `.vscode/settings.json`에 아래가 포함되어 있어야 합니다:

```json
{
    "cmake.useCMakePresets": "always",
    "cmake.configureOnOpen": true,
    "cmake.debugConfig": {
        "type": "cppvsdbg"
    }
}
```

GDB로 붙으면 MSVC의 예외 처리 방식을 이해하지 못해 실행 시작 직후 "Unknown signal"로 멈추는 증상이 발생합니다.

## OpenCV 빌드 옵션 (vcpkg.json)

```json
{
    "name": "eo-seeker",
    "version": "0.1.0",
    "dependencies": [
        {
            "name": "opencv4",
            "default-features": false,
            "features": ["contrib", "highgui", "msmf"]
        }
    ]
}
```

- `default-features: false` — dnn/gapi/directml 등 안 쓰는 무거운 기본 기능을 꺼서 빌드 시간 단축
- `contrib` — Tracker 모듈에서 KCF/CSRT 등 opencv_contrib 알고리즘 사용
- `highgui` — `imshow`/`waitKey` 등 디버그용 화면 출력
- `msmf` — Windows Media Foundation 비디오 백엔드. 현재 보유 중인 테스트 영상 전부가 이 백엔드만으로 정상 재생 확인됨 (`test_video_backend.cpp`로 검증)
- **ffmpeg는 의도적으로 제외**했습니다. Windows PC 테스트 환경에서는 불필요하며, vcpkg 재빌드 시간을 가장 크게 잡아먹는 항목이었습니다. 향후 MSMF로 안 열리는 영상(다른 코덱 등)이 생기면 그때 다시 추가 검토합니다.

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `LNK4272`, `x64 라이브러리가 x86 대상과 충돌` | 잘못된(x86) VS 툴체인으로 링크됨 | 4번 항목대로 x64 Native Tools Command Prompt에서 VS Code 실행 |
| `vcpkg install failed`, 경로가 `...VC\vcpkg\...`로 잡힘 | VS Developer Command Prompt가 `VCPKG_ROOT`를 덮어씀 | CMakePresets.json의 `environment.VCPKG_ROOT` 고정 확인 |
| `LNK1168: 쓰기용으로 열 수 없습니다` | 이전 실행/디버그 세션이 exe를 점유 중 | `Shift+F5`로 디버그 세션 종료, 작업 관리자에서 프로세스 확인 후 재빌드 |
| 디버그 시작 직후 "Unknown signal"로 멈춤 | GDB가 MSVC 예외 처리 방식을 이해 못함 | `.vscode/settings.json`에 `cmake.debugConfig.type: cppvsdbg` 설정 |
| 같은 데스크톱에서 재부팅마다 ffmpeg/OpenCV가 계속 재빌드됨 | vcpkg 바이너리 캐시 미설정 또는 MSVC 툴체인 버전 변경으로 ABI 불일치 | 3번 항목의 `VCPKG_BINARY_SOURCES` 로컬 캐시 설정 확인 |
| 새 데스크톱 첫 빌드가 오래 걸림 | 로컬 캐시(`C:/vcpkg-cache`)는 데스크톱 간 공유되지 않음 | 정상입니다 — 그 데스크톱의 최초 1회에 한함. 데스크톱 간 공유가 필요하면 클라우드 동기화 폴더로 전환 |
