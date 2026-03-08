# Image Enhancer - Android App

An AI-powered Android app that enhances photos using on-device processing.
No internet required after setup!

## Features
- **AI Upscaling** — Real-ESRGAN model upscales images to HD/4K
- **Noise Removal & Sharpening** — OpenCV denoising + unsharp masking
- **Face Enhancement** — ML Kit face detection + targeted improvement
- **Color Correction** — Brightness, contrast & saturation boost

---

## Setup Instructions

### Step 1 — Open in Android Studio
1. Open Android Studio
2. Click **File > Open**
3. Select this folder (`ImageEnhancer`)
4. Wait for Gradle sync to complete (~2–5 minutes first time)

### Step 2 — Add the AI Model (Required for Upscaling)
The AI upscaling model is NOT included (it's 15–25 MB).
Without it, the app falls back to basic bicubic upscaling automatically.

**To add Real-ESRGAN:**
1. Run this Python script to download & convert the model:
```bash
pip install tensorflow torch onnx onnx-tf requests
python download_model.py
```
2. Copy `realesrgan.tflite` into `app/src/main/assets/`

Or skip this step — the app still works with bicubic fallback.

### Step 3 — Build the APK
**Option A — Debug APK (for testing):**
- In Android Studio: **Build > Build Bundle(s)/APK(s) > Build APK(s)**
- APK location: `app/build/outputs/apk/debug/app-debug.apk`

**Option B — Release APK (for distribution):**
- **Build > Generate Signed Bundle/APK > APK**
- Create a keystore when prompted
- APK location: `app/build/outputs/apk/release/app-release.apk`

### Step 4 — Install on Device
```bash
# Via ADB (connect phone with USB debugging enabled):
adb install app/build/outputs/apk/debug/app-debug.apk
```
Or transfer the APK file to your phone and install it directly.

---

## Project Structure
```
ImageEnhancer/
├── app/src/main/
│   ├── java/com/example/imageenhancer/
│   │   ├── MainActivity.kt        ← Main UI + orchestration
│   │   ├── ImageUpscaler.kt       ← TFLite AI upscaling
│   │   ├── ImageProcessor.kt      ← OpenCV denoise + color
│   │   └── FaceEnhancer.kt        ← ML Kit face detection
│   ├── res/
│   │   ├── layout/activity_main.xml
│   │   └── values/themes.xml
│   ├── assets/                    ← Put realesrgan.tflite here
│   └── AndroidManifest.xml
├── build.gradle
└── README.md
```

## Requirements
- Android Studio Hedgehog or newer
- JDK 17
- Android device running Android 5.0+ (API 21+)
- Min 2GB RAM on device recommended

## Troubleshooting
| Problem | Fix |
|---------|-----|
| Gradle sync fails | File > Invalidate Caches > Restart |
| Model not found error | Add realesrgan.tflite to assets/ folder |
| Out of memory crash | Reduce image size before enhancing |
| GPU delegate crash | App automatically falls back to CPU |
