"""
download_model.py
Downloads Real-ESRGAN and converts it to TFLite format.
Run this once, then copy realesrgan.tflite to app/src/main/assets/

Requirements:
    pip install tensorflow torch onnx onnx-tf requests
"""

import os
import sys

def check_deps():
    missing = []
    for pkg in ['tensorflow', 'torch', 'onnx', 'onnx_tf']:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install tensorflow torch onnx onnx-tf")
        sys.exit(1)

def download_model():
    import requests
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    filename = "RealESRGAN_x4plus.pth"
    if os.path.exists(filename):
        print(f"Model already downloaded: {filename}")
        return filename

    print("Downloading Real-ESRGAN model (~67 MB)...")
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    downloaded = 0
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(f"\r  {pct}% ({downloaded // 1024 // 1024} MB)", end='', flush=True)
    print(f"\nDownloaded: {filename}")
    return filename

def convert_to_tflite(pth_path):
    import torch
    import onnx
    import numpy as np

    print("\nStep 1: Loading PyTorch model...")
    # Minimal RRDB architecture for conversion
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        model.load_state_dict(torch.load(pth_path, map_location='cpu')['params_ema'])
    except ImportError:
        print("basicsr not found. Installing...")
        os.system("pip install basicsr")
        from basicsr.archs.rrdbnet_arch import RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        model.load_state_dict(torch.load(pth_path, map_location='cpu')['params_ema'])

    model.eval()
    print("Model loaded!")

    print("\nStep 2: Exporting to ONNX...")
    dummy_input = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model, dummy_input, "realesrgan.onnx",
        opset_version=11,
        input_names=['input'],
        output_names=['output']
    )
    print("ONNX export done!")

    print("\nStep 3: Converting ONNX to TensorFlow...")
    from onnx_tf.backend import prepare
    onnx_model = onnx.load("realesrgan.onnx")
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph("realesrgan_saved_model")
    print("TF SavedModel created!")

    print("\nStep 4: Converting to TFLite (with float16 quantization)...")
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model("realesrgan_saved_model")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with open("realesrgan.tflite", "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / 1024 / 1024
    print(f"\nDone! realesrgan.tflite created ({size_mb:.1f} MB)")
    print("\nNext step: Copy realesrgan.tflite to app/src/main/assets/")

if __name__ == "__main__":
    print("=== Real-ESRGAN TFLite Converter ===\n")
    check_deps()
    pth_file = download_model()
    convert_to_tflite(pth_file)
