# setup.py — Install all dependencies for Phase 3 (GPU default)
import subprocess, sys

def run(cmd):
    print(f"\n>>> {cmd}")
    subprocess.check_call(cmd, shell=True)

print("=" * 55)
print("  OFFICE TRACKING SYSTEM — Setup (Phase 3, GPU)")
print("=" * 55)

# Core
run(f"{sys.executable} -m pip install --upgrade pip")
run(f"{sys.executable} -m pip install opencv-python numpy pandas scikit-learn flask Pillow openpyxl")

# InsightFace + ONNX Runtime GPU
run(f"{sys.executable} -m pip uninstall -y onnxruntime 2>nul")
run(f"{sys.executable} -m pip install insightface onnxruntime-gpu")

# YOLO + ByteTrack
run(f"{sys.executable} -m pip install ultralytics boxmot")

# PyTorch (CUDA 12.1) — needed by torchreid OSNet + YOLO GPU
run(f"{sys.executable} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

# torchreid (OSNet ReID)
try:
    run(f"{sys.executable} -m pip install git+https://github.com/KaiyangZhou/deep-person-reid.git")
except Exception:
    print("[WARN] torchreid install failed. Colour-histogram ReID fallback will be used.")

print("\n" + "=" * 55)
print("  Setup complete (GPU).")
print("  Set USE_GPU = True in config.py (default).")
print("  Next: python enroll_employees.py")
print("  Then: python run.py")
print("=" * 55)
