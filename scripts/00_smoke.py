"""Phase 0 done-check:
   1. MPS available
   2. detector draws rotated boxes on dota8 sample images
   3. one Real-ESRGAN call on a dummy 256x256 tensor returns 1024x1024
"""
import sys
import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import SR_WEIGHTS, DETECTOR, FIGURES, IMGSZ, device  # noqa: E402

FIGURES.mkdir(parents=True, exist_ok=True)

print(f"torch            {torch.__version__}")
print(f"mps.is_available {torch.backends.mps.is_available()}")
print(f"mps.is_built     {torch.backends.mps.is_built()}")
DEV = device()
print(f"device           {DEV}")

# --- 2. detector on dota8 ------------------------------------------------
from ultralytics import YOLO                      # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402

ds = check_det_dataset("dota8.yaml")
root = __import__("pathlib").Path(ds["path"])
imgs = sorted((root / "images" / "train").glob("*"))[:3]
print(f"dota8 root       {root}")
print(f"dota8 sample     {[p.name for p in imgs]}")

model = YOLO(DETECTOR)
print(f"detector classes {len(model.names)}; cls1={model.names[1]} cls7={model.names[7]} cls8={model.names[8]}")

total_boxes = 0
for p in imgs:
    r = model.predict(str(p), imgsz=IMGSZ, conf=0.25, device=DEV, verbose=False)[0]
    n = 0 if r.obb is None else len(r.obb)
    ships = 0 if r.obb is None else int((r.obb.cls == 1).sum())
    total_boxes += n
    out = FIGURES / f"smoke_{p.stem}.png"
    r.save(filename=str(out))
    print(f"  {p.name:28s} obb={n:3d} ships={ships:3d} -> {out.name}")
print(f"TOTAL OBB BOXES  {total_boxes}")

# --- 3. SR on a dummy tile ----------------------------------------------
from spandrel import ModelLoader                  # noqa: E402

desc = ModelLoader().load_from_file(str(SR_WEIGHTS))
print(f"spandrel model   {type(desc.model).__name__} scale={desc.scale} in={desc.input_channels} out={desc.output_channels}")
sr_model = desc.to(DEV).eval()

x = torch.from_numpy(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8).copy())
x = x.permute(2, 0, 1).float().div(255)[None].to(DEV)
with torch.no_grad():
    y = sr_model(x).clamp(0, 1)
print(f"SR shape         {tuple(x.shape)} -> {tuple(y.shape)}")

ok = (torch.backends.mps.is_available() and total_boxes > 0 and tuple(y.shape) == (1, 3, 1024, 1024))
print("PHASE0:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
