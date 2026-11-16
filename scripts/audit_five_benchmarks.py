import os
import sys

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

import cv2
from pipeline.deep_inference_pipeline import DeepInferencePipeline

pipe = DeepInferencePipeline()

images = [
    ('01_earthen', 'datasets/02_kaggle_pothole_600/real_images/user_upload_media_1794667697851_69045724.jpg'),
    ('02_kaggle', 'datasets/02_kaggle_pothole_600/real_images/kag_pothole-annotated_2_00001_potholes1.jpg'),
    ('03_crack500', 'datasets/01_rdd2022_india/real_images/1053962_DF_020_020BDF0070_06208_RAW.jpg'),
    ('04_zebra', 'datasets/10_missing_zebra_crossing/real_images/WM_10_zebra_001_RAW.png'),
    ('05_user_waterlogged', 'C:/Users/Dell/.gemini/antigravity/brain/2e3fd6d0-1829-4630-8460-e1b9f8293d96/.user_uploaded/media_1794799800018_8f601611.jpg')
]

for name, path in images:
    if not os.path.exists(path):
        print(f"Missing: {name} at {path}")
        continue
    res = pipe.audit_image(path)
    dets = [d for d in res.get('all_detections', []) if d.get('is_distress')]
    prim = res.get('primary_distress') or {}
    pci = res.get('astm_d6433_pci', {}).get('pci_score')
    print(f"=== {name} ===")
    print(f"  Primary: {prim.get('class_name')} ({prim.get('confidence')}) | PCI: {pci}")
    print(f"  Detections ({len(dets)}):")
    for i, d in enumerate(dets):
        print(f"    #{i+1}: {d.get('class_name')} conf={d.get('confidence'):.3f} box={d.get('bbox_pixels')}")
