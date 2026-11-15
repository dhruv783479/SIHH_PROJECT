import os
import sys
sys.path.insert(0, os.path.abspath("."))
import cv2
from pipeline.deep_inference_pipeline import DeepInferencePipeline

pipe = DeepInferencePipeline()
test_images = [
    ("Macro Pothole (crop5)", "reports/crop5_macro_pothole.png"),
    ("Alamy Street Pothole", "C:/Users/Dell/.gemini/antigravity/brain/2e3fd6d0-1829-4630-8460-e1b9f8293d96/.user_uploaded/media_1794670652002_83f72cbf.jpg"),
    ("Earthen Multi-pothole Road", "datasets/02_kaggle_pothole_600/real_images/user_upload_media_1794667697851_69045724.jpg"),
    ("Kaggle Asphalt Road (Annotated)", "datasets/02_kaggle_pothole_600/real_images/kag_pothole-annotated_2_00001_potholes1.jpg"),
    ("Kaggle Real Cavity #01", "datasets/02_kaggle_pothole_600/real_images/kaggle_real_pothole_01.jpg"),
]

for tag, p in test_images:
    if not os.path.exists(p):
        print(f"\n=== {tag} === [SKIPPED: Not found {p}]")
        continue
    res = pipe.audit_image(p)
    dets = res.get("all_detections", [])
    distress_dets = [d for d in dets if d.get("is_distress")]
    surf = res.get("surface_material", {}).get("name")
    surf_conf = res.get("surface_material", {}).get("confidence")
    print(f"\n=== {tag} ===")
    print(f"Surface: {surf} (conf={surf_conf}) | Distress count: {len(distress_dets)} | Total detections: {len(dets)}")
    for i, d in enumerate(dets):
        print(f"  #{i+1}: {d.get('class_name')} box={d.get('bbox_pixels')} conf={d.get('confidence')} is_distress={d.get('is_distress')} area={d.get('surface_area_m2')}m2")
