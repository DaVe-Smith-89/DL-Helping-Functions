import fiftyone as fo
import fiftyone.zoo as foz
import cv2
import os
import numpy as np

def download_from_open_image_v7_and_crop(target_class, num_samples, output_dir):
    """
    Downloads images of a specific class and crops the objects based on bboxes.
    Crops are saved ONLY if they meet the minimum size requirement (224x224).
    """
    print(f"--- Step 1: Downloading {num_samples} samples of '{target_class}' ---")
    
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        classes=[target_class],
        max_samples=num_samples,
        shuffle=True,
        only_matching=True
    )

    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Step 2: Processing and Cropping ---")
    
    crop_count = 0
    
    for sample in dataset:
        filepath = sample.filepath
        img = cv2.imread(filepath)
        
        if img is None:
            continue

        height, width, _ = img.shape
        filename = os.path.basename(filepath).split('.')[0]

        if sample.ground_truth is None:
            continue

        for idx, detection in enumerate(sample.ground_truth.detections):
            if detection.label.lower() != target_class.lower():
                continue

            rel_x, rel_y, rel_w, rel_h = detection.bounding_box

            x1 = int(rel_x * width)
            y1 = int(rel_y * height)
            x2 = int((rel_x + rel_w) * width)
            y2 = int((rel_y + rel_h) * height)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            crop = img[y1:y2, x1:x2]

            # minimum quality filter (224x224)
            if crop.size == 0:
                continue
            
            ch, cw = crop.shape[:2]
            if ch < 224 or cw < 224:
                # Skip too-small crops
                continue

            save_path = os.path.join(output_dir, f"{filename}_{idx}.jpg")
            cv2.imwrite(save_path, crop)
            crop_count += 1

    print(f"--- Done! ---")
    print(f"Processed {num_samples} source images.")
    print(f"Generated {crop_count} cropped images ≥224x224 in: {output_dir}")

def download_from_coco_and_crop(target_class, num_samples, output_dir):
    """
    Downloads specific classes from COCO-2017 and crops the objects.
    Filters for minimum size (224x224).
    """
    print(f"--- Step 1: Downloading {num_samples} samples of '{target_class}' from COCO-2017 ---")
    
    # FiftyOne handles the downloading automatically.
    # We specify 'classes' to only download images containing our target.
    # We use 'validation' split for speed, but use 'train' for more data.
    try:
        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split="validation", 
            label_types=["detections"],
            classes=[target_class],
            max_samples=num_samples, 
            shuffle=True,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Step 2: Processing and Cropping ---")
    
    crop_count = 0
    
    for sample in dataset:
        filepath = sample.filepath
        img = cv2.imread(filepath)
        
        if img is None:
            continue

        height, width, _ = img.shape
        filename = os.path.basename(filepath).split('.')[0]

        # COCO annotations are stored in 'ground_truth'
        if sample.ground_truth is None:
            continue

        for idx, detection in enumerate(sample.ground_truth.detections):
            # Filter to ensure we only crop the requested class
            # (COCO images often contain many different objects)
            if detection.label.lower() != target_class.lower():
                continue

            # FiftyOne standardizes COCO boxes to [x, y, w, h] (normalized 0-1)
            rel_x, rel_y, rel_w, rel_h = detection.bounding_box

            # Convert to absolute pixel coordinates
            x1 = int(rel_x * width)
            y1 = int(rel_y * height)
            x2 = int((rel_x + rel_w) * width)
            y2 = int((rel_y + rel_h) * height)

            # Clip coordinates to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            # Perform the Crop
            crop = img[y1:y2, x1:x2]

            # --- Quality Filters ---
            if crop.size == 0:
                continue

            # 1. Filter small crops (must be at least 224x224)
            ch, cw = crop.shape[:2]
            if ch < 224 or cw < 224:
                continue

            # Save the crop
            save_path = os.path.join(output_dir, f"{filename}_{idx}.jpg")
            cv2.imwrite(save_path, crop)
            crop_count += 1
            print(f"Saved: {save_path}")

    print(f"--- Done! ---")
    print(f"Processed {num_samples} source images.")
    print(f"Generated {crop_count} valid crops in: {output_dir}")

