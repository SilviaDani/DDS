import json
import os
import cv2
from tqdm import tqdm

def convert_visdrone_yolo_to_unified_coco(dataset_root, image_set, output_json):
    img_dir = os.path.join(dataset_root, image_set, "train")
    ann_dir = os.path.join(dataset_root, image_set, "train_annotations")
    
    # Mapping VisDrone_ID -> Unified KITTI/VisDrone ID
    visdrone_to_kittyVisdrone = {
        0: 0,        # pedestrian -> pedestrian
        1: 1,        # people -> person
        2: 2,        # bicycle -> bicycle
        3: 3,        # car -> car
        4: 4,        # van -> van
        5: 5         # truck -> truck
    }

    # Define the new unified categories for the JSON header
    unified_categories = [
        {"id": 0, "name": "pedestrian"}, 
        {"id": 1, "name": "person"},
        {"id": 2, "name": "bicycle"}, 
        {"id": 3, "name": "car"}, 
        {"id": 4, "name": "van"}, 
        {"id": 5, "name": "truck"}
    ]

    coco_output = {"images": [], "annotations": [], "categories": unified_categories}
    ann_id = 1
    
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # --- TRACKERS TO SHOW YOU WHAT WAS FIXED ---
    dropped_boxes = 0
    fixed_out_of_bounds = 0

    for img_idx, img_name in enumerate(tqdm(image_files)):
        img_path = os.path.join(img_dir, img_name)
        im = cv2.imread(img_path)
        if im is None: continue
        h, w, _ = im.shape

        coco_output["images"].append({
            "id": img_idx, "file_name": img_name, "width": w, "height": h
        })

        txt_path = os.path.join(ann_dir, os.path.splitext(img_name)[0] + ".txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                for line in f:
                    # YOLO format is space-separated, NOT comma-separated
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    
                    # YOLO format: class_id x_center y_center width height (normalized)
                    vis_cat_id = int(parts[0])

                    # Apply the filtering and remapping
                    if vis_cat_id in visdrone_to_kittyVisdrone:
                        unified_cat_id = visdrone_to_kittyVisdrone[vis_cat_id]
                        
                        # Grab the normalized YOLO floats
                        x_center_norm, y_center_norm, w_norm, h_norm = map(float, parts[1:5])
                        
                        # UN-NORMALIZE: Convert YOLO (0.0-1.0) to Absolute Pixels
                        abs_w = w_norm * w
                        abs_h = h_norm * h
                        abs_x = (x_center_norm * w) - (abs_w / 2.0)
                        abs_y = (y_center_norm * h) - (abs_h / 2.0)
                        
                        # Convert to integers for standard COCO format
                        x, y = int(abs_x), int(abs_y)
                        bw, bh = int(abs_w), int(abs_h)

                        # ==========================================
                        # NaN-PROOF BOUNDING BOX FILTER
                        # ==========================================
                        
                        # 1. Drop boxes with 0 or negative width/height
                        if bw <= 0 or bh <= 0:
                            dropped_boxes += 1
                            continue
                            
                        # 2. Fix out-of-bounds coordinates (Negative X or Y)
                        if x < 0:
                            bw += x  # Shrink width by the amount it was off-screen
                            x = 0
                            fixed_out_of_bounds += 1
                        if y < 0:
                            bh += y  # Shrink height by the amount it was off-screen
                            y = 0
                            fixed_out_of_bounds += 1
                            
                        # 3. Fix boxes that stretch past the right/bottom edge of the image
                        if x + bw > w:
                            bw = w - x
                            fixed_out_of_bounds += 1
                        if y + bh > h:
                            bh = h - y
                            fixed_out_of_bounds += 1
                            
                        # 4. Final safety check after clipping (VisDrone has tiny objects!)
                        if bw <= 1 or bh <= 1:
                            dropped_boxes += 1
                            continue
                        # ==========================================
                        
                        coco_output["annotations"].append({
                            "id": ann_id,
                            "image_id": img_idx,
                            "category_id": unified_cat_id,
                            "bbox": [x, y, bw, bh],
                            "area": bw * bh,
                            "segmentation": [], # No polygons in VisDrone
                            "iscrowd": 0
                        })
                        ann_id += 1

    # Ensure the output directory exists so the save doesn't fail
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(coco_output, f)
        
    print(f"\nFinished! Created {output_json} with unified 0-5 IDs.")
    print(f"-> Total valid annotations successfully saved: {len(coco_output['annotations'])}")
    print(f"-> 🛡️ Dropped {dropped_boxes} degenerate boxes (Saved you from NaNs!).")
    print(f"-> 🔧 Fixed {fixed_out_of_bounds} boxes stretching off-screen.")

# Run it
convert_visdrone_yolo_to_unified_coco(
    "datasets", 
    "VOC_COCO_visdrone", 
    "datasets/VOC_COCO_visdrone/annotations/instances_train.json"
)