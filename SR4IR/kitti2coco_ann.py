import json
import os
import cv2
from tqdm import tqdm

def convert_yolo_to_unified_coco(dataset_root, image_set, output_json):
    # Adjust "train" to "val" if you are converting the validation set
    img_dir = os.path.join(dataset_root, image_set, "val")
    ann_dir = os.path.join(dataset_root, image_set, "val_annotations")
    
    # Mapping KITTI YOLO ID -> Unified ID
    kitti_to_kittyVisdrone = {
        3: 0,        # pedestrian -> pedestrian
        4: 1,        # person_sitting -> person
        5: 2,        # cyclist -> bicycle
        0: 3,        # car -> car
        1: 4,        # van -> van
        2: 5         # truck -> truck
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
                    # YOLO format is space-separated
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    
                    # YOLO format: class_id x_center y_center width height (normalized)
                    yolo_cat_id = int(parts[0])
                    
                    if yolo_cat_id not in kitti_to_kittyVisdrone:
                        continue # Skip classes like 7 (Misc/DontCare)
                        
                    unified_cat_id = kitti_to_kittyVisdrone[yolo_cat_id]
                    
                    x_center_norm, y_center_norm, w_norm, h_norm = map(float, parts[1:5])
                    
                    # UN-NORMALIZE: Convert YOLO (0.0-1.0) to Absolute Pixels
                    abs_w = w_norm * w
                    abs_h = h_norm * h
                    abs_x = (x_center_norm * w) - (abs_w / 2.0)
                    abs_y = (y_center_norm * h) - (abs_h / 2.0)
                    
                    # Convert to integers for COCO format
                    x, y = int(abs_x), int(abs_y)
                    bw, bh = int(abs_w), int(abs_h)

                    # ==========================================
                    # NaN-PROOF BOUNDING BOX FILTER
                    # ==========================================
                    if bw <= 0 or bh <= 0:
                        dropped_boxes += 1
                        continue
                        
                    if x < 0:
                        bw += x
                        x = 0
                        fixed_out_of_bounds += 1
                    if y < 0:
                        bh += y
                        y = 0
                        fixed_out_of_bounds += 1
                        
                    if x + bw > w:
                        bw = w - x
                        fixed_out_of_bounds += 1
                    if y + bh > h:
                        bh = h - y
                        fixed_out_of_bounds += 1
                        
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
                        "segmentation": [],
                        "iscrowd": 0
                    })
                    ann_id += 1

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(coco_output, f)
        
    print(f"\nFinished! Created {output_json}")
    print(f"-> Total valid annotations successfully saved: {len(coco_output['annotations'])}")
    print(f"-> 🛡️ Dropped {dropped_boxes} degenerate boxes.")
    print(f"-> 🔧 Fixed {fixed_out_of_bounds} boxes stretching off-screen.")

# Run it
convert_yolo_to_unified_coco(
    "datasets", 
    "VOC_COCO_kitti", 
    "datasets/VOC_COCO_kitti/annotations/instances_val.json"
)