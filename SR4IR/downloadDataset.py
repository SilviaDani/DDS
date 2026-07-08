from ultralytics.data.utils import check_det_dataset as check_dataset

# ==========================================
# 1. DOWNLOAD DATASETS VIA ULTRALYTICS
# ==========================================
print("Downloading datasets... (This might take a while depending on your internet connection)")
kitti_info = check_dataset("kitti.yaml")

#visdrone_info = check_dataset("VisDrone.yaml")
