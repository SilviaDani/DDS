import os

# Folder containing the .txt files
folder_path = "/home/sdani/DDS/SR4IR/datasets/VOC_COCO/labels/val_annotations"   # Change this if needed

# Common starting text to remove
prefix_to_remove = "visdrone_"

for filename in os.listdir(folder_path):
    if filename.endswith(".txt") and filename.startswith(prefix_to_remove):
        old_path = os.path.join(folder_path, filename)

        # Remove the prefix
        new_filename = filename[len(prefix_to_remove):]
        new_path = os.path.join(folder_path, new_filename)

        os.rename(old_path, new_path)
        print(f'Renamed: "{filename}" -> "{new_filename}"')

print("Done.")