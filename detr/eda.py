import os 
import glob 

val_img_dir = "/data/coco_detr/val2017"
val_img_paths = glob.glob(os.path.join(val_img_dir, "*jpg"))
print(f"Validation : {len(val_img_paths)}")
