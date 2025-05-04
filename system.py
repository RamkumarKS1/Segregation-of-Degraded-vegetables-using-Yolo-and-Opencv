import os
import glob
import random
import shutil

random.seed(42)

base_path = r"C:\Users\Ramkumar K S\OneDrive\Desktop\Project Work\Dataset\Train"

class_map = {
    'Fresh': 0,
    'Degraded': 1
}

print("🔁 Creating label files based on folder names...")

image_paths = []

for folder_name, class_id in class_map.items():
    folder = os.path.join(base_path, folder_name)
    imgs = glob.glob(os.path.join(folder, '*.jpg')) + \
           glob.glob(os.path.join(folder, '*.jpeg')) + \
           glob.glob(os.path.join(folder, '*.png'))

    for img_path in imgs:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(folder, img_name + '.txt')

        yolo_label = f"{class_id} 0.5 0.5 1.0 1.0"

        with open(label_path, 'w') as f:
            f.write(yolo_label)

        image_paths.append(img_path)

print(f"✅ Labeled {len(image_paths)} images.")


print("🔁 Splitting into train and val sets...")

random.shuffle(image_paths)
split_idx = int(0.8 * len(image_paths))
train_imgs = image_paths[:split_idx]
val_imgs = image_paths[split_idx:]


for split in ['train', 'val']:
    os.makedirs(os.path.join(base_path, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'labels', split), exist_ok=True)

def move_files(img_list, split):
    for img_path in img_list:
        filename = os.path.basename(img_path)
        label_path = os.path.splitext(img_path)[0] + ".txt"

        dst_img = os.path.join(base_path, 'images', split, filename)
        dst_lbl = os.path.join(base_path, 'labels', split, os.path.splitext(filename)[0] + '.txt')

        shutil.copy(img_path, dst_img)

        if os.path.exists(label_path):
            shutil.copy(label_path, dst_lbl)
        else:
            print(f"⚠️ Missing label for {filename}")

move_files(train_imgs, 'train')
move_files(val_imgs, 'val')

print("✅ Dataset is now YOLOv8 ready!")


yaml_path = os.path.join(base_path, 'data.yaml')
with open(yaml_path, 'w') as f:
    f.write(f"""path: {base_path.replace("\\", "/")}
train: images/train
val: images/val

names:
  0: Fresh
  1: Degraded
""")

print(f"✅ data.yaml created at: {yaml_path}")
print("🚀 You can now start training with YOLOv8!")
