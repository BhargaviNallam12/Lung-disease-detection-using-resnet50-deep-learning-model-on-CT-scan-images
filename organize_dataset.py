import os
import shutil
import random

# ----------------------------
# 📁 Folder Setup
# ----------------------------
project_dir = r"C:\Users\chand\OneDrive\Desktop\major project team 19"
dataset_dir = os.path.join(project_dir, "dataset")

# Define the 5 final classes
classes = ["NORMAL", "PNEUMONIA", "ADENOCARCINOMA", "LARGE_CELL_CARCINOMA", "SQUAMOUS_CELL_CARCINOMA"]

# Create dataset folders
for subset in ["train", "val", "test"]:
    for label in classes:
        os.makedirs(os.path.join(dataset_dir, subset, label), exist_ok=True)

# ----------------------------
# ⚙️ Helper Function
# ----------------------------
def copy_images(src_folders, dest_folder):
    """Copies all images from multiple folders into one destination folder."""
    for folder in src_folders:
        if not os.path.exists(folder):
            print(f"⚠️ Folder not found: {folder}")
            continue
        for f in os.listdir(folder):
            src_path = os.path.join(folder, f)
            if os.path.isfile(src_path):
                shutil.copy(src_path, dest_folder)

def split_and_copy(src_folder, dest_label):
    """Splits one source folder into train/val/test automatically."""
    images = [f for f in os.listdir(src_folder)
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)
    total = len(images)

    train_split = int(0.7 * total)
    val_split = int(0.85 * total)

    subsets = {
        "train": images[:train_split],
        "val": images[train_split:val_split],
        "test": images[val_split:]
    }

    for subset, files in subsets.items():
        dest_subfolder = os.path.join(dataset_dir, subset, dest_label)
        for f in files:
            src_path = os.path.join(src_folder, f)
            dst_path = os.path.join(dest_subfolder, f)
            shutil.copy2(src_path, dst_path)

    print(f"✅ {dest_label}: {total} images split into train/val/test.")

# ----------------------------
# 🩻 NORMAL class
# ----------------------------
non_covid_folder = os.path.join(project_dir, "non-COVID")
if os.path.exists(non_covid_folder):
    split_and_copy(non_covid_folder, "NORMAL")
else:
    print("⚠️ non-COVID folder not found!")

# ----------------------------
# 🦠 PNEUMONIA (COVID)
# ----------------------------
covid_folder = os.path.join(project_dir, "COVID")
if os.path.exists(covid_folder):
    split_and_copy(covid_folder, "PNEUMONIA")
else:
    print("⚠️ COVID folder not found!")

# ----------------------------
# 🧫 CANCER TYPES (3 subtypes)
# ----------------------------
cancer_map = {
    "ADENOCARCINOMA": ["adenocarcinoma"],
    "LARGE_CELL_CARCINOMA": ["large.cell.carcinoma"],
    "SQUAMOUS_CELL_CARCINOMA": ["squamous.cell.carcinoma"]
}

for label, subfolders in cancer_map.items():
    src_train = [os.path.join(project_dir, "train", c) for c in subfolders]
    src_val = [os.path.join(project_dir, "valid", c) for c in subfolders]
    src_test = [os.path.join(project_dir, "test", c) for c in subfolders]

    copy_images(src_train, os.path.join(dataset_dir, "train", label))
    copy_images(src_val, os.path.join(dataset_dir, "val", label))
    copy_images(src_test, os.path.join(dataset_dir, "test", label))

print("\n🎉 Dataset organized successfully into 5 classes:")
print("👉 NORMAL, PNEUMONIA, ADENOCARCINOMA, LARGE_CELL_CARCINOMA, SQUAMOUS_CELL_CARCINOMA")
