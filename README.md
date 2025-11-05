# transformer_GPR

#structure path
```bash
project-root/
├─ data/
│ ├─ train/
│ │ ├─ images/
│ │ └─ masks/
│ ├─ val/
│ │ ├─ images/
│ │ └─ masks/
│ └─ test/
│ ├─ images/
├─ outputs_segformer_road/ # OUTPUT_DIR (saved checkpoints)
├─ segformer_trained_export/ # SAVE_PATH (saved final model & processor)
├─ tranformer_GPR.py or .ipynb
└─ requirements_GPR.txt
```

#library
```bash
!pip install -q transformers
!pip install -q numpy
!pip install  pillow
!pip install datasets
!pip install transformers accelerate
!pip install evaluate
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install torch torchvision torchaudio
!pip install -U "transformers>=4.44" "datasets" "evaluate" "accelerate" -q
!pip install -U "transformers>=4.44" "datasets>=2.18" "evaluate>=0.4" "accelerate>=0.33" -q
!pip install matplotlib
!pip install -U hf_transfer -q
!pip install PyDrive
!pip install --upgrade gdown
```
#download file from google drive and extract file

```bash
import gdown
import os
import zipfile

# File ID จากลิงก์ใหม่
file_id = "1DY7N8hbjjVkd4v_6a387yysjL4WGcZtZ"
zip_path = "./gpr_dataset.zip"

# ดาวน์โหลดไฟล์ zip จาก Google Drive
gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_path, quiet=False)

# แตกไฟล์ zip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("./gpr_dataset")

print("✅ Download and unzip completed! Files are in ./gpr_dataset")
```

# Cell 1 define path dataset

```bash
# ---- แก้พาธให้ตรงโครงสร้างของคุณ ----
ROOT = Path("/workspace/gpr_dataset/data")  # <-- เปลี่ยนตรงนี้ให้ตรงตำแหน่งจริงของคุณ ***ถ้าในcolab ใช้ content แทน workspace

TRAIN_IMG_DIR = ROOT / "train/images"
TRAIN_MSK_DIR = ROOT / "train/masks"

VAL_IMG_DIR   = ROOT / "val/images"
VAL_MSK_DIR   = ROOT / "val/masks"

TEST_IMG_DIR  = ROOT / "test"   # ผู้ใช้พิมพ์ว่า test/image -> ผมถือว่าเป็น "images"   # ต้องมี ถ้าจะคำนวณ mIoU ของ test
```

# Cell 1 config
```bash
# ---- Task config ----
NUM_CLASSES   = 2              # แก้จำนวนคลาสตามงานคุณ
IGNORE_INDEX  = 255            # ถ้ามี label=255 เป็น void/ignore
IMAGE_SIZE    = 512            # ขนาดรีไซส์สำหรับโมเดล (SegFormer fine-tuned ส่วนใหญ่ 512)

MODEL_NAME    = "nvidia/segformer-b0-finetuned-ade-512-512"
OUTPUT_DIR    = "./outputs_segformer_road"  # โฟลเดอร์เซฟ checkpoint

SEED          = 42
EPOCHS        = 25
LR            = 5e-5
BATCH_SIZE    = 10
```

# Cell 3 Create dataset
```bash
# ====== Cell 3: BUILD DATASETS & MODEL (no test masks) ======
train_ds = RoadSegDataset(TRAIN_IMG_DIR, TRAIN_MSK_DIR, processor, image_size=IMAGE_SIZE, has_mask=True)
val_ds   = RoadSegDataset(VAL_IMG_DIR,   VAL_MSK_DIR,   processor, image_size=IMAGE_SIZE, has_mask=True)

# --- TEST: images only (no masks) ---
test_ds  = RoadSegDataset(TEST_IMG_DIR, None, processor, image_size=IMAGE_SIZE, has_mask=False)
```
