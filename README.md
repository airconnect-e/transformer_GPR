# transformer_GPR

#structure path
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
│ └─ masks/ # (optional — ถ้าไม่มี test masks จะข้ามการคำนวณ mIoU)
├─ outputs_segformer_road/ # OUTPUT_DIR (saved checkpoints)
├─ segformer_trained_export/ # SAVE_PATH (saved final model & processor)
├─ tranformer_GPR.py or .ipynb
└─ requirements.txt
