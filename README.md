# Bài tập lớn: Deep learning for motion deblurring, học phần Nhập môn Học máy và Khai phá dữ liệu, kì 20252 Mã lớp 168732

## 1. Dataset

Tải dataset từ Google Drive:

<https://drive.google.com/drive/u/0/folders/1t_NCjFa4kvDnWY1PVI70Kx8wAPaHuObB>

Đặt folder `datasets/` ở root repo theo cấu trúc:

```text
IntroML/
├── DeblurGAN/
├── DeblurGANv2/
├── Restormer/
├── Uformer/
├── eval/
├── datasets/
│   └── deblurring/
│       ├── train/
│       │   └── GoPro/
│       │       ├── full/
│       │       │   ├── input/
│       │       │   └── target/
│       │       └── crops/
│       │           ├── input_crops/
│       │           └── target_crops/
│       ├── val/
│       │   └── GoPro/
│       │       ├── input_crops/
│       │       └── target_crops/
│       └── test/
│           ├── GoPro/
│           │   ├── input/
│           │   └── target/
│           ├── HIDE/
│           │   ├── input/
│           │   └── target/
│           ├── RealBlur_J/
│           │   ├── input/
│           │   └── target/
│           └── RealBlur_R/
│               ├── input/
│               └── target/
├── muon.py
├── README.md
└── requirements.txt
```

## 2. Environment

```bash
cd IntroML
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

Cài thêm package local của Restormer:

```bash
cd Restormer
pip install -e . --no-build-isolation
cd ..
```

## 3. Train/Test

### DeblurGAN

```bash
cd DeblurGAN
bash train.sh
bash test.sh
```

Chỉnh trong `train.sh`:

- `EXPERIMENT_NAME`: tên experiment/checkpoint.
- `DATAROOT`: folder train.
- `PHASE`: subfolder train, mặc định `full`.
- `BATCH_SIZE`: batch size.
- `CROP_SIZE`: crop size.

Chỉnh trong `test.sh`:

- `NAME`: tên checkpoint folder.
- `WHICH_EPOCH`: epoch cần load.
- `BLURRED_DIR`: folder ảnh blur.
- `SHARP_DIR`: folder ground truth.
- `RESULTS_DIR`: folder output.
- `GPU_IDS`: GPU id, dùng `-1` cho CPU.

### DeblurGANv2

```bash
cd DeblurGANv2
bash train.sh
bash test.sh
```

Chỉnh trong `train.sh`:

- `CONFIG_PATH`: file config train.
- `AUTO_RESUME`: resume từ checkpoint `last_*.h5`.

Chỉnh trong `test.sh`:

- `TEST_DIR`: folder test có `input/` và `target/`.
- `WEIGHTS_PATH`: checkpoint `.h5`.
- `OUT_DIR`: folder output.

Chỉnh hyperparameter trong `config/config.yaml`: `num_epochs`, `batch_size`, `train_batches_per_epoch`, `val_batches_per_epoch`, `optimizer.lr`, scheduler, `files_a`, `files_b`.

### Uformer

```bash
cd Uformer
bash script/train_motiondeblur.sh
bash script/test.sh
```

Chỉnh trong `script/train_motiondeblur.sh`:

- `ARCH`: kiến trúc.
- `BATCH_SIZE`: batch size.
- `GPU`: GPU ids.
- `TRAIN_DIR`, `VAL_DIR`: folder dataset.
- `SAVE_DIR`, `ENV_NAME`: folder log/checkpoint.
- `NEPOCH`, `CHECKPOINT`: số epoch và tần suất lưu.
- `OPTIMIZER`: `adam`, `adamw`, hoặc `SingleDeviceMuonWithAuxAdam`.
- `LR_INITIAL`, `WEIGHT_DECAY`, `MUON_*`: tham số optimizer.

Chỉnh trong `script/test.sh`:

- `DATASET`: `GoPro`, `HIDE`, `RealBlur_J`, `RealBlur_R`, hoặc `RealBlur_J,RealBlur_R`.
- `WEIGHTS`: checkpoint `.pth`.
- `INPUT_DIR`: root dataset.
- `RESULT_DIR`: folder output.
- `ARCH`: kiến trúc.
- `GPUS`: GPU id.

### Restormer

```bash
cd Restormer
bash train.sh
bash test.sh
```

Chỉnh trong `train.sh`:

- `CONFIG`: file yaml train.
- `AUTO_RESUME`: resume training.
- `OPTIMIZER`: `Adam`, `AdamW`, hoặc `SingleDeviceMuonWithAuxAdam`.
- `LR`, `WEIGHT_DECAY`, `BETA1`, `BETA2`: tham số optimizer.
- `MUON_*`: tham số optimizer Muon.

Chỉnh trong `Motion_Deblurring/eval_all_motiondeblur.sh`:

- `WEIGHTS`: checkpoint `.pth`.
- `DATA_ROOT`: root dataset.
- `RESULT_ROOT`: folder output.

## 4. Checkpoint

Khi infer, truyền đúng checkpoint `.pth`/`.h5` và đúng folder input. Code phải dựng đúng architecture tương ứng với checkpoint để load state dict thành công.
