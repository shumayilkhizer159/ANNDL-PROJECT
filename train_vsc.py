#!/usr/bin/env python3
"""
ANNDL Project — Full Training Pipeline for VSC (HPC)
Converts the Jupyter notebook into a standalone script.

Usage:
    python train_vsc.py --data-dir /path/to/VOCtrainval_11-May-2012_2 --output-dir /path/to/output

Three parts:
  1. Image Classification  (3 custom CNNs + Xception transfer learning)
  2. Image Segmentation    (U-Net)
  3. Object Detection      (YOLO-v1-like)
"""

import os
import sys
import argparse
import gc
import random
import time

# ── Keras backend must be set BEFORE importing keras ──────────────────────────
os.environ["KERAS_BACKEND"] = "torch"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET

import keras
import keras.ops as kops
from keras import layers, models, callbacks

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
_IMAGE_SHAPE = (128, 128)
IMG_SIZE     = 180
SEG_SIZE     = 128
DET_SIZE     = 160
GRID_S       = 7
C_DET        = 20
_BATCH_SIZE  = 16   # V100 32GB can handle larger batches
XC_BATCH     = 16
NUM_WORKERS  = 4    # Linux supports multi-worker loading

_VOC_LABELS = (
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor",
)

NUM_CLASSES = len(_VOC_LABELS)
class_to_idx = {name: idx for idx, name in enumerate(_VOC_LABELS)}

# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    print(f"GPU memory free: {free/1e9:.2f} GB / {total/1e9:.2f} GB")

def save_fig(fig, output_dir, name):
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved plot: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_classification_data(voc_root_folder):
    annotation_dir = os.path.join(voc_root_folder, "Annotations")
    image_dir      = os.path.join(voc_root_folder, "JPEGImages")
    dataset = {}
    for xml_file in os.listdir(annotation_dir):
        if not xml_file.endswith(".xml"):
            continue
        tree    = ET.parse(os.path.join(annotation_dir, xml_file))
        root    = tree.getroot()
        fname   = root.find("filename").text
        img_id  = os.path.splitext(fname)[0]
        classes = list({obj.find("name").text for obj in root.findall("object")})
        dataset[img_id] = {"path": os.path.join(image_dir, fname), "classes": classes}
    return dataset

def get_segmentation_paths(base_dir):
    IMAGE_PATH      = os.path.join(base_dir, "JPEGImages")
    ANNOTATION_PATH = os.path.join(base_dir, "SegmentationClass")
    LISTS           = os.path.join(base_dir, "ImageSets", "Segmentation")
    img_paths, ann_paths = [], []
    with open(os.path.join(LISTS, "trainval.txt")) as f:
        names = [l.strip() for l in f if l.strip()]
    for name in names:
        img_paths.append(os.path.join(IMAGE_PATH, name + ".jpg"))
        ann_paths.append(os.path.join(ANNOTATION_PATH, name + ".png"))
    return img_paths, ann_paths

def preprocess_detection_data(voc_root_folder):
    annotation_dir = os.path.join(voc_root_folder, "Annotations")
    image_dir      = os.path.join(voc_root_folder, "JPEGImages")
    dataset = {}
    for xml_file in os.listdir(annotation_dir):
        if not xml_file.endswith(".xml"): continue
        tree = ET.parse(os.path.join(annotation_dir, xml_file))
        root = tree.getroot()
        fname  = root.find("filename").text
        img_id = os.path.splitext(fname)[0]
        size   = root.find("size")
        W, H   = int(size.find("width").text), int(size.find("height").text)
        boxes, labels = [], []
        for obj in root.findall("object"):
            label = obj.find("name").text
            bb    = obj.find("bndbox")
            xmin  = float(bb.find("xmin").text)
            ymin  = float(bb.find("ymin").text)
            xmax  = float(bb.find("xmax").text)
            ymax  = float(bb.find("ymax").text)
            boxes.append([xmin/W, ymin/H, (xmax-xmin)/W, (ymax-ymin)/H])
            labels.append(label)
        dataset[img_id] = {"boxes": boxes, "labels": labels,
                           "path": os.path.join(image_dir, fname)}
    return dataset

def encode_annotation(boxes, labels, grid_s=7):
    target = np.zeros((grid_s, grid_s, 5 + C_DET), dtype=np.float32)
    for box, label in zip(boxes, labels):
        ci = class_to_idx.get(label)
        if ci is None: continue
        rx, ry, rw, rh = box
        xc, yc = rx + rw/2, ry + rh/2
        gx, gy = min(int(xc * grid_s), grid_s-1), min(int(yc * grid_s), grid_s-1)
        if target[gy, gx, 0] == 0:
            target[gy, gx, 0]     = 1.0
            target[gy, gx, 1:5]   = [xc*grid_s - gx, yc*grid_s - gy, rw, rh]
            target[gy, gx, 5+ci]  = 1.0
    return target

# ══════════════════════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════════════════════
class VOCClassificationDataset(Dataset):
    def __init__(self, paths, labels, img_size=180, augment=False):
        self.paths  = paths
        self.labels = labels.astype(np.float32)
        aug = [T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.1, 0.05)] if augment else []
        self.transform = T.Compose([T.Resize((img_size, img_size)), *aug, T.ToTensor()])

    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), torch.tensor(self.labels[idx])

class VOCDatasetCL(Dataset):
    """Channels-last for Xception."""
    def __init__(self, paths, labels, img_size=150, augment=False):
        self.paths  = paths
        self.labels = labels.astype(np.float32)
        aug = [T.RandomHorizontalFlip()] if augment else []
        self.transform = T.Compose([T.Resize((img_size, img_size)), *aug, T.ToTensor()])

    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        t   = self.transform(img)
        t   = t.permute(1, 2, 0)    # CHW → HWC
        t   = t * 2.0 - 1.0         # [0,1] → [-1,1] for Xception
        return t, torch.tensor(self.labels[idx])

class VOCSegDataset(Dataset):
    def __init__(self, img_paths, mask_paths, size=SEG_SIZE, augment=False):
        self.img_paths  = img_paths
        self.mask_paths = mask_paths
        self.size       = size
        self.augment    = augment

    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx):
        img  = Image.open(self.img_paths[idx]).convert("RGB").resize((self.size, self.size))
        mask = Image.open(self.mask_paths[idx]).resize((self.size, self.size), Image.NEAREST)
        img  = np.array(img,  dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.uint8)
        mask[mask > 0] = 1
        mask = mask.astype(np.float32)[..., np.newaxis]
        if self.augment and random.random() > 0.5:
            img  = img[:, ::-1, :].copy()
            mask = mask[:, ::-1, :].copy()
        img  = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).permute(2, 0, 1)
        return img, mask

class VOCDetDataset(Dataset):
    def __init__(self, paths, targets, size=DET_SIZE):
        self.paths   = paths
        self.targets = targets
        self.size    = size
        self.tf      = T.Compose([T.Resize((size, size)), T.ToTensor()])

    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img), torch.from_numpy(self.targets[idx])

# ══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

# ── 1.1 Custom CNNs ──────────────────────────────────────────────────────────
def build_baseline_cnn():
    inp = keras.Input(shape=(3, IMG_SIZE, IMG_SIZE))
    x   = layers.Conv2D(32, 3, activation='relu', padding='same', data_format='channels_first')(inp)
    x   = layers.MaxPooling2D(data_format='channels_first')(x)
    x   = layers.Conv2D(64, 3, activation='relu', padding='same', data_format='channels_first')(x)
    x   = layers.MaxPooling2D(data_format='channels_first')(x)
    x   = layers.Conv2D(128, 3, activation='relu', padding='same', data_format='channels_first')(x)
    x   = layers.GlobalAveragePooling2D(data_format='channels_first')(x)
    out = layers.Dense(NUM_CLASSES, activation='sigmoid')(x)
    return keras.Model(inp, out, name="Baseline_CNN")

def build_regularised_cnn():
    inp = keras.Input(shape=(3, IMG_SIZE, IMG_SIZE))
    x   = inp
    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(filters, 3, padding='same', data_format='channels_first')(x)
        x = layers.BatchNormalization(axis=1)(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(data_format='channels_first')(x)
    x   = layers.GlobalAveragePooling2D(data_format='channels_first')(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(NUM_CLASSES, activation='sigmoid')(x)
    return keras.Model(inp, out, name="Regularised_CNN")

def res_block(x, filters):
    shortcut = layers.Conv2D(filters, 1, padding='same', data_format='channels_first')(x)
    x = layers.DepthwiseConv2D(3, padding='same', data_format='channels_first')(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 1, padding='same', data_format='channels_first')(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Add()([x, shortcut])
    return layers.ReLU()(x)

def build_resnet_like():
    inp = keras.Input(shape=(3, IMG_SIZE, IMG_SIZE))
    x   = layers.Conv2D(32, 3, padding='same', data_format='channels_first')(inp)
    x   = layers.BatchNormalization(axis=1)(x)
    x   = layers.ReLU()(x)
    for f in [32, 64, 128]:
        x = res_block(x, f)
        x = layers.MaxPooling2D(data_format='channels_first')(x)
    x   = layers.GlobalAveragePooling2D(data_format='channels_first')(x)
    x   = layers.Dropout(0.5)(x)
    out = layers.Dense(NUM_CLASSES, activation='sigmoid')(x)
    return keras.Model(inp, out, name="ResNet_Like")

# ── U-Net ─────────────────────────────────────────────────────────────────────
def build_unet(img_size=128):
    inp = keras.Input(shape=(3, img_size, img_size))

    def enc_block(x, f):
        x = layers.Conv2D(f, 3, padding='same', activation='relu', data_format='channels_first')(x)
        x = layers.Conv2D(f, 3, padding='same', activation='relu', data_format='channels_first')(x)
        p = layers.MaxPooling2D(data_format='channels_first')(x)
        return x, p

    c1, p1 = enc_block(inp, 32)
    c2, p2 = enc_block(p1,  64)
    c3, p3 = enc_block(p2, 128)

    b = layers.Conv2D(256, 3, padding='same', activation='relu', data_format='channels_first')(p3)
    b = layers.Conv2D(256, 3, padding='same', activation='relu', data_format='channels_first')(b)

    def dec_block(x, skip, f):
        x = layers.Conv2DTranspose(f, 2, strides=2, padding='same', data_format='channels_first')(x)
        x = layers.Concatenate(axis=1)([x, skip])
        x = layers.Conv2D(f, 3, padding='same', activation='relu', data_format='channels_first')(x)
        x = layers.Conv2D(f, 3, padding='same', activation='relu', data_format='channels_first')(x)
        return x

    d1 = dec_block(b,  c3, 128)
    d2 = dec_block(d1, c2,  64)
    d3 = dec_block(d2, c1,  32)

    out = layers.Conv2D(1, 1, activation='sigmoid', data_format='channels_first')(d3)
    return keras.Model(inp, out, name="UNet")

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = kops.reshape(kops.cast(y_true, 'float32'), [-1])
    y_pred_f = kops.reshape(y_pred, [-1])
    inter    = kops.sum(y_true_f * y_pred_f)
    return (2. * inter + smooth) / (kops.sum(y_true_f) + kops.sum(y_pred_f) + smooth)

# ── YOLO-v1 ──────────────────────────────────────────────────────────────────
def build_yolo(grid_s=7, num_boxes=1, num_classes=20, img_size=DET_SIZE):
    inp = keras.Input(shape=(3, img_size, img_size))
    x   = layers.Conv2D(32, 7, strides=2, padding='same', data_format='channels_first')(inp)
    x   = layers.BatchNormalization(axis=1)(x); x = layers.LeakyReLU(0.1)(x)
    x   = layers.MaxPooling2D(2, strides=2, data_format='channels_first')(x)

    x   = layers.Conv2D(64, 3, padding='same', data_format='channels_first')(x)
    x   = layers.BatchNormalization(axis=1)(x); x = layers.LeakyReLU(0.1)(x)
    x   = layers.MaxPooling2D(2, strides=2, data_format='channels_first')(x)

    for f in [128, 256]:
        x = layers.Conv2D(f, 3, padding='same', data_format='channels_first')(x)
        x = layers.BatchNormalization(axis=1)(x); x = layers.LeakyReLU(0.1)(x)
    x   = layers.MaxPooling2D(2, strides=2, data_format='channels_first')(x)

    x   = layers.Conv2D(256, 3, padding='same', data_format='channels_first')(x)
    x   = layers.BatchNormalization(axis=1)(x); x = layers.LeakyReLU(0.1)(x)
    x   = layers.MaxPooling2D(2, strides=2, data_format='channels_first')(x)

    x   = layers.GlobalAveragePooling2D(data_format='channels_first')(x)
    x   = layers.Dense(512, activation='relu')(x)
    x   = layers.Dropout(0.5)(x)

    num_out = grid_s * grid_s * (num_boxes * 5 + num_classes)
    out     = layers.Dense(num_out, activation='sigmoid')(x)
    out     = layers.Reshape((grid_s, grid_s, num_boxes * 5 + num_classes))(out)
    return keras.Model(inp, out, name="YOLO_v1")

def yolo_loss(y_true, y_pred):
    obj_mask   = y_true[..., 0]
    noobj_mask = 1.0 - obj_mask
    xy_loss = kops.sum(
        kops.sum(kops.square(y_true[..., 1:3] - y_pred[..., 1:3]), axis=-1) * obj_mask)
    wh_loss = kops.sum(
        kops.sum(kops.square(
            kops.sqrt(y_true[..., 3:5] + 1e-8) - kops.sqrt(kops.abs(y_pred[..., 3:5]) + 1e-8)
        ), axis=-1) * obj_mask)
    c_loss = (kops.sum(kops.square(obj_mask   - y_pred[..., 0]) * obj_mask) +
              0.5 * kops.sum(kops.square(noobj_mask - y_pred[..., 0]) * noobj_mask))
    cls_loss = kops.sum(
        kops.sum(kops.square(y_true[..., 5:] - y_pred[..., 5:]), axis=-1) * obj_mask)
    return 5.0 * (xy_loss + wh_loss) + c_loss + cls_loss


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="ANNDL Project — VSC Training")
    parser.add_argument("--data-dir", required=True,
                        help="Path to VOCtrainval_11-May-2012_2 folder")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save models and plots")
    args = parser.parse_args()

    voc_folder = os.path.join(args.data_dir, "VOCdevkit", "VOC2012")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("ANNDL PROJECT — FULL TRAINING PIPELINE")
    print("=" * 80)
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"Keras version   : {keras.__version__}")
    print(f"Keras backend   : {keras.backend.backend()}")
    print(f"VOC folder      : {voc_folder}")
    print(f"Output dir      : {output_dir}")
    print()

    t_start = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # PART 1: IMAGE CLASSIFICATION
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("PART 1: IMAGE CLASSIFICATION")
    print("=" * 80)

    image_classes = preprocess_classification_data(voc_folder)
    print(f"Total images: {len(image_classes)}")

    image_paths_cls, labels_cls = [], []
    for vals in image_classes.values():
        image_paths_cls.append(vals["path"])
        labels_cls.append(vals["classes"])

    mlb = MultiLabelBinarizer(classes=_VOC_LABELS)
    y_cls_encoded = mlb.fit_transform(labels_cls)
    print(f"Samples: {len(image_paths_cls)}, Label shape: {y_cls_encoded.shape}")

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths_cls, y_cls_encoded, test_size=0.2, random_state=42)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.25, random_state=42)
    print(f"Train {len(train_paths)} | Val {len(val_paths)} | Test {len(test_paths)}")

    train_loader = DataLoader(
        VOCClassificationDataset(train_paths, train_labels, augment=True),
        batch_size=_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(
        VOCClassificationDataset(val_paths, val_labels),
        batch_size=_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(
        VOCClassificationDataset(test_paths, test_labels),
        batch_size=_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # ── 1.1 Custom CNNs ──────────────────────────────────────────────────────
    def make_callbacks(ckpt_name):
        return [
            callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss'),
            callbacks.ModelCheckpoint(
                os.path.join(output_dir, ckpt_name),
                save_best_only=True, monitor='val_loss'),
        ]

    all_histories = {}

    # V1: Baseline
    clear_gpu()
    model_v1 = build_baseline_cnn()
    model_v1.compile(optimizer='adam', loss='binary_crossentropy',
                     metrics=['accuracy', keras.metrics.AUC(multi_label=True, name='auc')])
    model_v1.summary()
    print("\n=== Training Baseline CNN ===")
    hist_v1 = model_v1.fit(train_loader, validation_data=val_loader,
                           epochs=20, callbacks=make_callbacks('best_cnn_v1.keras'))
    all_histories['Baseline'] = hist_v1.history
    model_v1.evaluate(test_loader)
    del model_v1

    # V2: Regularised
    clear_gpu()
    model_v2 = build_regularised_cnn()
    model_v2.compile(optimizer='adam', loss='binary_crossentropy',
                     metrics=['accuracy', keras.metrics.AUC(multi_label=True, name='auc')])
    model_v2.summary()
    print("\n=== Training Regularised CNN ===")
    hist_v2 = model_v2.fit(train_loader, validation_data=val_loader,
                           epochs=20, callbacks=make_callbacks('best_cnn_v2.keras'))
    all_histories['+ BN+Dropout'] = hist_v2.history
    model_v2.evaluate(test_loader)
    del model_v2

    # V3: ResNet-like (128x128 variant)
    clear_gpu()
    v3_train = DataLoader(
        VOCClassificationDataset(train_paths, train_labels, img_size=128, augment=True),
        batch_size=_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    v3_val = DataLoader(
        VOCClassificationDataset(val_paths, val_labels, img_size=128),
        batch_size=_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    v3_test = DataLoader(
        VOCClassificationDataset(test_paths, test_labels, img_size=128),
        batch_size=_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    def build_resnet_v3():
        inp = keras.Input(shape=(3, 128, 128))
        x = layers.Conv2D(32, 3, padding='same', data_format='channels_first')(inp)
        x = layers.BatchNormalization(axis=1)(x); x = layers.ReLU()(x)
        for f in [32, 64, 128]:
            x = res_block(x, f)
            x = layers.MaxPooling2D(data_format='channels_first')(x)
        x = layers.GlobalAveragePooling2D(data_format='channels_first')(x)
        x = layers.Dropout(0.5)(x)
        out = layers.Dense(NUM_CLASSES, activation='sigmoid')(x)
        return keras.Model(inp, out, name="ResNet_Like")

    model_v3 = build_resnet_v3()
    model_v3.compile(optimizer='adam', loss='binary_crossentropy',
                     metrics=['accuracy', keras.metrics.AUC(multi_label=True, name='auc')])
    model_v3.summary()
    print("\n=== Training ResNet-like CNN (128x128) ===")
    hist_v3 = model_v3.fit(v3_train, validation_data=v3_val,
                           epochs=20, callbacks=make_callbacks('best_cnn_v3.keras'))
    all_histories['ResNet-like'] = hist_v3.history
    model_v3.evaluate(v3_test)
    del model_v3, v3_train, v3_val, v3_test

    # V3b: ResNet-like (180x180)
    clear_gpu()
    model_v3b = build_resnet_like()
    model_v3b.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy', keras.metrics.AUC(multi_label=True, name='auc')])
    model_v3b.summary()
    print("\n=== Training ResNet-like CNN (180x180) ===")
    hist_v3b = model_v3b.fit(train_loader, validation_data=val_loader,
                             epochs=20, callbacks=make_callbacks('best_cnn_v3.keras'))
    all_histories['ResNet-like'] = hist_v3b.history
    model_v3b.evaluate(test_loader)
    del model_v3b
    clear_gpu()

    # ── Plot training curves ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, hist in all_histories.items():
        axes[0].plot(hist['val_loss'], label=label)
        axes[1].plot(hist['val_auc'],  label=label)
    axes[0].set(title='Validation Loss', xlabel='Epoch', ylabel='BCE Loss'); axes[0].legend()
    axes[1].set(title='Validation AUC',  xlabel='Epoch', ylabel='AUC');      axes[1].legend()
    plt.tight_layout()
    save_fig(fig, output_dir, "1_classification_curves.png")
    print("All 3 CNN iterations completed.\n")

    # ── 1.2 Transfer Learning with Xception ──────────────────────────────────
    print("--- 1.2 Transfer Learning with Xception ---")
    clear_gpu()

    XC_SIZE = 150
    xc_train = DataLoader(
        VOCDatasetCL(train_paths, train_labels, img_size=XC_SIZE, augment=True),
        batch_size=XC_BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    xc_val = DataLoader(
        VOCDatasetCL(val_paths, val_labels, img_size=XC_SIZE),
        batch_size=XC_BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    xc_test = DataLoader(
        VOCDatasetCL(test_paths, test_labels, img_size=XC_SIZE),
        batch_size=XC_BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    from keras.applications import Xception
    base = Xception(weights='imagenet', include_top=False,
                    input_shape=(XC_SIZE, XC_SIZE, 3))
    base.trainable = False

    def xc_head(dropout=0.3, hidden=None):
        inp  = keras.Input(shape=(XC_SIZE, XC_SIZE, 3))
        x    = base(inp, training=False)
        x    = layers.GlobalAveragePooling2D()(x)
        x    = layers.Dropout(dropout)(x)
        if hidden:
            x = layers.Dense(hidden, activation='relu')(x)
            x = layers.Dropout(dropout)(x)
        out  = layers.Dense(NUM_CLASSES, activation='sigmoid')(x)
        return keras.Model(inp, out)

    cb_xc = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    ]

    xc_histories = {}
    best_xc_model = None
    for name, dropout, hidden in [("Xc-A1 (simple)", 0.3, None),
                                   ("Xc-A2 (+dense512)", 0.4, 512),
                                   ("Xc-A3 (+dense256)", 0.5, 256)]:
        clear_gpu()
        m = xc_head(dropout=dropout, hidden=hidden)
        m.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.AUC(multi_label=True, name='auc')])
        print(f"\n=== {name} ===")
        h = m.fit(xc_train, validation_data=xc_val, epochs=10, callbacks=cb_xc)
        xc_histories[name] = h.history
        print("Test:"); m.evaluate(xc_test)
        if name == "Xc-A1 (simple)":
            best_xc_model = m
        else:
            del m

    # ── Fine-tune last Xception block ────────────────────────────────────────
    clear_gpu()
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

    best_xc_model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
                          loss='binary_crossentropy',
                          metrics=['accuracy', keras.metrics.AUC(multi_label=True, name='auc')])

    print("\n=== Fine-tuning last Xception block ===")
    hist_ft = best_xc_model.fit(xc_train, validation_data=xc_val, epochs=10,
                        callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                                   callbacks.ModelCheckpoint(
                                       os.path.join(output_dir, 'best_xception.keras'),
                                       save_best_only=True)])
    print("Test:"); best_xc_model.evaluate(xc_test)
    del best_xc_model, base
    clear_gpu()

    t_part1 = time.time()
    print(f"\n✓ Part 1 completed in {(t_part1 - t_start)/60:.1f} minutes\n")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 2: IMAGE SEGMENTATION
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("PART 2: IMAGE SEGMENTATION")
    print("=" * 80)

    image_paths_seg, annotation_paths = get_segmentation_paths(voc_folder)
    print(f"Segmentation samples: {len(image_paths_seg)}")

    X_train, X_rest, y_train_seg, y_rest = train_test_split(
        image_paths_seg, annotation_paths, train_size=0.6, shuffle=True, random_state=2022)
    X_val, X_test, y_val_seg, y_test_seg = train_test_split(
        X_rest, y_rest, train_size=0.6, shuffle=True, random_state=2022)
    print(f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")

    seg_train_loader = DataLoader(
        VOCSegDataset(X_train, y_train_seg, augment=True),
        batch_size=16, shuffle=True, num_workers=NUM_WORKERS)
    seg_val_loader = DataLoader(
        VOCSegDataset(X_val, y_val_seg),
        batch_size=16, shuffle=False, num_workers=NUM_WORKERS)
    seg_test_loader = DataLoader(
        VOCSegDataset(X_test, y_test_seg),
        batch_size=16, shuffle=False, num_workers=NUM_WORKERS)

    model_unet = build_unet(SEG_SIZE)
    model_unet.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', dice_coef]
    )
    model_unet.summary()

    cb_seg = [
        callbacks.EarlyStopping(patience=7, restore_best_weights=True,
                                monitor='val_dice_coef', mode='max'),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=4,
                                    monitor='val_dice_coef', mode='max'),
        callbacks.ModelCheckpoint(os.path.join(output_dir, 'best_unet.keras'),
                                  save_best_only=True,
                                  monitor='val_dice_coef', mode='max'),
    ]

    print("=== Training U-Net ===")
    hist_unet = model_unet.fit(seg_train_loader,
                                validation_data=seg_val_loader,
                                epochs=30, callbacks=cb_seg)
    print("\nTest evaluation:")
    model_unet.evaluate(seg_test_loader)

    # ── Save segmentation predictions plot ───────────────────────────────────
    imgs_t, masks_t = next(iter(seg_test_loader))
    preds = model_unet.predict(imgs_t[:4])

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for i in range(4):
        img  = imgs_t[i].permute(1,2,0).numpy()
        gt   = masks_t[i].permute(1,2,0).numpy().squeeze()
        pr   = preds[i, 0] if preds.ndim == 4 else preds[i].squeeze()
        axes[0,i].imshow(img);           axes[0,i].set_title('Image');      axes[0,i].axis('off')
        axes[1,i].imshow(gt,  cmap='gray'); axes[1,i].set_title('GT mask'); axes[1,i].axis('off')
        axes[2,i].imshow(pr>0.5, cmap='gray'); axes[2,i].set_title('Predicted'); axes[2,i].axis('off')
    plt.tight_layout()
    save_fig(fig, output_dir, "2_segmentation_predictions.png")

    del model_unet
    clear_gpu()

    t_part2 = time.time()
    print(f"\n✓ Part 2 completed in {(t_part2 - t_part1)/60:.1f} minutes\n")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 3: OBJECT DETECTION
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("PART 3: OBJECT DETECTION")
    print("=" * 80)

    annotations = preprocess_detection_data(voc_folder)
    print(f"Detection samples: {len(annotations)}")

    img_paths_det, Y_det = [], []
    for data in annotations.values():
        if len(data["boxes"]) > 0:
            img_paths_det.append(data["path"])
            Y_det.append(encode_annotation(data["boxes"], data["labels"]))
    Y_det = np.array(Y_det, dtype=np.float32)
    print(f"Encoded samples: {len(img_paths_det)}, target shape: {Y_det[0].shape}")

    det_img_train, det_img_test, det_y_train, det_y_test = train_test_split(
        img_paths_det, Y_det, test_size=0.2, random_state=42)

    det_loader_train = DataLoader(
        VOCDetDataset(det_img_train, det_y_train),
        batch_size=_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    det_loader_test = DataLoader(
        VOCDetDataset(det_img_test, det_y_test),
        batch_size=_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    clear_gpu()
    model_yolo = build_yolo(GRID_S, 1, C_DET, DET_SIZE)
    model_yolo.summary()

    model_yolo.compile(optimizer=keras.optimizers.Adam(1e-4), loss=yolo_loss)

    print("=== Training YOLO-like detector ===")
    cb_det = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        callbacks.ModelCheckpoint(os.path.join(output_dir, 'best_yolo.keras'),
                                  save_best_only=True),
    ]
    hist_yolo = model_yolo.fit(det_loader_train, epochs=20, callbacks=cb_det)

    # ── Save detection predictions plot ──────────────────────────────────────
    from matplotlib.patches import Rectangle
    from matplotlib.colors import hsv_to_rgb

    color_map = {}
    def label_to_color(label):
        if label not in color_map:
            h, s, v = (len(color_map) * 0.618) % 1, 0.5, 0.9
            color_map[label] = hsv_to_rgb((h, s, v))
        return color_map[label]

    def draw_box(ax, box, text, color):
        x, y, w, h = box
        ax.add_patch(Rectangle((x, y), w, h, lw=2, ec=color, fc="none"))
        ax.text(x, y, text, c="white", size=9, va="bottom",
                bbox=dict(fc=color, pad=1, ec="none"))

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    xi_t, yi_t = next(iter(det_loader_test))
    preds_det  = model_yolo.predict(xi_t[:4])

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, ax in enumerate(axes):
        img = xi_t[i].permute(1,2,0).numpy()
        ax.imshow(img); ax.set(xlim=(0,1), ylim=(1,0), xticks=[], yticks=[])
        pred = preds_det[i]
        for gy in range(GRID_S):
            for gx in range(GRID_S):
                cell = pred[gy, gx]
                if cell[0] > 0.3:
                    cx = (gx + cell[1]) / GRID_S
                    cy = (gy + cell[2]) / GRID_S
                    w, h = cell[3], cell[4]
                    x0, y0 = cx - w/2, cy - h/2
                    cls_idx = int(np.argmax(cell[5:]))
                    lbl = idx_to_class.get(cls_idx, str(cls_idx))
                    draw_box(ax, [x0, y0, w, h], lbl, label_to_color(lbl))
    plt.tight_layout()
    save_fig(fig, output_dir, "3_detection_predictions.png")

    del model_yolo
    clear_gpu()

    t_end = time.time()
    print(f"\n✓ Part 3 completed in {(t_end - t_part2)/60:.1f} minutes")
    print(f"\n{'=' * 80}")
    print(f"ALL DONE — Total time: {(t_end - t_start)/60:.1f} minutes ({(t_end - t_start)/3600:.2f} hours)")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
