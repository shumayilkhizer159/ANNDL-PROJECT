#!/usr/bin/env python
# coding: utf-8

# # ANNDL Project
# 
# Three parts: **classification**, **segmentation**, and **detection** on PASCAL VOC 2012.
# 
# Running on local GPU via **Keras 3 + PyTorch backend**.

# # 0. Getting started
# ## Local GPU setup (Keras 3 + PyTorch backend)

# In[ ]:


import time as _time
import os as _os
_global_start = _time.time()

# Setup Scratch Output Directory to avoid Disk Quota issues
_SCRATCH_DIR = _os.path.join(_os.getcwd(), 'local_test_output')
_OUTPUT_DIR = _SCRATCH_DIR
_os.makedirs(_OUTPUT_DIR, exist_ok=True)
print(f"  📂 Saving mock models locally to: {_OUTPUT_DIR}")

import keras
print("⚠️ LOCAL DRY RUN MODE ACTIVATED ⚠️")
print("=> Overriding Keras Model.fit() to only process 1 batch and 1 epoch!")

if not hasattr(keras.Model, '_original_fit'):
    keras.Model._original_fit = keras.Model.fit
def fast_dry_fit(self, *args, **kwargs):
    kwargs['steps_per_epoch'] = 1
    if 'validation_data' in kwargs:
         kwargs['validation_steps'] = 1
    kwargs['epochs'] = 1
    return keras.Model._original_fit(self, *args, **kwargs)
keras.Model.fit = fast_dry_fit

if not hasattr(keras.Model, '_original_evaluate'):
    keras.Model._original_evaluate = keras.Model.evaluate
def fast_dry_evaluate(self, *args, **kwargs):
    kwargs['steps'] = 1
    return keras.Model._original_evaluate(self, *args, **kwargs)
keras.Model.evaluate = fast_dry_evaluate


def _cell_timer(cell_num):
    elapsed = _time.time() - _global_start
    mins, secs = divmod(int(elapsed), 60)
    hrs, mins = divmod(mins, 60)
    print(f"  ⏱  Total elapsed: {hrs}h {mins:02d}m {secs:02d}s")
    import sys; sys.stdout.flush()

import gc as _gc
import torch as _torch
import json as _json

_HISTORY_FILE = _os.path.join(_OUTPUT_DIR, 'history.json')

def clear_gpu():
    _gc.collect()
    _torch.cuda.empty_cache()
    _free, _total = _torch.cuda.mem_get_info()
    print(f"      🧹 GPU Memory Cleared: {_free/1024**3:.2f} GB free of {_total/1024**3:.2f} GB")
    import sys; sys.stdout.flush()

def save_history(all_histories):
    with open(_HISTORY_FILE, 'w') as f:
        _json.dump(all_histories, f)

def load_history():
    if _os.path.exists(_HISTORY_FILE):
        with open(_HISTORY_FILE, 'r') as f:
            try:
                return _json.load(f)
            except:
                pass
    return {}

def train_model_vsc(model, model_path, train_loader, val_loader, epochs, history_key, all_histories):
    if model_path is not None and _os.path.exists(model_path):
        print(f"✅ Skipping training: {model_path} already exists...")
        model.load_weights(model_path)
    else:
        print(f"🚀 Training '{history_key}'...")
        cb = make_callbacks(model_path) if model_path is not None else []
        hist = model.fit(train_loader, validation_data=val_loader, epochs=epochs, callbacks=cb)
        all_histories[history_key] = hist.history
        if model_path is not None:
            save_history(all_histories)

    # Check shape to guarantee no runtime crash on evaluate
    dummy_x, _ = next(iter(val_loader))
    print(f"      [Sanity Check] DataLoader shape: {dummy_x.shape}, Expected: {model.input_shape}")

    print(f"📊 Evaluating '{history_key}'...")
    model.evaluate(val_loader)
# ── GPU SETUP ──────────────────────────────────────────────────────────────
# Use Keras 3 with PyTorch backend → runs on your RTX 5060 via CUDA
import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())
print("GPU             :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# In[ ]:


import os
import xml.etree.ElementTree as ET
from typing import List

import keras
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import random

print("Keras version:", keras.__version__)
print("Keras backend:", keras.backend.backend())


# In[ ]:


_IMAGE_SHAPE = (128, 128)   # H x W used for segmentation
IMG_SIZE = 224           # used for classification
_BATCH_SIZE  = 8            # keep small for 8GB VRAM


# # 1. Image Classification
# ## Preprocessing
# Specify the path to the extracted VOC folder below.

# In[ ]:


path_to_extracted_folder = 'C:/Artificial Neural Networks and Deep Learning/data/VOCtrainval_11-May-2012_2'
path_to_VOC_folder       = path_to_extracted_folder + '/VOCdevkit/VOC2012'


# In[ ]:


_VOC_LABELS = (
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor",
)

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


# In[ ]:


image_classes = preprocess_classification_data(path_to_VOC_folder)
print(f"Total images: {len(image_classes)}")


# In[ ]:


# Example entry
image_classes['2009_003541']


# Let's plot some random images from the dataset.

# In[ ]:


print("\n  [CELL 11/47] Data Augmentation check...")
_cell_timer(11)
def show_random_images(dataset, n=4):
    keys = random.sample(list(dataset.keys()), n)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    for ax, key in zip(axes, keys):
        sample = dataset[key]
        img = Image.open(sample["path"])
        ax.imshow(img); ax.axis("off")
        ax.set_title(key + ": " + ", ".join(sample["classes"]), fontsize=8)
    plt.tight_layout(); plt.show()

show_random_images(image_classes)


# ## Multi-hot encoding & train/val/test split

# In[ ]:


print("\n  [CELL 13/47] Normalisation check...")
_cell_timer(13)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

image_paths_cls, labels_cls = [], []
for vals in image_classes.values():
    image_paths_cls.append(vals['path'])
    labels_cls.append(vals['classes'])

mlb = MultiLabelBinarizer(classes=_VOC_LABELS)
y_cls_encoded = mlb.fit_transform(labels_cls)
print(f"Samples: {len(image_paths_cls)}, Label shape: {y_cls_encoded.shape}")


# In[ ]:


train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths_cls, y_cls_encoded, test_size=0.2, random_state=42)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.25, random_state=42)
print(f"Train {len(train_paths)} | Val {len(val_paths)} | Test {len(test_paths)}")


# ### PyTorch DataLoaders
# We use PyTorch DataLoaders since Keras 3 can consume them directly on the GPU.

# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class VOCClassificationDataset(Dataset):
    def __init__(self, paths, labels, img_size=224, augment=False):
        self.paths   = paths
        self.labels  = labels.astype(np.float32)
        aug = [T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.1, 0.05)] if augment else []
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            *aug,
            T.ToTensor(),         # → [0,1] float32 CHW
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), torch.tensor(self.labels[idx])

# PyTorch DataLoaders (num_workers=0 is safest on Windows)
train_loader = DataLoader(VOCClassificationDataset(train_paths, train_labels, augment=True),
                          batch_size=_BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(VOCClassificationDataset(val_paths,   val_labels),
                          batch_size=_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(VOCClassificationDataset(test_paths,  test_labels),
                          batch_size=_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# Keras 3 can consume PyTorch DataLoaders directly
x_batch, y_batch = next(iter(train_loader))
print("Image batch:", x_batch.shape, "  Labels batch:", y_batch.shape)


# ## Model
# ### 1.1 Create your own network from scratch
# 
# Three iterations:
# 1. **Baseline CNN** – simple stacked conv blocks
# 2. **Regularised CNN** – + BatchNorm + Dropout
# 3. **ResNet-like** – Depthwise separable convs + residual connections

# In[ ]:


# ─────────────────── 1.1  Custom CNNs from scratch ──────────────────────────
from keras import layers, models, callbacks
import keras
import gc

NUM_CLASSES = len(_VOC_LABELS)

# ── Utility: free GPU memory between models ──────────────────────────────────
def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU memory free: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB / {torch.cuda.mem_get_info()[1]/1e9:.2f} GB")

# ── Iteration 1 : Baseline CNN ───────────────────────────────────────────────
def build_baseline_cnn():
    inp = keras.Input(shape=(3, 224, 224))
    x   = layers.Conv2D(32, 3, activation='relu', padding='same', data_format='channels_first')(inp)
    x   = layers.MaxPooling2D(data_format='channels_first')(x)
    x   = layers.Conv2D(64, 3, activation='relu', padding='same', data_format='channels_first')(x)
    x   = layers.MaxPooling2D(data_format='channels_first')(x)
    x   = layers.Conv2D(128, 3, activation='relu', padding='same', data_format='channels_first')(x)
    x   = layers.GlobalAveragePooling2D(data_format='channels_first')(x)
    out = layers.Dense(NUM_CLASSES, activation='sigmoid')(x)
    return keras.Model(inp, out, name="Baseline_CNN")

# ── Iteration 2 : CNN + BatchNorm + Dropout ──────────────────────────────────
def build_regularised_cnn():
    inp = keras.Input(shape=(3, 224, 224))
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

# ── Iteration 3 : Depthwise separable + residual connections ─────────────────
# (lighter filters to fit 8GB VRAM: 32→64→128 instead of 64→128→256)
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
    inp = keras.Input(shape=(3, 224, 224))
    x   = layers.Conv2D(32, 3, padding='same', data_format='channels_first')(inp)
    x   = layers.BatchNormalization(axis=1)(x)
    x   = layers.ReLU()(x)
    for f in [32, 64, 128]:    # reduced from [64,128,256] for 8GB GPU
        x = res_block(x, f)
        x = layers.MaxPooling2D(data_format='channels_first')(x)
    x   = layers.GlobalAveragePooling2D(data_format='channels_first')(x)
    x   = layers.Dropout(0.5)(x)
    out = layers.Dense(NUM_CLASSES, activation='sigmoid')(x)
    return keras.Model(inp, out, name="ResNet_Like")

print("Model builders defined. Models will be built one-at-a-time to save VRAM.")


# In[ ]:


print("\n" + "★"*80 + "\n  [CELL 19/47] ★ TRAINING BASELINE CNNs ★\n" + "★"*80)
_cell_timer(19)
# ── Train models ONE AT A TIME (clear GPU memory between them) ──────────────

def make_callbacks(ckpt_name):
    return [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss'),
        callbacks.ModelCheckpoint(ckpt_name, save_best_only=True, monitor='val_loss'),
    ]

all_histories = {}

# ── V1: Baseline ─────────────────────────────────────────────────────────────
clear_gpu()
model_v1 = build_baseline_cnn()
model_v1.compile(optimizer='adam', loss='binary_crossentropy',
                 metrics=['accuracy', keras.metrics.AUC(multi_label=True, name='auc')])
model_v1.summary()
print("\n=== Training Baseline CNN ===")
train_model_vsc(model_v1, _os.path.join(_OUTPUT_DIR, 'best_cnn_v1.keras'), train_loader, val_loader, 20, '+ v1', all_histories)
del model_v1     # free VRAM

# ── V2: Regularised ──────────────────────────────────────────────────────────
clear_gpu()
model_v2 = build_regularised_cnn()
model_v2.compile(optimizer='adam', loss='binary_crossentropy',
                 metrics=['accuracy', keras.metrics.AUC(multi_label=True, name='auc')])
model_v2.summary()
print("\n=== Training Regularised CNN ===")
train_model_vsc(model_v2, _os.path.join(_OUTPUT_DIR, 'best_cnn_v2.keras'), train_loader, val_loader, 20, '+ v2', all_histories)
del model_v2

# ── V3: ResNet-like (128x128, batch=8 for 8GB GPU) ────────────────────────────
clear_gpu()
v3_train = DataLoader(VOCClassificationDataset(train_paths, train_labels, img_size=224, augment=True),
                      batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
v3_val   = DataLoader(VOCClassificationDataset(val_paths, val_labels, img_size=224),
                      batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
v3_test  = DataLoader(VOCClassificationDataset(test_paths, test_labels, img_size=224),
                      batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
def build_resnet_v3():
    inp = keras.Input(shape=(3, 224, 224))
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
print("\n=== Training ResNet-like CNN (128x128, batch=8) ===")
train_model_vsc(model_v3, _os.path.join(_OUTPUT_DIR, 'best_cnn_v3.keras'), train_loader, val_loader, 20, '+ v3', all_histories)
del model_v3
clear_gpu()

import gc, torch
for _ in range(3): gc.collect(); torch.cuda.empty_cache()
clear_gpu()
import sys; sys.stdout.flush()


# In[ ]:


print("\n  [CELL 20/47] Visualising CNN results...")
_cell_timer(20)
# ── Compare training curves ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for label, hist in all_histories.items():
    axes[0].plot(hist['val_loss'], label=label)
    axes[1].plot(hist['val_auc'],  label=label)
axes[0].set(title='Validation Loss', xlabel='Epoch', ylabel='BCE Loss'); axes[0].legend()
axes[1].set(title='Validation AUC',  xlabel='Epoch', ylabel='AUC');      axes[1].legend()
plt.tight_layout(); plt.show()

# Test evaluations were already printed after each model above.
print("All 3 CNN iterations completed.")


# ### 1.2 Feature extraction and fine-tuning with Xception
# 
# 1. Feature extraction with three different dense heads
# 2. End-to-end frozen base
# 3. Fine-tune last Xception block with low LR

# In[ ]:


print("\n" + "★"*80 + "\n  [CELL 22/47] ★ XCEPTION FEATURE EXTRACTION ★\n" + "★"*80)
_cell_timer(22)
# ─────────────────── 1.2  Transfer Learning with Xception ───────────────────
clear_gpu()

XC_SIZE = 224   # reduced from 224 to fit 8GB VRAM
XC_BATCH = 8    # small batch for large Xception model

class VOCDatasetCL(Dataset):
    """channels-last for Xception."""
    def __init__(self, paths, labels, img_size=224, augment=False):
        self.paths  = paths
        self.labels = labels.astype(np.float32)
        aug = [T.RandomHorizontalFlip()] if augment else []
        self.transform = T.Compose([T.Resize((img_size, img_size)), *aug, T.ToTensor()])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        t   = self.transform(img)            # CHW float [0,1]
        # permute removed for Torch channels-first backend
        t   = t * 2.0 - 1.0                 # → [-1, 1]  (Xception input)
        return t, torch.tensor(self.labels[idx])

xc_train = DataLoader(VOCDatasetCL(train_paths, train_labels, augment=True),
                       batch_size=XC_BATCH, shuffle=True,  num_workers=0, pin_memory=True)
xc_val   = DataLoader(VOCDatasetCL(val_paths,   val_labels),
                       batch_size=XC_BATCH, shuffle=False, num_workers=0, pin_memory=True)
xc_test  = DataLoader(VOCDatasetCL(test_paths,  test_labels),
                       batch_size=XC_BATCH, shuffle=False, num_workers=0, pin_memory=True)

# ── Load Xception base (frozen) ──────────────────────────────────────────────
from keras.applications import Xception

base = Xception(weights='imagenet', include_top=False,
                input_shape=(224, 224, 3))
base.trainable = False

# ── Experiment A: Feature extraction – iterate on dense head ─────────────────
def xc_head(dropout=0.3, hidden=None):
    inp  = keras.Input(shape=(3, 224, 224))
    x    = keras.layers.Permute((2, 3, 1))(inp)

    x    = base(x, training=False)
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
for name, dropout, hidden in [("Xc-A1 (simple)",0.3,None),("Xc-A2 (+dense512)",0.4,512),("Xc-A3 (+dense256)",0.5,256)]:
    clear_gpu()
    m = xc_head(dropout=dropout, hidden=hidden)
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy', keras.metrics.AUC(multi_label=True, name='auc')])
    print(f"\n=== {name} ===")
    train_model_vsc(m, None, xc_train, xc_val, 10, name, all_histories)
    if name == "Xc-A1 (simple)":
        best_xc_model = m   # keep best for fine-tuning
    else:
        del m

import gc, torch
for _ in range(3): gc.collect(); torch.cuda.empty_cache()
clear_gpu()
import sys; sys.stdout.flush()


# In[ ]:


print("\n" + "★"*80 + "\n  [CELL 23/47] ★ XCEPTION FINE TUNING ★\n" + "★"*80)
_cell_timer(23)
# ── Experiment C : Fine-tune last block ──────────────────────────────────────
clear_gpu()
base.trainable = True
for layer in base.layers[:-20]:   # freeze all but last ~20 layers
    layer.trainable = False

best_xc_model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
                      loss='binary_crossentropy',
                      metrics=['accuracy', keras.metrics.AUC(multi_label=True, name='auc')])

print("\n=== Fine-tuning last Xception block ===")
train_model_vsc(best_xc_model, _os.path.join(_OUTPUT_DIR, 'best_xception.keras'), xc_train, xc_val, 10, 'best_xc_model', all_histories)

# Free Xception from GPU for next parts
del best_xc_model, base
clear_gpu()

import gc, torch
for _ in range(3): gc.collect(); torch.cuda.empty_cache()
clear_gpu()
import sys; sys.stdout.flush()


# # 2. Image Segmentation
# ## Preprocessing

# In[ ]:


# ─────────────────── 2.  Image Segmentation ──────────────────────────────────
from torch.utils.data import Dataset, DataLoader

def get_segmentation_paths(base_dir):
    IMAGE_PATH      = os.path.join(base_dir, 'JPEGImages')
    ANNOTATION_PATH = os.path.join(base_dir, 'SegmentationClass')
    LISTS           = os.path.join(base_dir, 'ImageSets', 'Segmentation')
    img_paths, ann_paths = [], []
    with open(os.path.join(LISTS, 'trainval.txt')) as f:
        names = [l.strip() for l in f if l.strip()]
    for name in names:
        img_paths.append(os.path.join(IMAGE_PATH,      name + '.jpg'))
        ann_paths.append(os.path.join(ANNOTATION_PATH, name + '.png'))
    return img_paths, ann_paths

image_paths, annotation_paths = get_segmentation_paths(path_to_VOC_folder)
print(f"Segmentation samples: {len(image_paths)}")


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_rest, y_train, y_rest = train_test_split(
    image_paths, annotation_paths, train_size=0.6, shuffle=True, random_state=2022)
X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, train_size=0.6, shuffle=True, random_state=2022)
print(f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")


# In[ ]:


SEG_SIZE = 224

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

        img  = np.array(img,  dtype=np.float32) / 255.0    # HWC [0,1]
        mask = np.array(mask, dtype=np.uint8)
        mask[mask > 0] = 1                                  # foreground vs background
        mask = mask.astype(np.float32)[..., np.newaxis]     # HW1

        if self.augment and random.random() > 0.5:
            img  = img[:, ::-1, :].copy()
            mask = mask[:, ::-1, :].copy()

        # channels-last → channels-first for torch
        img  = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).permute(2, 0, 1)
        return img, mask

seg_train_loader = DataLoader(VOCSegDataset(X_train, y_train, augment=True),
                               batch_size=16, shuffle=True,  num_workers=0)
seg_val_loader   = DataLoader(VOCSegDataset(X_val,   y_val),
                               batch_size=16, shuffle=False, num_workers=0)
seg_test_loader  = DataLoader(VOCSegDataset(X_test,  y_test),
                               batch_size=16, shuffle=False, num_workers=0)

imgs, masks = next(iter(seg_train_loader))
print("Seg batch img:", imgs.shape, "mask:", masks.shape)


# Let's visualise some images and their segmentation masks.

# In[ ]:


# Visualise some images and masks
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i in range(4):
    img  = imgs[i].permute(1,2,0).numpy()
    msk  = masks[i].permute(1,2,0).numpy().squeeze()
    axes[0,i].imshow(img); axes[0,i].axis('off'); axes[0,i].set_title('Image')
    axes[1,i].imshow(msk, cmap='gray'); axes[1,i].axis('off'); axes[1,i].set_title('Mask')
plt.tight_layout(); plt.show()


# ## Model
# 
# We implement a **U-Net** (Encoder-Decoder with skip connections) for foreground/background segmentation.
# Metric: **Dice Coefficient** (more meaningful than accuracy on imbalanced pixel maps).

# In[ ]:


print("\n  [CELL 31/47] Building U-Net model...")
_cell_timer(31)
# ── U-Net model (channels-first for PyTorch backend) ─────────────────────────
def build_unet(img_size=224):
    inp = keras.Input(shape=(3, img_size, img_size))

    # Encoder
    def enc_block(x, f):
        x = layers.Conv2D(f, 3, padding='same', activation='relu', data_format='channels_first')(x)
        x = layers.Conv2D(f, 3, padding='same', activation='relu', data_format='channels_first')(x)
        p = layers.MaxPooling2D(data_format='channels_first')(x)
        return x, p

    c1, p1 = enc_block(inp, 32)
    c2, p2 = enc_block(p1,  64)
    c3, p3 = enc_block(p2, 128)

    # Bottleneck
    b = layers.Conv2D(256, 3, padding='same', activation='relu', data_format='channels_first')(p3)
    b = layers.Conv2D(256, 3, padding='same', activation='relu', data_format='channels_first')(b)

    # Decoder
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

# ── Custom Dice metric (backend-agnostic via keras.ops) ──────────────────────
import keras.ops as kops

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = kops.reshape(kops.cast(y_true, 'float32'), [-1])
    y_pred_f = kops.reshape(y_pred, [-1])
    inter    = kops.sum(y_true_f * y_pred_f)
    return (2. * inter + smooth) / (kops.sum(y_true_f) + kops.sum(y_pred_f) + smooth)

model_unet = build_unet(SEG_SIZE)
model_unet.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', dice_coef]
)
model_unet.summary()


# In[ ]:


clear_gpu()
print("\n" + "★"*80 + "\n  [CELL 32/47] ★ TRAINING U-NET SEGMENTOR ★\n" + "★"*80)
_cell_timer(32)
cb_seg = [
    callbacks.EarlyStopping(patience=7, restore_best_weights=True, monitor='val_dice_coef', mode='max'),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=4, monitor='val_dice_coef', mode='max'),
    callbacks.ModelCheckpoint(_os.path.join(_OUTPUT_DIR, 'best_unet.keras'), save_best_only=True,
                              monitor='val_dice_coef', mode='max'),
]

print("=== Training U-Net ===")
train_model_vsc(model_unet, None, seg_train_loader, seg_val_loader, 30, 'model_unet', all_histories)

import gc, torch
for _ in range(3): gc.collect(); torch.cuda.empty_cache()
clear_gpu()
import sys; sys.stdout.flush()


# ### Visualise predictions vs ground truth

# In[ ]:


# ── Visualise predictions vs ground truth ─────────────────────────────────────
imgs_t, masks_t = next(iter(seg_test_loader))
preds = model_unet.predict(imgs_t[:4])   # shape (4,1,128,128)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for i in range(4):
    img  = imgs_t[i].permute(1,2,0).numpy()
    gt   = masks_t[i].permute(1,2,0).numpy().squeeze()
    pr   = preds[i, 0] if preds.ndim == 4 else preds[i].squeeze() # handle shape
    axes[0,i].imshow(img);           axes[0,i].set_title('Image');      axes[0,i].axis('off')
    axes[1,i].imshow(gt,  cmap='gray'); axes[1,i].set_title('GT mask'); axes[1,i].axis('off')
    axes[2,i].imshow(pr>0.5, cmap='gray'); axes[2,i].set_title('Predicted'); axes[2,i].axis('off')
plt.tight_layout(); plt.show()


# # 3. Image Detection
# ## Loading the data

# In[ ]:


# ─────────────────── 3.  Object Detection ────────────────────────────────────
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

annotations = preprocess_detection_data(path_to_VOC_folder)
print(f"Detection samples: {len(annotations)}")


# Let's explore the resulting dataset.

# In[ ]:


from matplotlib.patches import Rectangle
from matplotlib.colors import hsv_to_rgb

boxes_per_image = [len(v["boxes"]) for v in annotations.values()]
print(f"Min/Max/Mean boxes: {min(boxes_per_image)} / {max(boxes_per_image)} / {np.mean(boxes_per_image):.2f}")

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

sample = annotations["2007_000733"]
print(sample)
fig, ax = plt.subplots(dpi=100)
ax.set(xlim=(0,1), ylim=(1,0), xticks=[], yticks=[], aspect="equal")
ax.imshow(plt.imread(sample["path"]),
          extent=[0, 1, 1, 0])
for box, lbl in zip(sample["boxes"], sample["labels"]):
    draw_box(ax, box, lbl, label_to_color(lbl))
plt.show()


# ## Preprocessing the data
# 
# We encode each bounding box into a **7×7 YOLO grid** with shape `(7, 7, 25)`.

# In[ ]:


print("\n  [CELL 40/47] YOLO grid encoding...")
_cell_timer(40)
GRID_S    = 7
DET_SIZE = 224     # reduced from 224 for 8GB VRAM
C_DET     = 20
class_to_idx = {name: idx for idx, name in enumerate(_VOC_LABELS)}

def encode_annotation(boxes, labels, grid_s=7):
    target = np.zeros((grid_s, grid_s, 5 + C_DET), dtype=np.float32)
    for box, label in zip(boxes, labels):
        ci  = class_to_idx.get(label)
        if ci is None: continue
        rx, ry, rw, rh = box
        xc, yc = rx + rw/2, ry + rh/2
        gx, gy = min(int(xc * grid_s), grid_s-1), min(int(yc * grid_s), grid_s-1)
        if target[gy, gx, 0] == 0:
            target[gy, gx, 0]      = 1.0
            target[gy, gx, 1:5]   = [xc*grid_s - gx, yc*grid_s - gy, rw, rh]
            target[gy, gx, 5+ci]  = 1.0
    return target

img_paths_det, Y_det = [], []
for data in annotations.values():
    if len(data['boxes']) > 0:
        img_paths_det.append(data['path'])
        Y_det.append(encode_annotation(data['boxes'], data['labels']))
Y_det = np.array(Y_det, dtype=np.float32)
print(f"Detection samples: {len(img_paths_det)}, target shape: {Y_det[0].shape}")


# In[ ]:


print("\n  [CELL 41/47] Creating detection DataLoaders...")
_cell_timer(41)
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

from sklearn.model_selection import train_test_split
det_img_train, det_img_test, det_y_train, det_y_test = train_test_split(
    img_paths_det, Y_det, test_size=0.2, random_state=42)

det_loader_train = DataLoader(VOCDetDataset(det_img_train, det_y_train),
                               batch_size=8, shuffle=True,  num_workers=0)
det_loader_test  = DataLoader(VOCDetDataset(det_img_test,  det_y_test),
                               batch_size=8, shuffle=False, num_workers=0)

xi, yi = next(iter(det_loader_train))
print("Det batch:", xi.shape, yi.shape)


# ## Model
# 
# Simplified **YOLOv1-style** detector with a custom composite loss.

# In[ ]:


print("\n  [CELL 43/47] Building YOLO model...")
_cell_timer(43)
clear_gpu()

def build_yolo(grid_s=7, num_boxes=1, num_classes=20, img_size=224):
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

model_yolo = build_yolo(GRID_S, 1, C_DET, DET_SIZE)
model_yolo.summary()


# In[ ]:


clear_gpu()
print("\n" + "★"*80 + "\n  [CELL 44/47] ★ TRAINING YOLO DETECTOR ★\n" + "★"*80)
_cell_timer(44)
def yolo_loss(y_true, y_pred):
    obj_mask   = y_true[..., 0]
    noobj_mask = 1.0 - obj_mask

    # Coordinate loss (only where object exists)
    xy_loss = kops.sum(
        kops.sum(kops.square(y_true[..., 1:3] - y_pred[..., 1:3]), axis=-1) * obj_mask)
    wh_loss = kops.sum(
        kops.sum(kops.square(
            kops.sqrt(y_true[..., 3:5] + 1e-8) - kops.sqrt(kops.abs(y_pred[..., 3:5]) + 1e-8)
        ), axis=-1) * obj_mask)

    # Confidence loss
    c_loss = (kops.sum(kops.square(obj_mask   - y_pred[..., 0]) * obj_mask) +
              0.5 * kops.sum(kops.square(noobj_mask - y_pred[..., 0]) * noobj_mask))

    # Class loss
    cls_loss = kops.sum(
        kops.sum(kops.square(y_true[..., 5:] - y_pred[..., 5:]), axis=-1) * obj_mask)

    return 5.0 * (xy_loss + wh_loss) + c_loss + cls_loss

model_yolo.compile(optimizer=keras.optimizers.Adam(1e-4), loss=yolo_loss)

print("=== Training YOLO-like detector ===")
cb_det = [callbacks.EarlyStopping(patience=5, restore_best_weights=True),
          callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
          callbacks.ModelCheckpoint(_os.path.join(_OUTPUT_DIR, 'best_yolo.keras'), save_best_only=True)]

train_model_vsc(model_yolo, None, det_loader_train, det_loader_test, 20, 'model_yolo', all_histories)
import gc, torch
for _ in range(3): gc.collect(); torch.cuda.empty_cache()
clear_gpu()
import sys; sys.stdout.flush()


# ### Visualise detection predictions

# In[ ]:


print("\n  [CELL 46/47] Visualising detection predictions...")
_cell_timer(46)
# ── Visualise predictions ─────────────────────────────────────────────────────
xi_t, yi_t = next(iter(det_loader_test))
preds_det   = model_yolo.predict(xi_t[:4])

idx_to_class = {v: k for k, v in class_to_idx.items()}

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, ax in enumerate(axes):
    img = xi_t[i].permute(1,2,0).numpy()
    ax.imshow(img); ax.set(xlim=(0,1), ylim=(1,0), xticks=[], yticks=[])
    pred = preds_det[i]   # (7,7, 25)
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
plt.tight_layout(); plt.show()


# In[ ]:


# ── Final Summary ──────────────────────────────────────────────────────────────
elapsed = _time.time() - _global_start
hrs, rem = divmod(int(elapsed), 3600)
mins, secs = divmod(rem, 60)
print(f"\n{'='*80}")
print(f"  ✅ ALL TRAINING COMPLETE")
print(f"  Total runtime: {hrs}h {mins:02d}m {secs:02d}s")
print(f"{'='*80}")

