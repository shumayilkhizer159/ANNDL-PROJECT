"""
Update the VSC notebook with larger image sizes for V100 32GB.
"""
import json
import re
import copy

with open('ANNDL2526_Project_Template.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

vsc_nb = copy.deepcopy(nb)

# Progress markers
progress_markers = {
    2:  'print("\\n" + "="*80 + "\\n  [CELL 2/47] GPU Setup\\n" + "="*80)',
    3:  'print("\\n  [CELL 3/47] Importing libraries...")',
    4:  'print("\\n  [CELL 4/47] Setting constants...")',
    6:  'print("\\n  [CELL 6/47] Setting data path...")',
    7:  'print("\\n  [CELL 7/47] Defining VOC labels & preprocessing function...")',
    8:  'print("\\n" + "="*80 + "\\n  [CELL 8/47] Loading classification data...\\n" + "="*80)',
    9:  'print("\\n  [CELL 9/47] Example entry...")',
    11: 'print("\\n  [CELL 11/47] Showing random images...")',
    13: 'print("\\n  [CELL 13/47] Multi-hot encoding...")',
    14: 'print("\\n  [CELL 14/47] Train/Val/Test split...")',
    16: 'print("\\n  [CELL 16/47] Creating PyTorch DataLoaders...")',
    18: 'print("\\n" + "="*80 + "\\n  [CELL 18/47] Defining CNN model builders...\\n" + "="*80)',
    19: 'print("\\n" + "★"*80 + "\\n  [CELL 19/47] ★ TRAINING 3 CUSTOM CNNs ★\\n" + "★"*80)',
    20: 'print("\\n  [CELL 20/47] Plotting training curves...")',
    22: 'print("\\n" + "★"*80 + "\\n  [CELL 22/47] ★ TRAINING XCEPTION (feature extraction) ★\\n" + "★"*80)',
    23: 'print("\\n" + "★"*80 + "\\n  [CELL 23/47] ★ FINE-TUNING XCEPTION ★\\n" + "★"*80)',
    25: 'print("\\n" + "="*80 + "\\n  [CELL 25/47] Loading segmentation data...\\n" + "="*80)',
    26: 'print("\\n  [CELL 26/47] Segmentation train/val/test split...")',
    27: 'print("\\n  [CELL 27/47] Creating segmentation DataLoaders...")',
    29: 'print("\\n  [CELL 29/47] Visualising segmentation masks...")',
    31: 'print("\\n  [CELL 31/47] Building U-Net model...")',
    32: 'print("\\n" + "★"*80 + "\\n  [CELL 32/47] ★ TRAINING U-NET ★\\n" + "★"*80)',
    34: 'print("\\n  [CELL 34/47] Visualising segmentation predictions...")',
    36: 'print("\\n" + "="*80 + "\\n  [CELL 36/47] Loading detection data...\\n" + "="*80)',
    38: 'print("\\n  [CELL 38/47] Detection data statistics...")',
    40: 'print("\\n  [CELL 40/47] YOLO grid encoding...")',
    41: 'print("\\n  [CELL 41/47] Creating detection DataLoaders...")',
    43: 'print("\\n  [CELL 43/47] Building YOLO model...")',
    44: 'print("\\n" + "★"*80 + "\\n  [CELL 44/47] ★ TRAINING YOLO DETECTOR ★\\n" + "★"*80)',
    46: 'print("\\n  [CELL 46/47] Visualising detection predictions...")',
}

timer_code = """import time as _time
_global_start = _time.time()

def _cell_timer(cell_num):
    elapsed = _time.time() - _global_start
    mins, secs = divmod(int(elapsed), 60)
    hrs, mins = divmod(mins, 60)
    print(f"  ⏱  Total elapsed: {hrs}h {mins:02d}m {secs:02d}s")
    import sys; sys.stdout.flush()
"""

modified_count = 0

for i, cell in enumerate(vsc_nb['cells']):
    if cell['cell_type'] != 'code':
        continue

    source = ''.join(cell.get('source', []))
    original = source

    # ── Cell 2: Add timer + Agg backend ──────────────────────────────
    # Timer code MUST come before progress markers (which call _cell_timer)
    if 'os.environ["KERAS_BACKEND"]' in source and 'import torch' in source:
        source = timer_code + '\nprint("\\n" + "="*80 + "\\n  [CELL 2/47] GPU Setup\\n" + "="*80)\n_cell_timer(2)\n\n' + source
        source = source.replace(
            'import torch',
            'import matplotlib\nmatplotlib.use("Agg")  # non-interactive backend for HPC\n\nimport torch'
        )
        # Remove cell 2 from progress_markers since we handle it specially
        progress_markers.pop(2, None)

    # ── Cell 4: Increase batch size + image sizes ────────────────────
    if '_BATCH_SIZE  = 8' in source:
        source = source.replace('_BATCH_SIZE  = 8', '_BATCH_SIZE  = 64   # V100 32GB — matches original template')
        source = source.replace('_IMAGE_SHAPE = (128, 128)', '_IMAGE_SHAPE = (256, 256)  # V100 32GB (was 128 for 8GB GPU)')
        source = source.replace('IMG_SIZE     = 180', 'IMG_SIZE     = 224  # standard ImageNet size (was 180 for 8GB GPU)')

    # ── Cell 6: Change data path ─────────────────────────────────────
    if "path_to_extracted_folder = 'C:/Artificial Neural Networks and Deep Learning/data/VOCtrainval_11-May-2012_2'" in source:
        source = source.replace(
            "path_to_extracted_folder = 'C:/Artificial Neural Networks and Deep Learning/data/VOCtrainval_11-May-2012_2'",
            "import os\npath_to_extracted_folder = os.environ.get('DATA_DIR', os.path.expandvars('$VSC_DATA/ANNDL/data/VOCtrainval_11-May-2012_2'))"
        )

    # ── num_workers ──────────────────────────────────────────────────
    source = source.replace('num_workers=0', 'num_workers=4')

    # ── Fix default img_size in dataset class definitions ────────────
    # VOCClassificationDataset default: 180 → 224 (matches IMG_SIZE)
    source = source.replace('img_size=180, augment', 'img_size=224, augment')
    # VOCDatasetCL (Xception) default: 150 → 299 (matches XC_SIZE)
    source = source.replace('img_size=150, augment', 'img_size=299, augment')

    # ── XC_SIZE + XC_BATCH ───────────────────────────────────────────
    if 'XC_SIZE = 150' in source:
        source = source.replace('XC_SIZE = 150', 'XC_SIZE = 299  # Xception native resolution (was 150 for 8GB GPU)')
    if 'XC_BATCH = 8' in source:
        source = source.replace('XC_BATCH = 8', 'XC_BATCH = 16   # V100 32GB (smaller batch for large Xception @ 299x299)')

    # ── Xception input_shape to match new XC_SIZE ────────────────────
    if 'input_shape=(XC_SIZE, XC_SIZE, 3)' in source:
        pass  # Already uses XC_SIZE variable, no change needed

    # ── SEG_SIZE: 128 → 256 ──────────────────────────────────────────
    if 'SEG_SIZE = 128' in source:
        source = source.replace('SEG_SIZE = 128', 'SEG_SIZE = 256  # 4x more pixels (was 128 for 8GB GPU)')

    # ── DET_SIZE: 160 → 320 ──────────────────────────────────────────
    if 'DET_SIZE  = 160' in source:
        source = source.replace('DET_SIZE  = 160', 'DET_SIZE  = 320  # standard YOLO input (was 160 for 8GB GPU)')

    # ── Batch sizes in DataLoader calls ──────────────────────────────
    source = re.sub(r'batch_size=8(?!_)', 'batch_size=64', source)
    source = source.replace('batch_size=16', 'batch_size=64')

    # ── ResNet V3 variant: also use 224 instead of 128 ───────────────
    # The v3 variant in cell 19 creates loaders with img_size=128
    if 'img_size=128' in source and 'VOCClassificationDataset' in source:
        source = source.replace('img_size=128', 'img_size=224')
    # And the model input shape
    if "shape=(3, 128, 128)" in source and "build_resnet_v3" in source:
        source = source.replace("shape=(3, 128, 128)", "shape=(3, 224, 224)")

    # ── Add progress markers ─────────────────────────────────────────
    if i in progress_markers:
        marker = progress_markers[i]
        timer_call = f"_cell_timer({i})"
        source = marker + '\n' + timer_call + '\n' + source

    # ── Flush after training cells ───────────────────────────────────
    if i in [19, 22, 23, 32, 44]:
        source += '\nimport sys; sys.stdout.flush()'

    # Clear old outputs
    cell['outputs'] = []
    cell['execution_count'] = None

    if source != original:
        cell['source'] = [line + '\n' for line in source.split('\n')]
        if cell['source']:
            cell['source'][-1] = cell['source'][-1].rstrip('\n')
        modified_count += 1
        print(f"  Modified cell {i}")

final_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "final_summary",
    "metadata": {},
    "outputs": [],
    "source": [
        "# ── Final Summary ──────────────────────────────────────────────────────────────\n",
        "elapsed = _time.time() - _global_start\n",
        "hrs, rem = divmod(int(elapsed), 3600)\n",
        "mins, secs = divmod(rem, 60)\n",
        "print(f\"\\n{'='*80}\")\n",
        "print(f\"  ✅ ALL TRAINING COMPLETE\")\n",
        "print(f\"  Total runtime: {hrs}h {mins:02d}m {secs:02d}s\")\n",
        "print(f\"{'='*80}\")"
    ]
}
vsc_nb['cells'].append(final_cell)

# Add cell IDs to all cells (fixes nbformat MissingIDFieldWarning)
import uuid
for idx, cell in enumerate(vsc_nb['cells']):
    if 'id' not in cell:
        cell['id'] = str(uuid.uuid4())[:8]

# Ensure nbformat version supports cell IDs
vsc_nb['nbformat_minor'] = 5

with open('ANNDL2526_Project_Template_vsc.ipynb', 'w', encoding='utf-8') as f:
    json.dump(vsc_nb, f, indent=1, ensure_ascii=False)

print(f"\nDone! Modified {modified_count} cells.")
print("VSC notebook updated with larger image sizes and batch sizes.")
print("\nSize summary:")
print("  IMG_SIZE:  180 → 224  (classification CNNs)")
print("  XC_SIZE:   150 → 299  (Xception native)")
print("  SEG_SIZE:  128 → 256  (segmentation U-Net)")
print("  DET_SIZE:  160 → 320  (YOLO detection)")
print("  Batch:     8   → 64   (V100 32GB — matches original template)")
print("  XC_BATCH:  8   → 16   (Xception is big @ 299x299)")
