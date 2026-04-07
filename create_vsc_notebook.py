import json
import os
import re
import time

# Load the original notebook
original_nb_path = 'ANNDL2526_Project_Template.ipynb'
vsc_nb_path = 'ANNDL2526_Project_Template_vsc.ipynb'

with open(original_nb_path, 'r', encoding='utf-8') as f:
    vsc_nb = json.load(f)

# Progress marker mapping
progress_markers = {
    1:  'print("\\n  [CELL 1/47] Setting up environment...")',
    10: 'print("\\n  [CELL 10/47] Visualising initial data...")',
    11: 'print("\\n  [CELL 11/47] Data Augmentation check...")',
    12: 'print("\\n  [CELL 12/47] Creating DataLoaders...")',
    13: 'print("\\n  [CELL 13/47] Normalisation check...")',
    19: 'print("\\n" + "★"*80 + "\\n  [CELL 19/47] ★ TRAINING BASELINE CNNs ★\\n" + "★"*80)',
    20: 'print("\\n  [CELL 20/47] Visualising CNN results...")',
    22: 'print("\\n" + "★"*80 + "\\n  [CELL 22/47] ★ XCEPTION FEATURE EXTRACTION ★\\n" + "★"*80)',
    23: 'print("\\n" + "★"*80 + "\\n  [CELL 23/47] ★ XCEPTION FINE TUNING ★\\n" + "★"*80)',
    30: 'print("\\n  [CELL 30/47] Loading segmentation data...")',
    31: 'print("\\n  [CELL 31/47] Building U-Net model...")',
    32: 'print("\\n" + "★"*80 + "\\n  [CELL 32/47] ★ TRAINING U-NET SEGMENTOR ★\\n" + "★"*80)',
    40: 'print("\\n  [CELL 40/47] YOLO grid encoding...")',
    41: 'print("\\n  [CELL 41/47] Creating detection DataLoaders...")',
    43: 'print("\\n  [CELL 43/47] Building YOLO model...")',
    44: 'print("\\n" + "★"*80 + "\\n  [CELL 44/47] ★ TRAINING YOLO DETECTOR ★\\n" + "★"*80)',
    46: 'print("\\n  [CELL 46/47] Visualising detection predictions...")',
}

timer_code = """import time as _time
import os as _os
_global_start = _time.time()

# Setup Scratch Output Directory to avoid Disk Quota issues
_SCRATCH_DIR = _os.environ.get('VSC_SCRATCH', _os.path.expanduser('~'))
_OUTPUT_DIR = _os.path.join(_SCRATCH_DIR, 'anndl_models')
_os.makedirs(_OUTPUT_DIR, exist_ok=True)
print(f"  📂 Saving models to: {_OUTPUT_DIR}")

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
    # Also print memory status to logs
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
                return {}
    return {}

def train_model_vsc(model, model_path, train_loader, val_loader, epochs, history_key, all_histories):
    if _os.path.exists(model_path):
        print(f"✅ Skipping training: {model_path} already exists. Loading weights...")
        model.load_weights(model_path)
    else:
        print(f"🚀 Training '{history_key}'...")
        # Note: make_callbacks is assumed to be defined in your notebook
        hist = model.fit(train_loader, validation_data=val_loader, epochs=epochs, callbacks=make_callbacks(model_path))
        all_histories[history_key] = hist.history
        save_history(all_histories)
    
    # Always evaluate to ensure the current session has the result
    print(f"📊 Evaluating '{history_key}'...")
    model.evaluate(val_loader)
"""

modified_count = 0

for i, cell in enumerate(vsc_nb['cells']):
    if cell['cell_type'] != 'code':
        continue

    source = ''.join(cell.get('source', []))
    original = source

    # ── Cell 2: Add timer + Agg backend ──────────────────────────────
    if 'os.environ["KERAS_BACKEND"]' in source and 'import torch' in source:
        source = timer_code + source
        # Force Matplotlib to use Agg backend (non-interactive)
        source = source.replace('import matplotlib.pyplot as plt', 'import matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt')

    # ── Replace Data Paths ──────────────────────────────────────────
    # User confirmed VOC path is in leuven-data (vsc-hard-mounts)
    vsc_data_base = '/vsc-hard-mounts/leuven-data/375/vsc37509/ANNDL/data/VOCtrainval_11-May-2012_2'
    source = source.replace("input_dir = 'dataset'", f"input_dir = '{vsc_data_base}'")
    source = source.replace('input_dir = "dataset"', f'input_dir = "{vsc_data_base}"')
    
    # Fix the VOC folder path that was hardcoded as Windows path in Classification/Segmentation
    source = re.sub(r"path_to_extracted_folder\s*=\s*['\"]C:/.*?['\"]", f"path_to_extracted_folder = '{vsc_data_base}'", source)
    
    # Also catch general VOCdevkit paths just in case
    source = re.sub(r"[A-Z]:/.*?/VOCdevkit/VOC2012", f"{vsc_data_base}/VOCdevkit/VOC2012", source)
    source = source.replace("/scratch/leuven/375/vsc37509/ANNDL-PROJECT/dataset", vsc_data_base)
    source = source.replace("/data/leuven/375/vsc37509/ANNDL-PROJECT/dataset", vsc_data_base)
    
    # ── Memory Optimization: Unified img_size 224 (for A100/V100 32GB) ────────
    # Force all size variables to be consistent
    source = source.replace('IMG_SIZE     = 180', 'IMG_SIZE     = 224')
    source = source.replace('XC_SIZE = 150', 'XC_SIZE = 224')
    source = source.replace('SEG_SIZE = 128', 'SEG_SIZE = 224')
    source = source.replace('DET_SIZE = 128', 'DET_SIZE = 224')
    
    # Force data loaders to match (global replace)
    source = source.replace('img_size=128', 'img_size=224')
    source = source.replace('img_size=180', 'img_size=224')
    source = source.replace('img_size=150', 'img_size=224')
    source = source.replace('img_size=DET_SIZE', 'img_size=224')
    source = source.replace('img_size=XC_SIZE', 'img_size=224')
    source = source.replace('img_size=SEG_SIZE', 'img_size=224')

    # Force model input shapes to match (global replace for common hardcoded sizes)
    # Using regex to more reliably catch (3, 128, 128), (3, 150, 150), and (3, 180, 180)
    source = re.sub(r'shape=\(3, (128|150|180|XC_SIZE|SEG_SIZE|DET_SIZE), (128|150|180|XC_SIZE|SEG_SIZE|DET_SIZE)\)', 'shape=(3, 224, 224)', source)

    # ── Add progress markers ─────────────────────────────────────────
    if i in progress_markers:
        marker = progress_markers[i]
        timer_call = f"_cell_timer({i})"
        source = marker + '\n' + timer_call + '\n' + source

    # ── Redirect model saves to Scratch Output Directory ────────────
    source = re.sub(r"'(?![_/])([^']+?\.keras)'", r"_os.path.join(_OUTPUT_DIR, '\1')", source)
    source = re.sub(r"\"(?![_/])([^\"]+?\.keras)\"", r"_os.path.join(_OUTPUT_DIR, '\1')", source)

    # ── Cell 18: Load saved history instead of resetting ──────────────
    if i == 18:
        source = source.replace('all_histories = {}', 'all_histories = load_history()')

    # ── Cells with Fit logic: Inject train_model_vsc Helper ──────────
    if i in [19, 22, 23, 32, 44]:
        # Handle Cell 19 (multi-model: v1, v2, v3)
        if i == 19:
            for v in ['v1', 'v2', 'v3']:
                m_path = f"_os.path.join(_OUTPUT_DIR, 'best_cnn_{v}.keras')"
                fit_call = f"hist_{v} = model_{v}.fit("
                if fit_call in source:
                    # Find start and end of this model's specific block
                    start_ptr = source.find(fit_call)
                    end_marker = f"model_{v}.evaluate(test_loader)"
                    end_ptr = source.find(end_marker, start_ptr)
                    if start_ptr != -1 and end_ptr != -1:
                        end_ptr += len(end_marker)
                        repl = f"train_model_vsc(model_{v}, {m_path}, train_loader, val_loader, 20, '+ {v}', all_histories)"
                        source = source[:start_ptr] + repl + source[end_ptr:]

        # Handle single-model cells (22, 23, 32, 44)
        else:
            # We search for .fit and .evaluate and replace the span
            if ".fit(" in source:
                fit_start = source.find(".fit(")
                # Go back to start of line
                line_start = source.rfind("\n", 0, fit_start) + 1
                
                # Find ending mark (usually evaluate or end of cell)
                eval_mark = ".evaluate("
                eval_idx = source.find(eval_mark, fit_start)
                
                if eval_idx != -1:
                    line_end = source.find(")", eval_idx) + 1
                else:
                    line_end = len(source)

                # Determine model name and model path
                model_match = re.search(r"(\w+)\.fit\(", source)
                m_var = model_match.group(1) if model_match else "model"
                
                path_match = re.search(r"(_os\.path\.join\(_OUTPUT_DIR, '[^']+\.keras'\))", source)
                m_path_code = path_match.group(1) if path_match else "None"
                
                # Determine history key
                key_match = re.search(r"all_histories\[([^\]]+)\]", source)
                h_key_code = key_match.group(1) if key_match else "unknown"
                
                repl = f"train_model_vsc({m_var}, {m_path_code}, train_loader, val_loader, 20, {h_key_code}, all_histories)"
                source = source[:line_start] + repl + source[line_end:]

        # Prepend clear_gpu()
        if 'clear_gpu()' not in source:
            source = 'clear_gpu()\n' + source
        
        # Append manual cleanup
        cleanup = "\nimport gc, torch\nfor _ in range(3): gc.collect(); torch.cuda.empty_cache()\nclear_gpu()\nimport sys; sys.stdout.flush()\n"
        if cleanup.strip() not in source:
            source += cleanup

    # Clear old outputs
    cell['outputs'] = []
    cell['execution_count'] = None

    if source != original:
        cell['source'] = [line + '\n' for line in source.split('\n')]
        if cell['source']:
            cell['source'][-1] = cell['source'][-1].rstrip('\n')
        modified_count += 1

# Final Summary Cell
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

# Add cell IDs to all cells
import uuid
for cell in vsc_nb['cells']:
    if 'id' not in cell:
        cell['id'] = str(uuid.uuid4())

with open(vsc_nb_path, 'w', encoding='utf-8') as f:
    json.dump(vsc_nb, f, indent=1)

print(f"Successfully created {vsc_nb_path} ({modified_count} cells modified)")
