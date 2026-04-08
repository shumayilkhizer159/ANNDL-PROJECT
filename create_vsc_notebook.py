import json
import os
import re
import uuid

# Load the original notebook
original_nb_path = 'ANNDL2526_Project_Template.ipynb'
vsc_nb_path = 'ANNDL2526_Project_Template_vsc.ipynb'

with open(original_nb_path, 'r', encoding='utf-8') as f:
    vsc_nb = json.load(f)

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
"""

modified_count = 0
vsc_data_base = '/vsc-hard-mounts/leuven-data/375/vsc37509/ANNDL/data/VOCtrainval_11-May-2012_2'

for i, cell in enumerate(vsc_nb['cells']):
    if cell['cell_type'] != 'code':
        continue

    source = ''.join(cell.get('source', []))
    original = source

    # Inject Timer + Agg Backend
    if 'os.environ["KERAS_BACKEND"]' in source and 'import torch' in source:
        source = timer_code + source
        source = source.replace('import matplotlib.pyplot as plt', 'import matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt')

    # Path Normalization
    source = re.sub(r"input_dir\s*=\s*['\"].*?['\"]", f"input_dir = '{vsc_data_base}'", source)
    source = re.sub(r"path_to_extracted_folder\s*=\s*['\"].*?['\"]", f"path_to_extracted_folder = '{vsc_data_base}'", source)
    source = re.sub(r"[A-Z]:/.*?/VOCdevkit/VOC2012", f"{vsc_data_base}/VOCdevkit/VOC2012", source)
    
    # =========================================================================
    # UNIVERSAL SHAPE & CHANNEL NORMALIZATION
    # =========================================================================
    # 1. Force exact constant definitions to 224
    source = re.sub(r'\bIMG_SIZE\s*=\s*\d+', 'IMG_SIZE = 224', source)
    source = re.sub(r'\bXC_SIZE\s*=\s*\d+', 'XC_SIZE = 224', source)
    source = re.sub(r'\bSEG_SIZE\s*=\s*\d+', 'SEG_SIZE = 224', source)
    source = re.sub(r'\bDET_SIZE\s*=\s*\d+', 'DET_SIZE = 224', source)
    
    # 2. Force img_size keywords to 224
    source = re.sub(r'img_size\s*=\s*(128|150|160|180|IMG_SIZE|XC_SIZE|SEG_SIZE|DET_SIZE)', 'img_size=224', source)
    
    # 2.5 I/O Optimization: Massive speedup for Network Drives (NFS)
    # The template uses num_workers=0 for local windows, which causes agonizing I/O starvation on clusters.
    source = source.replace('num_workers=0', 'num_workers=16, pin_memory=True')
    # Clean up duplicate pin_memory if the template already had it
    source = re.sub(r'pin_memory=True\s*,\s*pin_memory=True', 'pin_memory=True', source)
    
    # 3. Universal shape replacement to (3, 224, 224) 
    # Match any shape=(X, Y, Z) or input_shape=(X, Y, Z)
    # Exclude the exact constructor for `keras.applications.Xception` if we can, 
    # but the safest way is to change ALL shapes to (3, 224, 224), then fix Xception specifically.
    
    # Replace (150, 150, 3) or (XC_SIZE, XC_SIZE, 3) etc to (3, 224, 224)
    source = re.sub(r'(?:input_)?shape\s*=\s*\(\s*(?:\d+|[A-Z_]+)\s*,\s*(?:\d+|[A-Z_]+)\s*,\s*3\s*\)', 'shape=(3, 224, 224)', source)
    source = re.sub(r'(?:input_)?shape\s*=\s*\(\s*3\s*,\s*(?:\d+|[A-Z_]+)\s*,\s*(?:\d+|[A-Z_]+)\s*\)', 'shape=(3, 224, 224)', source)
    
    # Fix the Xception constructor specifically so it doesn't crash on ImageNet check
    source = re.sub(r'(Xception\([\s\S]*?)(shape=\(3, 224, 224\))([\s\S]*?\))', r'\1input_shape=(224, 224, 3)\3', source)
    
    # 3.5 Fix the Xception Head Data Pipeline Layout
    # Xception fundamentally expects Channels_Last because we forced it in the constructor above, 
    # but the Torch dataloader yields Channels_First (3, 224, 224) via the Input layer.
    source = re.sub(r'(\s*x\s*=\s*)base\(inp,\s*training=False\)', r'\1keras.layers.Permute((2, 3, 1))(inp)\n\1base(x, training=False)\n\1keras.layers.Permute((3, 1, 2))(x)', source)
    
    # 4. Remove ALL permute(1, 2, 0) manual channel swaps
    source = re.sub(r't\s*=\s*t\.permute\(1,\s*2,\s*0\).*', '# permute removed for Torch channels-first backend', source)
    # =========================================================================

    # Progress Map
    if i in progress_markers:
        marker = progress_markers[i]
        timer_call = f"_cell_timer({i})"
        source = marker + '\n' + timer_call + '\n' + source

    # Save to Scratch output redirection
    source = re.sub(r"'(?![_/])([^']+?\.keras)'", r"_os.path.join(_OUTPUT_DIR, '\1')", source)
    source = re.sub(r"\"(?![_/])([^\"]+?\.keras)\"", r"_os.path.join(_OUTPUT_DIR, '\1')", source)

    # Load History (Cell 18)
    if source.strip() == 'all_histories = {}':
        source = 'all_histories = load_history()'

    # Train Helper Injection
    if '.fit(' in source and 'epochs=' in source and i not in [1, 10]:
        # Special logic for Cell 19 (multiple models)
        if 'model_v1.fit' in source:
            for v in ['v1', 'v2', 'v3']:
                m_path = f"_os.path.join(_OUTPUT_DIR, 'best_cnn_{v}.keras')"
                fit_call = f"model_{v}.fit("
                if fit_call in source:
                    start_ptr = source.find(fit_call)
                    # Go back to start of line to grab fit_call
                    line_start = source.rfind("\n", 0, start_ptr) + 1
                    end_marker = f"model_{v}.evaluate(test_loader)"
                    end_ptr = source.find(end_marker, start_ptr)
                    if start_ptr != -1 and end_ptr != -1:
                        # Grab all text line_start to end_ptr + end_marker len
                        end_idx = end_ptr + len(end_marker)
                        old_block = source[line_start:end_idx]
                        repl = f"train_model_vsc(model_{v}, {m_path}, train_loader, val_loader, 20, '+ {v}', all_histories)"
                        source = source.replace(old_block, repl)
        else:
            # Generalized single model replacement
            fit_starts = [m.start() for m in re.finditer(r'\w+\.fit\(', source)]
            for fit_start in reversed(fit_starts):
                line_start = source.rfind('\n', 0, fit_start)
                line_start = 0 if line_start == -1 else line_start + 1
                
                eval_idx = source.find('.evaluate(', fit_start)
                if eval_idx != -1:
                    line_end = source.find(')', eval_idx) + 1
                else:
                    line_end = len(source)
                
                old_block = source[line_start:line_end]
                indent = old_block[:len(old_block) - len(old_block.lstrip())]
                
                model_match = re.search(r'(\w+)\.fit\(', old_block)
                m_var = model_match.group(1) if model_match else 'model'
                
                path_match = re.search(r"(_os\.path\.join\(_OUTPUT_DIR, '[^']+\.keras'\))", old_block)
                m_path_code = path_match.group(1) if path_match else "None"
                
                # key extraction
                key_match = re.search(r"all_histories\[([^\]]+)\]", old_block)
                if key_match:
                    h_key_code = key_match.group(1)
                elif 'name' in old_block:
                    h_key_code = 'name'
                else:
                    h_key_code = f"'{m_var}'"
                
                # Check for dropout loops (e.g. model v2 experimental loops)
                epochs_match = re.search(r'epochs\s*=\s*(\d+|epochs)', old_block)
                ep_val = epochs_match.group(1) if epochs_match else '20'
                
                repl = f"{indent}train_model_vsc({m_var}, {m_path_code}, train_loader, val_loader, {ep_val}, {h_key_code}, all_histories)"
                source = source.replace(old_block, repl)

    # Prepend/Append Clear GPU
    if 'model.fit' in original or '.fit(' in original:
        if 'clear_gpu()' not in source:
            source = 'clear_gpu()\n' + source
        cleanup = "\nimport gc, torch\nfor _ in range(3): gc.collect(); torch.cuda.empty_cache()\nclear_gpu()\nimport sys; sys.stdout.flush()\n"
        if cleanup.strip() not in source:
            source += cleanup

    cell['outputs'] = []
    cell['execution_count'] = None

    if source != original:
        cell['source'] = [line + '\n' for line in source.split('\n')]
        if cell['source']:
            cell['source'][-1] = cell['source'][-1].rstrip('\n')
        modified_count += 1

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

for cell in vsc_nb['cells']:
    if 'id' not in cell:
        cell['id'] = str(uuid.uuid4())

with open(vsc_nb_path, 'w', encoding='utf-8') as f:
    json.dump(vsc_nb, f, indent=1)

print(f"Successfully created {vsc_nb_path} ({modified_count} cells modified)")
