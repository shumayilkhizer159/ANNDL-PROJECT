"""Verify the VSC notebook for common errors."""
import json
import re

with open('ANNDL2526_Project_Template_vsc.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"nbformat: {nb.get('nbformat')}.{nb.get('nbformat_minor')}")
print(f"Total cells: {len(nb['cells'])}")
print()

errors = []
warnings = []

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell.get('source', []))
    
    # Check cell has ID
    if 'id' not in cell:
        errors.append(f"Cell {i}: Missing 'id' field")
    
    # Check for hardcoded Windows paths
    if 'C:/' in source or 'C:\\' in source:
        errors.append(f"Cell {i}: Contains hardcoded Windows path")
    
    # Check for old img_size defaults that don't match
    if 'img_size=180' in source:
        errors.append(f"Cell {i}: Still has img_size=180 (should be 224)")
    if 'img_size=150' in source:
        errors.append(f"Cell {i}: Still has img_size=150 (should be 299)")
    
    # Check for old batch sizes
    if 'batch_size=8' in source and '_BATCH_SIZE' not in source.split('batch_size=8')[0][-20:]:
        warnings.append(f"Cell {i}: Has batch_size=8")
    
    # Check for num_workers=0
    if 'num_workers=0' in source:
        warnings.append(f"Cell {i}: Still has num_workers=0")
    
    # Check _BATCH_SIZE
    if '_BATCH_SIZE' in source and '= 8' in source and 'was' not in source:
        errors.append(f"Cell {i}: _BATCH_SIZE still set to 8")
    
    # Check IMG_SIZE
    if 'IMG_SIZE' in source and '= 180' in source and 'was' not in source:
        errors.append(f"Cell {i}: IMG_SIZE still set to 180")
    
    # Check SEG_SIZE
    if 'SEG_SIZE = 128' in source and 'was' not in source:
        errors.append(f"Cell {i}: SEG_SIZE still 128")
    
    # Check DET_SIZE
    if 'DET_SIZE  = 160' in source and 'was' not in source:
        errors.append(f"Cell {i}: DET_SIZE still 160")
    
    # Check XC_SIZE
    if 'XC_SIZE = 150' in source and 'was' not in source:
        errors.append(f"Cell {i}: XC_SIZE still 150")

    # Check _IMAGE_SHAPE
    if '_IMAGE_SHAPE = (128, 128)' in source and 'was' not in source:
        errors.append(f"Cell {i}: _IMAGE_SHAPE still (128, 128)")

    # Check for relative model paths (should use _OUTPUT_DIR)
    if '.keras' in source and '_OUTPUT_DIR' not in source:
        if 'import ' not in source and 'def ' not in source: # skip definitions
             errors.append(f"Cell {i}: Saving model to relative path (needs _OUTPUT_DIR)")

    # Check shape mismatches - model input vs data
    # Models use IMG_SIZE variable, so check that it's consistent
    if 'shape=(3, 128, 128)' in source and 'build_unet' not in source:
        errors.append(f"Cell {i}: Has hardcoded shape=(3, 128, 128) not for UNet")
    
    # Check _cell_timer is called only after it's defined
    if '_cell_timer' in source and 'def _cell_timer' not in source:
        # Find positions
        call_pos = source.find('_cell_timer(')
        def_pos = source.find('def _cell_timer')
        if def_pos == -1:
            # _cell_timer defined in cell 2, called elsewhere - OK if cell > 2
            pass  # will be defined by then
        elif call_pos < def_pos:
            errors.append(f"Cell {i}: _cell_timer called before defined")

# Check cell 2 specifically for timer ordering
cell2_src = ''.join(nb['cells'][2].get('source', []))
timer_def_pos = cell2_src.find('def _cell_timer')
timer_call_pos = cell2_src.find('_cell_timer(')
if timer_call_pos < timer_def_pos and timer_def_pos != -1:
    errors.append("Cell 2: _cell_timer() called before def _cell_timer()")
elif timer_def_pos == -1:
    errors.append("Cell 2: _cell_timer not defined!")

# Print key values found
print("=== KEY VALUES IN NOTEBOOK ===")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell.get('source', []))
    for pattern in ['_BATCH_SIZE', 'IMG_SIZE', '_IMAGE_SHAPE', 'SEG_SIZE', 'DET_SIZE', 'XC_SIZE', 'XC_BATCH']:
        match = re.search(rf'{pattern}\s*=\s*[^\n]+', source)
        if match and 'import' not in match.group() and 'def ' not in match.group():
            print(f"  Cell {i:3d}: {match.group().strip()}")

print()

# Print model input shapes
print("=== MODEL INPUT SHAPES ===")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell.get('source', []))
    for match in re.finditer(r'shape=\([^)]+\)', source):
        if '3,' in match.group() or ',3' in match.group():
            print(f"  Cell {i:3d}: {match.group()}")

print()

# Print DataLoader batch_size values
print("=== DATALOADER BATCH SIZES ===")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell.get('source', []))
    for match in re.finditer(r'batch_size=\d+', source):
        print(f"  Cell {i:3d}: {match.group()}")

print()

# Print dataset class default img_sizes
print("=== DATASET CLASS DEFAULTS ===")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell.get('source', []))
    for match in re.finditer(r'img_size=\d+', source):
        print(f"  Cell {i:3d}: {match.group()}")
    for match in re.finditer(r'size=\w+', source):
        if 'img_size' not in source[max(0,match.start()-4):match.start()]:
            print(f"  Cell {i:3d}: {match.group()}")

print()

if errors:
    print("❌ ERRORS FOUND:")
    for e in errors:
        print(f"  {e}")
else:
    print("✅ No errors found!")

if warnings:
    print("\n⚠️  WARNINGS:")
    for w in warnings:
        print(f"  {w}")

print()
