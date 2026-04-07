import json
import os
import re

def audit_final(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    results = []
    # Keywords to audit for channel and shape consistency
    keywords = ['Input(shape=', 'InputLayer(', 'permute(', 'XC_SIZE', 'SEG_SIZE', 'DET_SIZE']
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = "".join(cell.get('source', []))
        if any(k in source for k in keywords):
            # Check for any "Channels Last" (..., 3) hardcodings
            if re.search(r'shape=\([^)]*,\s*3\)', source):
                line = [l for l in source.split('\n') if 'shape=' in l and ', 3)' in l]
                results.append(f"❌ Error in Cell {i}: Found hardcoded Channels-Last shape: {line[0].strip()}")
            
            # Check for any remaining permute(1, 2, 0)
            if 'permute(1, 2, 0)' in source and '#' not in source.split('permute')[0]:
                results.append(f"❌ Error in Cell {i}: Found active Channels-Last permutation: permute(1, 2, 0)")

            results.append(f"✅ Cell {i} audited for Channel Sync.")
    
    return results

if __name__ == "__main__":
    audit_results = audit_final('ANNDL2526_Project_Template_vsc.ipynb')
    with open('audit_results_final.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(audit_results))
    print("Final Audit Complete. See audit_results_final.txt")
