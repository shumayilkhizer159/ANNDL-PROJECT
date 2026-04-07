import json
import os
import re

def audit_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    results = []
    keywords = ['.fit(', '.evaluate(', 'IMG_SIZE', 'XC_SIZE', 'SEG_SIZE', 'DET_SIZE', 'train_model_vsc']
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = "".join(cell.get('source', []))
        found = [k for k in keywords if k in source]
        
        if found:
            # Check indentation for train_model_vsc if in a loop
            if 'for ' in source and 'train_model_vsc' in source:
                loop_match = re.search(r'for .*?:', source)
                train_match = re.search(r'(\s+)train_model_vsc', source)
                if loop_match and train_match:
                    loop_indent = len(source[:loop_match.start()]) - len(source[:loop_match.start()].lstrip())
                    train_indent = len(train_match.group(1))
                    if train_indent <= loop_indent:
                        results.append(f"❌ Error in Cell {i}: train_model_vsc is NOT indented correctly inside the loop.")
            
            # Check resolutions
            for size_var in ['IMG_SIZE', 'XC_SIZE', 'SEG_SIZE', 'DET_SIZE']:
                if size_var in source and '= 224' not in source and 'if' not in source and 'def' not in source:
                    # Ignore comment-only mentions
                    line = [l for l in source.split('\n') if size_var in l and '=' in l and '#' not in l.split('=')[0]]
                    if line and '224' not in line[0]:
                        results.append(f"⚠️ Warning in Cell {i}: {size_var} might not be 224: {line[0].strip()}")

            results.append(f"✅ Cell {i} audited. Found: {found}")
    
    return results

if __name__ == "__main__":
    audit_results = audit_notebook('ANNDL2526_Project_Template_vsc.ipynb')
    with open('audit_results.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(audit_results))
    print("Audit Complete. See audit_results.txt")
