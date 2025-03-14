#\!/usr/bin/env python
import os
import re

def update_imports_in_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Define patterns to search for and their replacements
    patterns = [
        (r'^from (core|copyline|gadgets|gadgets_ext|mapping|utils|weighted|dragondrop|unit_disk_mapping) import', 
         r'from .\1 import'),
        (r'^from pathdecomposition import', 
         r'from .pathdecomposition import'),
        (r'^import (core|copyline|gadgets|gadgets_ext|mapping|utils|weighted|dragondrop|unit_disk_mapping)$', 
         r'import .\1'),
    ]
    
    # Apply all patterns
    modified = False
    for pattern, replacement in patterns:
        updated_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        if updated_content \!= content:
            content = updated_content
            modified = True
    
    # Write back if modified
    if modified:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Updated imports in {filepath}")
    
    return modified

def process_directory(directory):
    updated_files = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if update_imports_in_file(filepath):
                    updated_files += 1
    return updated_files

directory = "src"
if os.path.exists(directory):
    print(f"Processing {directory}...")
    updated = process_directory(directory)
    print(f"Updated {updated} files in {directory}")
