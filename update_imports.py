#!/usr/bin/env python
"""
Script to update imports after restructuring the project.
"""
import os
import re
import sys

def update_imports_in_file(filepath):
    """Update imports in a single file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Define patterns to search for and their replacements
    # Format: (pattern, replacement)
    patterns = [
        # For direct imports
        (r'^from (core|copyline|gadgets|gadgets_ext|mapping|utils|weighted|dragondrop|unit_disk_mapping) import', 
         r'from src.\1 import'),
        # For imports from pathdecomposition
        (r'^from pathdecomposition import', 
         r'from src.pathdecomposition import'),
        # For direct module imports
        (r'^import (core|copyline|gadgets|gadgets_ext|mapping|utils|weighted|dragondrop|unit_disk_mapping)$', 
         r'import src.\1'),
    ]
    
    # Apply all patterns
    modified = False
    for pattern, replacement in patterns:
        updated_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        if updated_content != content:
            content = updated_content
            modified = True
    
    # Write back if modified
    if modified:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Updated imports in {filepath}")
    
    return modified

def process_directory(directory):
    """Process all Python files in a directory."""
    updated_files = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if update_imports_in_file(filepath):
                    updated_files += 1
    return updated_files

if __name__ == "__main__":
    # Process examples and tests directories
    directories = ["examples", "tests", "tests_additional"]
    total_updated = 0
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"Processing {directory}...")
            updated = process_directory(directory)
            total_updated += updated
            print(f"Updated {updated} files in {directory}")
    
    print(f"Total files updated: {total_updated}")