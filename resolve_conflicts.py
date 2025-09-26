#!/usr/bin/env python3
"""
Automated merge conflict resolver.
Resolves conflicts by choosing HEAD version by default.
"""

import os
import re
from pathlib import Path

def resolve_merge_conflicts(file_path):
    """Resolve merge conflicts in a file by choosing HEAD version."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to match merge conflict blocks
    conflict_pattern = re.compile(
        r'<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> origin/main',
        re.DOTALL
    )

    # Replace with HEAD version (first group)
    resolved_content = conflict_pattern.sub(r'\1', content)

    # Check if any conflicts were resolved
    conflicts_found = len(conflict_pattern.findall(content))

    if conflicts_found > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(resolved_content)
        print(f"Resolved {conflicts_found} conflicts in {file_path}")
        return conflicts_found
    else:
        print(f"No conflicts found in {file_path}")
        return 0

def main():
    """Main function to resolve all merge conflicts."""
    agent_forge_dir = Path("agent_forge")
    total_conflicts = 0
    files_processed = 0

    # Find all Python files with merge conflicts
    for py_file in agent_forge_dir.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if "<<<<<<< HEAD" in content:
                    conflicts = resolve_merge_conflicts(py_file)
                    total_conflicts += conflicts
                    files_processed += 1
        except Exception as e:
            print(f"Error processing {py_file}: {e}")

    print(f"\nSummary:")
    print(f"Files processed: {files_processed}")
    print(f"Total conflicts resolved: {total_conflicts}")

if __name__ == "__main__":
    main()