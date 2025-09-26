#!/usr/bin/env python3
"""
Enhanced automated merge conflict resolver.
Resolves conflicts across all file types by choosing HEAD version by default.
Handles multiple conflict patterns and provides comprehensive reporting.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple

class MergeConflictResolver:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.supported_extensions = {'.py', '.js', '.ts', '.md', '.json', '.yaml', '.yml', '.txt', '.sh', '.bat'}
        self.conflict_patterns = [
            # Standard git merge conflict
            re.compile(r'<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> origin/main', re.DOTALL),
            # Alternative patterns
            re.compile(r'<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> main', re.DOTALL),
            re.compile(r'<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> [a-f0-9]+', re.DOTALL),
            # Simple separator pattern
            re.compile(r'======\n', re.DOTALL)
        ]

    def resolve_merge_conflicts(self, file_path: Path) -> Tuple[int, List[str]]:
        """Resolve merge conflicts in a file by choosing HEAD version."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            return 0, [f"Error reading file: {e}"]

        original_content = content
        conflicts_resolved = 0
        resolution_details = []

        # Apply each conflict pattern
        for i, pattern in enumerate(self.conflict_patterns):
            if i < 3:  # For the first 3 patterns, replace with HEAD version
                matches = pattern.findall(content)
                if matches:
                    content = pattern.sub(lambda m: m.group(1) if len(m.groups()) >= 1 else '', content)
                    conflicts_resolved += len(matches)
                    resolution_details.append(f"Resolved {len(matches)} conflicts with pattern {i+1}")
            else:  # For simple separator, just remove
                matches = pattern.findall(content)
                if matches:
                    content = pattern.sub('', content)
                    conflicts_resolved += len(matches)
                    resolution_details.append(f"Removed {len(matches)} conflict separators")

        # Write back if changes were made
        if content != original_content:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                resolution_details.append(f"File updated successfully")
            except Exception as e:
                resolution_details.append(f"Error writing file: {e}")
                return 0, resolution_details

        return conflicts_resolved, resolution_details

    def scan_for_conflicts(self) -> List[Path]:
        """Scan for files containing merge conflict markers."""
        conflict_files = []
        conflict_markers = ['<<<<<<<', '======', '>>>>>>>']

        for ext in self.supported_extensions:
            for file_path in self.base_dir.rglob(f"*{ext}"):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if any(marker in content for marker in conflict_markers):
                            conflict_files.append(file_path)
                except Exception:
                    continue

        return conflict_files

    def cleanup_backup_files(self) -> List[str]:
        """Remove .backup files after resolution."""
        backup_files = list(self.base_dir.rglob("*.backup"))
        removed_files = []

        for backup_file in backup_files:
            try:
                backup_file.unlink()
                removed_files.append(str(backup_file))
            except Exception as e:
                print(f"Error removing {backup_file}: {e}")

        return removed_files

    def generate_report(self, results: Dict) -> str:
        """Generate comprehensive resolution report."""
        report = {
            "summary": {
                "files_scanned": results["files_scanned"],
                "files_with_conflicts": results["files_with_conflicts"],
                "total_conflicts_resolved": results["total_conflicts"],
                "backup_files_removed": len(results["backup_files_removed"])
            },
            "file_details": results["file_details"],
            "backup_files_removed": results["backup_files_removed"],
            "errors": results["errors"]
        }
        return json.dumps(report, indent=2)

def main():
    """Main function to resolve all merge conflicts."""
    resolver = MergeConflictResolver()

    print("=== Enhanced Merge Conflict Resolver ===")
    print(f"Scanning directory: {resolver.base_dir.absolute()}")

    # Scan for conflict files
    print("\n1. Scanning for files with merge conflicts...")
    conflict_files = resolver.scan_for_conflicts()
    print(f"Found {len(conflict_files)} files with potential conflicts")

    # Resolve conflicts
    print("\n2. Resolving merge conflicts...")
    results = {
        "files_scanned": len(conflict_files),
        "files_with_conflicts": 0,
        "total_conflicts": 0,
        "file_details": {},
        "backup_files_removed": [],
        "errors": []
    }

    for file_path in conflict_files:
        try:
            conflicts_resolved, details = resolver.resolve_merge_conflicts(file_path)
            if conflicts_resolved > 0:
                results["files_with_conflicts"] += 1
                results["total_conflicts"] += conflicts_resolved
                results["file_details"][str(file_path)] = {
                    "conflicts_resolved": conflicts_resolved,
                    "details": details
                }
                print(f"  [OK] {file_path}: {conflicts_resolved} conflicts resolved")
            else:
                print(f"  [-] {file_path}: No conflicts to resolve")
        except Exception as e:
            error_msg = f"Error processing {file_path}: {e}"
            results["errors"].append(error_msg)
            print(f"  [ERROR] {error_msg}")

    # Cleanup backup files
    print("\n3. Cleaning up backup files...")
    results["backup_files_removed"] = resolver.cleanup_backup_files()
    print(f"Removed {len(results['backup_files_removed'])} backup files")

    # Generate and save report
    print("\n4. Generating resolution report...")
    report = resolver.generate_report(results)

    report_file = resolver.base_dir / "merge_conflict_resolution_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n=== RESOLUTION SUMMARY ===")
    print(f"Files scanned: {results['files_scanned']}")
    print(f"Files with conflicts: {results['files_with_conflicts']}")
    print(f"Total conflicts resolved: {results['total_conflicts']}")
    print(f"Backup files removed: {len(results['backup_files_removed'])}")
    print(f"Errors encountered: {len(results['errors'])}")
    print(f"Report saved to: {report_file}")

    if results['errors']:
        print(f"\nErrors:")
        for error in results['errors']:
            print(f"  - {error}")

if __name__ == "__main__":
    main()