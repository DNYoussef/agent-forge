#!/usr/bin/env python3
"""
Directory consolidation script for agent-forge project.
Consolidates duplicate directory structures and updates import paths.
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Set
import re

class DirectoryConsolidator:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.consolidation_plan = {}
        self.updated_imports = []
        self.errors = []

    def analyze_duplicate_directories(self) -> Dict[str, List[Path]]:
        """Analyze and identify duplicate directory structures."""
        duplicate_analysis = {
            "phases": [],
            "src_directories": [],
            "config_directories": []
        }

        # Check for phases directories
        phases_dirs = list(self.base_dir.rglob("phases"))
        phases_dirs = [d for d in phases_dirs if d.is_dir()]
        duplicate_analysis["phases"] = phases_dirs

        # Check for src directories
        src_dirs = list(self.base_dir.rglob("src"))
        src_dirs = [d for d in src_dirs if d.is_dir()]
        duplicate_analysis["src_directories"] = src_dirs

        # Check for config directories
        config_dirs = list(self.base_dir.rglob("config"))
        config_dirs = [d for d in config_dirs if d.is_dir()]
        duplicate_analysis["config_directories"] = config_dirs

        return duplicate_analysis

    def create_consolidation_plan(self, duplicate_analysis: Dict) -> Dict:
        """Create a consolidation plan based on analysis."""
        plan = {
            "target_structure": {
                "phases": self.base_dir / "phases",
                "src": self.base_dir / "src",
                "config": self.base_dir / "config"
            },
            "moves": [],
            "deletions": []
        }

        # Plan phases consolidation
        if len(duplicate_analysis["phases"]) > 1:
            # Keep the root phases directory, move contents from others
            target_phases = self.base_dir / "phases"
            for phases_dir in duplicate_analysis["phases"]:
                if phases_dir != target_phases:
                    plan["moves"].append({
                        "source": phases_dir,
                        "target": target_phases,
                        "action": "merge_contents"
                    })
                    plan["deletions"].append(phases_dir)

        return plan

    def find_import_references(self, old_path: str, new_path: str) -> List[Path]:
        """Find files that need import path updates."""
        files_to_update = []

        # Common file extensions that might contain imports
        extensions = ['.py', '.js', '.ts', '.json', '.yaml', '.yml']

        # Convert paths to relative for import matching
        old_import = old_path.replace('\\', '/').replace('/', '.')

        for ext in extensions:
            for file_path in self.base_dir.rglob(f"*{ext}"):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Look for import statements referencing the old path
                        if old_path in content or old_import in content:
                            files_to_update.append(file_path)
                except Exception:
                    continue

        return files_to_update

    def merge_directory_contents(self, source_dir: Path, target_dir: Path) -> bool:
        """Merge contents from source directory into target directory."""
        try:
            if not target_dir.exists():
                target_dir.mkdir(parents=True, exist_ok=True)

            for item in source_dir.iterdir():
                target_item = target_dir / item.name

                if item.is_file():
                    if target_item.exists():
                        # Handle file conflicts
                        backup_name = f"{target_item.stem}_backup{target_item.suffix}"
                        backup_path = target_item.parent / backup_name
                        shutil.move(str(target_item), str(backup_path))
                        print(f"  Backed up existing file: {target_item} -> {backup_path}")

                    shutil.move(str(item), str(target_item))
                    print(f"  Moved file: {item} -> {target_item}")

                elif item.is_dir():
                    if target_item.exists():
                        # Recursively merge subdirectories
                        self.merge_directory_contents(item, target_item)
                    else:
                        shutil.move(str(item), str(target_item))
                        print(f"  Moved directory: {item} -> {target_item}")

            return True
        except Exception as e:
            self.errors.append(f"Error merging {source_dir} into {target_dir}: {e}")
            return False

    def update_import_paths(self, file_path: Path, old_path: str, new_path: str) -> bool:
        """Update import paths in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # Update various import patterns
            patterns = [
                # Python imports
                (rf'from\s+{re.escape(old_path.replace("/", "."))}\s+import', f'from {new_path.replace("/", ".")} import'),
                (rf'import\s+{re.escape(old_path.replace("/", "."))}', f'import {new_path.replace("/", ".")}'),
                # File path references
                (rf'{re.escape(old_path)}', new_path),
                # Require statements (JS/TS)
                (rf'require\(["\']([^"\']*{re.escape(old_path)}[^"\']*)["\']', lambda m: f'require("{m.group(1).replace(old_path, new_path)}")'),
                # Import statements (JS/TS)
                (rf'from\s+["\']([^"\']*{re.escape(old_path)}[^"\']*)["\']', lambda m: f'from "{m.group(1).replace(old_path, new_path)}"'),
            ]

            for pattern, replacement in patterns:
                if callable(replacement):
                    content = re.sub(pattern, replacement, content)
                else:
                    content = re.sub(pattern, replacement, content)

            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.updated_imports.append(str(file_path))
                return True

            return False
        except Exception as e:
            self.errors.append(f"Error updating imports in {file_path}: {e}")
            return False

    def execute_consolidation(self, plan: Dict) -> Dict:
        """Execute the consolidation plan."""
        results = {
            "moves_completed": 0,
            "deletions_completed": 0,
            "import_updates": 0,
            "errors": []
        }

        print("Executing consolidation plan...")

        # Execute moves
        for move in plan["moves"]:
            source = move["source"]
            target = move["target"]

            print(f"\nMerging: {source} -> {target}")

            if self.merge_directory_contents(source, target):
                results["moves_completed"] += 1

                # Update import paths
                old_path_str = str(source.relative_to(self.base_dir))
                new_path_str = str(target.relative_to(self.base_dir))

                files_to_update = self.find_import_references(old_path_str, new_path_str)
                for file_path in files_to_update:
                    if self.update_import_paths(file_path, old_path_str, new_path_str):
                        results["import_updates"] += 1

        # Execute deletions
        for deletion in plan["deletions"]:
            try:
                if deletion.exists() and not any(deletion.iterdir()):  # Only delete if empty
                    deletion.rmdir()
                    results["deletions_completed"] += 1
                    print(f"Deleted empty directory: {deletion}")
                elif deletion.exists():
                    print(f"Skipped deletion of non-empty directory: {deletion}")
            except Exception as e:
                error_msg = f"Error deleting {deletion}: {e}"
                results["errors"].append(error_msg)
                self.errors.append(error_msg)

        results["errors"] = self.errors
        return results

    def generate_report(self, duplicate_analysis: Dict, plan: Dict, results: Dict) -> str:
        """Generate consolidation report."""
        report = {
            "duplicate_analysis": {
                "phases_directories": [str(p) for p in duplicate_analysis["phases"]],
                "src_directories": [str(p) for p in duplicate_analysis["src_directories"]],
                "config_directories": [str(p) for p in duplicate_analysis["config_directories"]]
            },
            "consolidation_plan": {
                "target_structure": {k: str(v) for k, v in plan["target_structure"].items()},
                "moves_planned": len(plan["moves"]),
                "deletions_planned": len(plan["deletions"])
            },
            "execution_results": results,
            "updated_imports": self.updated_imports,
            "errors": self.errors
        }
        return json.dumps(report, indent=2)

def main():
    """Main function to consolidate directory structure."""
    consolidator = DirectoryConsolidator()

    print("=== Directory Consolidation Tool ===")
    print(f"Analyzing directory: {consolidator.base_dir.absolute()}")

    # Analyze duplicate directories
    print("\n1. Analyzing duplicate directory structures...")
    duplicate_analysis = consolidator.analyze_duplicate_directories()

    for category, dirs in duplicate_analysis.items():
        print(f"  {category}: {len(dirs)} directories found")
        for d in dirs:
            print(f"    - {d}")

    # Create consolidation plan
    print("\n2. Creating consolidation plan...")
    plan = consolidator.create_consolidation_plan(duplicate_analysis)
    print(f"  Moves planned: {len(plan['moves'])}")
    print(f"  Deletions planned: {len(plan['deletions'])}")

    # Execute consolidation
    print("\n3. Executing consolidation...")
    results = consolidator.execute_consolidation(plan)

    # Generate report
    print("\n4. Generating consolidation report...")
    report = consolidator.generate_report(duplicate_analysis, plan, results)

    report_file = consolidator.base_dir / "directory_consolidation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n=== CONSOLIDATION SUMMARY ===")
    print(f"Moves completed: {results['moves_completed']}")
    print(f"Deletions completed: {results['deletions_completed']}")
    print(f"Import updates: {results['import_updates']}")
    print(f"Files with updated imports: {len(consolidator.updated_imports)}")
    print(f"Errors encountered: {len(consolidator.errors)}")
    print(f"Report saved to: {report_file}")

    if consolidator.errors:
        print(f"\nErrors:")
        for error in consolidator.errors:
            print(f"  - {error}")

if __name__ == "__main__":
    main()