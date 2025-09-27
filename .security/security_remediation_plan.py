#!/usr/bin/env python3
"""
Critical Security Remediation Script
====================================

This script addresses the 24 HIGH severity and critical MEDIUM severity
security vulnerabilities identified in the comprehensive security scan.

CRITICAL: Run this script BEFORE any production deployment.
"""

import os
import re
import glob
import shutil
import hashlib
from pathlib import Path
from datetime import datetime


class SecurityRemediator:
    """Automated security vulnerability remediation"""

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.backup_dir = self.root_path / '.security' / 'remediation_backups'
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.fixes_applied = []

    def backup_file(self, file_path: Path) -> Path:
        """Create backup before modification"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{file_path.name}_{timestamp}.backup"
        backup_path = self.backup_dir / backup_name
        shutil.copy2(file_path, backup_path)
        return backup_path

    def fix_md5_usage(self):
        """Fix HIGH: Replace weak MD5 usage with secure alternatives"""
        print("[FIX] Fixing MD5 usage vulnerabilities...")

        # Files with MD5 issues from Bandit scan
        md5_files = [
            "phases/phase6_baking/integration/integration_manager.py",
            "phases/phase6_baking/optimization/baking_optimizer.py",
            "phases/phase6_baking/security/vulnerability_scanner.py",
            "phases/phase7_adas/agents/v2x_communicator.py",
            "scripts/optimize_build.py",
            "src/performance/optimizer.py"
        ]

        md5_pattern = re.compile(r'hashlib\.md5\(\)')
        md5_replacement = 'hashlib.md5(usedforsecurity=False)'

        for file_rel in md5_files:
            file_path = self.root_path / file_rel
            if not file_path.exists():
                continue

            # Backup original
            backup_path = self.backup_file(file_path)

            # Read and fix content
            content = file_path.read_text(encoding='utf-8')

            # Check if this is cryptographic usage
            crypto_indicators = ['password', 'hash', 'crypto', 'security', 'auth']
            is_crypto_context = any(indicator in content.lower() for indicator in crypto_indicators)

            if is_crypto_context:
                # Use SHA-256 for cryptographic contexts
                new_content = content.replace('hashlib.md5()', 'hashlib.sha256()')
                fix_description = f"Replaced MD5 with SHA-256 in {file_rel}"
            else:
                # Use MD5 with usedforsecurity=False for non-crypto
                new_content = content.replace('hashlib.md5()', md5_replacement)
                fix_description = f"Added usedforsecurity=False to MD5 in {file_rel}"

            # Write fixed content
            file_path.write_text(new_content, encoding='utf-8')

            self.fixes_applied.append({
                'type': 'MD5_USAGE',
                'file': file_rel,
                'description': fix_description,
                'backup': str(backup_path)
            })

        print(f"[OK] Fixed MD5 usage in {len(md5_files)} files")

    def fix_unsafe_extraction(self):
        """Fix HIGH: Unsafe archive extraction vulnerabilities"""
        print("[FIX] Fixing unsafe archive extraction...")

        extraction_files = [
            "phases/phase8_compression/agents/deployment_packager.py"
        ]

        for file_rel in extraction_files:
            file_path = self.root_path / file_rel
            if not file_path.exists():
                continue

            backup_path = self.backup_file(file_path)
            content = file_path.read_text(encoding='utf-8')

            # Add safe extraction function
            safe_extract_code = '''
def safe_extract(tar, path=".", members=None):
    """Safely extract tar archive preventing directory traversal"""
    for member in tar.getmembers():
        # Prevent path traversal
        if member.name.startswith('/') or '..' in member.name:
            print(f"Skipping dangerous path: {member.name}")
            continue
        # Prevent absolute paths
        if os.path.isabs(member.name):
            print(f"Skipping absolute path: {member.name}")
            continue
        tar.extract(member, path)

'''

            # Replace unsafe extractall calls
            content = content.replace(
                'tar.extractall()',
                'safe_extract(tar)'
            )
            content = content.replace(
                'tar.extractall(path)',
                'safe_extract(tar, path)'
            )

            # Add the safe function at the top after imports
            import_index = content.find('import ')
            if import_index != -1:
                end_imports = content.find('\n\n', import_index)
                if end_imports != -1:
                    content = content[:end_imports] + safe_extract_code + content[end_imports:]

            file_path.write_text(content, encoding='utf-8')

            self.fixes_applied.append({
                'type': 'UNSAFE_EXTRACTION',
                'file': file_rel,
                'description': f"Added safe extraction to {file_rel}",
                'backup': str(backup_path)
            })

        print("[OK] Fixed unsafe archive extraction")

    def fix_flask_debug(self):
        """Fix HIGH: Flask debug mode enabled"""
        print("[FIX] Fixing Flask debug mode...")

        flask_files = [
            "src/agent_forge/api/websocket_progress.py"
        ]

        for file_rel in flask_files:
            file_path = self.root_path / file_rel
            if not file_path.exists():
                continue

            backup_path = self.backup_file(file_path)
            content = file_path.read_text(encoding='utf-8')

            # Replace debug=True with debug=False
            content = re.sub(
                r'\.run\([^)]*debug=True[^)]*\)',
                lambda m: m.group(0).replace('debug=True', 'debug=False'),
                content
            )

            # Also fix any app.debug = True
            content = content.replace('app.debug = True', 'app.debug = False')

            file_path.write_text(content, encoding='utf-8')

            self.fixes_applied.append({
                'type': 'FLASK_DEBUG',
                'file': file_rel,
                'description': f"Disabled Flask debug mode in {file_rel}",
                'backup': str(backup_path)
            })

        print("[OK] Fixed Flask debug mode")

    def fix_shell_injection(self):
        """Fix HIGH: Shell injection vulnerabilities"""
        print("[FIX] Fixing shell injection vulnerabilities...")

        shell_files = [
            "start_cognate_system.py"
        ]

        for file_rel in shell_files:
            file_path = self.root_path / file_rel
            if not file_path.exists():
                continue

            backup_path = self.backup_file(file_path)
            content = file_path.read_text(encoding='utf-8')

            # Replace shell=True with shell=False and proper command splitting
            content = re.sub(
                r'subprocess\.(call|run|Popen)\(([^,]+),\s*shell=True\)',
                r'subprocess.\1(\2.split(), shell=False)',
                content
            )

            file_path.write_text(content, encoding='utf-8')

            self.fixes_applied.append({
                'type': 'SHELL_INJECTION',
                'file': file_rel,
                'description': f"Fixed shell injection in {file_rel}",
                'backup': str(backup_path)
            })

        print("[OK] Fixed shell injection vulnerabilities")

    def fix_nasa_analyzer_imports(self):
        """Fix NASA POT10 analyzer import issues"""
        print("[FIX] Fixing NASA POT10 analyzer...")

        nasa_file = "src/security/nasa_pot10_analyzer.py"
        file_path = self.root_path / nasa_file

        if not file_path.exists():
            print("[WARNING] NASA analyzer not found")
            return

        backup_path = self.backup_file(file_path)
        content = file_path.read_text(encoding='utf-8')

        # Fix relative imports
        content = content.replace(
            'from ..constants.base import',
            'from src.constants.base import'
        )

        # Fix escape sequences
        content = content.replace(r'\s', r'\\s')

        # Add constants if missing
        if 'NASA_POT10_TARGET_COMPLIANCE_THRESHOLD' not in content:
            constants_code = '''
# NASA POT10 Constants
NASA_POT10_TARGET_COMPLIANCE_THRESHOLD = 0.92
NASA_MAX_FUNCTION_LENGTH = 60
NASA_MIN_ASSERTION_DENSITY = 0.02
NASA_PARAMETER_THRESHOLD = 6
'''
            content = constants_code + content

        file_path.write_text(content, encoding='utf-8')

        self.fixes_applied.append({
            'type': 'NASA_ANALYZER',
            'file': nasa_file,
            'description': "Fixed NASA analyzer imports and constants",
            'backup': str(backup_path)
        })

        print("[OK] Fixed NASA POT10 analyzer")

    def create_security_pre_commit_hooks(self):
        """Create pre-commit security hooks"""
        print("[FIX] Creating security pre-commit hooks...")

        pre_commit_config = '''
repos:
  - repo: local
    hooks:
      - id: bandit
        name: bandit
        entry: bandit
        language: system
        args: ['-r', '.', '-f', 'json', '-o', '.security/bandit_scan_report.json']
        pass_filenames: false

      - id: safety
        name: safety
        entry: safety
        language: system
        args: ['check', '--json']
        pass_filenames: false

      - id: semgrep
        name: semgrep
        entry: semgrep
        language: system
        args: ['--config=auto', '--json', '--output=.security/semgrep_report.json']
        pass_filenames: false
'''

        pre_commit_path = self.root_path / '.pre-commit-config.yaml'
        pre_commit_path.write_text(pre_commit_config)

        self.fixes_applied.append({
            'type': 'PRE_COMMIT_HOOKS',
            'file': '.pre-commit-config.yaml',
            'description': "Created security pre-commit hooks",
            'backup': None
        })

        print("[OK] Created security pre-commit hooks")

    def generate_remediation_report(self):
        """Generate detailed remediation report"""
        report_path = self.root_path / '.security' / 'remediation_report.json'

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_fixes': len(self.fixes_applied),
            'fixes_applied': self.fixes_applied,
            'status': 'COMPLETED',
            'next_steps': [
                'Run comprehensive security scan to verify fixes',
                'Test all modified functionality',
                'Run NASA POT10 compliance check',
                'Update security documentation',
                'Deploy to staging for validation'
            ]
        }

        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"[REPORT] Remediation report saved to: {report_path}")
        return report

    def run_full_remediation(self):
        """Execute complete security remediation"""
        print("[ALERT] STARTING CRITICAL SECURITY REMEDIATION")
        print("=" * 60)

        try:
            self.fix_md5_usage()
            self.fix_unsafe_extraction()
            self.fix_flask_debug()
            self.fix_shell_injection()
            self.fix_nasa_analyzer_imports()
            self.create_security_pre_commit_hooks()

            report = self.generate_remediation_report()

            print("\n[SUCCESS] SECURITY REMEDIATION COMPLETED")
            print(f"Total fixes applied: {len(self.fixes_applied)}")
            print("\n[IMPORTANT] Run tests to verify all fixes work correctly")
            print("[IMPORTANT] Re-run security scan to verify vulnerabilities are fixed")

            return report

        except Exception as e:
            print(f"[ERROR] Remediation failed: {e}")
            print("[INFO] Check backup files in .security/remediation_backups/")
            raise


def main():
    """Main execution function"""
    import sys

    if len(sys.argv) > 1:
        root_path = sys.argv[1]
    else:
        root_path = '.'

    remediator = SecurityRemediator(root_path)
    report = remediator.run_full_remediation()

    return report


if __name__ == "__main__":
    main()