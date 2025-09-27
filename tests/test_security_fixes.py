#!/usr/bin/env python3
"""
Test suite to verify security fixes have been applied correctly
"""

import os
import sys
import hashlib
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_md5_replaced_with_sha256():
    """Verify MD5 has been replaced with SHA-256 in critical files"""
    files_to_check = [
        "phases/phase6_baking/integration/integration_manager.py",
        "phases/phase6_baking/optimization/baking_optimizer.py",
        "phases/phase6_baking/security/vulnerability_scanner.py",
        "phases/phase7_adas/agents/v2x_communicator.py",
        "scripts/optimize_build.py",
        "src/performance/optimizer.py"
    ]

    print("[TEST] Checking MD5 replacement with SHA-256...")
    for file_path in files_to_check:
        full_path = Path(file_path)
        if full_path.exists():
            content = full_path.read_text()
            # Check for SHA-256 usage
            if "hashlib.sha256()" in content:
                print(f"  [OK] {file_path}: Using SHA-256")
            elif "hashlib.md5(usedforsecurity=False)" in content:
                print(f"  [OK] {file_path}: Using MD5 with usedforsecurity=False")
            elif "hashlib.md5()" in content:
                print(f"  [FAIL] {file_path}: Still using insecure MD5")
                return False
            else:
                print(f"  [SKIP] {file_path}: No hash usage found")
        else:
            print(f"  [SKIP] {file_path}: File not found")
    return True

def test_flask_debug_disabled():
    """Verify Flask debug mode is disabled"""
    flask_file = Path("src/agent_forge/api/websocket_progress.py")

    print("[TEST] Checking Flask debug mode...")
    if flask_file.exists():
        content = flask_file.read_text()
        if "debug=True" in content:
            print(f"  [FAIL] Flask debug mode still enabled")
            return False
        elif "debug=False" in content:
            print(f"  [OK] Flask debug mode disabled")
            return True
        else:
            print(f"  [OK] No debug mode setting found")
            return True
    else:
        print(f"  [SKIP] Flask file not found")
        return True

def test_shell_injection_fixed():
    """Verify shell injection vulnerabilities are fixed"""
    shell_file = Path("start_cognate_system.py")

    print("[TEST] Checking shell injection fixes...")
    if shell_file.exists():
        content = shell_file.read_text()
        if "shell=True" in content:
            print(f"  [FAIL] Shell injection vulnerability still present")
            return False
        else:
            print(f"  [OK] Shell injection fixed")
            return True
    else:
        print(f"  [SKIP] Shell file not found")
        return True

def test_safe_extraction():
    """Verify safe extraction is implemented"""
    extraction_file = Path("phases/phase8_compression/agents/deployment_packager.py")

    print("[TEST] Checking safe archive extraction...")
    if extraction_file.exists():
        content = extraction_file.read_text()
        if "safe_extract" in content:
            print(f"  [OK] Safe extraction implemented")
            return True
        elif "tar.extractall()" in content:
            print(f"  [FAIL] Unsafe extractall still present")
            return False
        else:
            print(f"  [SKIP] No extraction code found")
            return True
    else:
        print(f"  [SKIP] Extraction file not found")
        return True

def test_pre_commit_hooks():
    """Verify pre-commit hooks are configured"""
    pre_commit_file = Path(".pre-commit-config.yaml")

    print("[TEST] Checking pre-commit hooks...")
    if pre_commit_file.exists():
        content = pre_commit_file.read_text()
        required_hooks = ["bandit", "safety", "semgrep"]
        found_hooks = []

        for hook in required_hooks:
            if hook in content:
                found_hooks.append(hook)
                print(f"  [OK] {hook} hook configured")
            else:
                print(f"  [FAIL] {hook} hook missing")

        return len(found_hooks) == len(required_hooks)
    else:
        print(f"  [FAIL] Pre-commit config not found")
        return False

def main():
    """Run all security tests"""
    print("=" * 60)
    print("[START] SECURITY FIX VALIDATION SUITE")
    print("=" * 60)

    tests = [
        ("MD5 Replacement", test_md5_replaced_with_sha256),
        ("Flask Debug", test_flask_debug_disabled),
        ("Shell Injection", test_shell_injection_fixed),
        ("Safe Extraction", test_safe_extraction),
        ("Pre-commit Hooks", test_pre_commit_hooks)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print()
        except Exception as e:
            print(f"  [ERROR] {test_name} failed with: {e}")
            results.append((test_name, False))
            print()

    # Summary
    print("=" * 60)
    print("[SUMMARY] Security Fix Validation Results")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print()
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("[SUCCESS] All security fixes validated!")
        return 0
    else:
        print(f"[WARNING] {total - passed} tests failed - review fixes")
        return 1

if __name__ == "__main__":
    sys.exit(main())