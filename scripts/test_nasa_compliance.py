#!/usr/bin/env python3
"""
Simple NASA POT10 Compliance Test
=================================

Quick test of the NASA POT10 compliance system
"""

import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Now import the modules
try:
    from security.nasa_pot10_analyzer import NASAPOT10Analyzer
    from security.compliance_scorer import ComplianceScorer
    from security.compliance_gate import ComplianceGate
    from constants.base import NASA_POT10_TARGET_COMPLIANCE_THRESHOLD

    print("✅ All imports successful!")
    print(f"🎯 Target compliance threshold: {NASA_POT10_TARGET_COMPLIANCE_THRESHOLD:.1%}")

    # Create analyzer instance
    analyzer = NASAPOT10Analyzer()
    print("✅ NASA POT10 Analyzer created")

    # Create scorer instance
    scorer = ComplianceScorer()
    print("✅ Compliance Scorer created")

    # Create gate instance
    gate = ComplianceGate()
    print("✅ Compliance Gate created")

    # Test on a simple Python file
    test_file = project_root / "src" / "constants" / "base.py"
    if test_file.exists():
        print(f"🔍 Testing analysis on: {test_file}")

        violations = analyzer.analyze_file(str(test_file))
        print(f"📊 Found {len(violations)} violations")

        if violations:
            print("Top violations:")
            for i, v in enumerate(violations[:5], 1):
                print(f"  {i}. Rule {v.rule_number}: {v.description} (severity: {v.severity})")
        else:
            print("🎉 No violations found!")

        # Test compliance scoring
        with open(test_file, 'r') as f:
            lines = len([line for line in f if line.strip() and not line.strip().startswith('#')])

        file_compliance = scorer.calculate_file_compliance(str(test_file), violations, lines)
        print(f"📈 File compliance score: {file_compliance.compliance_score:.1%}")
        print(f"🏷️  Compliance level: {file_compliance.compliance_level.value}")

        # Test gate functionality
        print("🚪 Testing compliance gate...")
        gate_result = gate.check_compliance(str(test_file), mode="info")
        print(f"Gate result: {gate_result.message}")
        print(f"Gate passed: {'✅' if gate_result.passed else '❌'}")

    else:
        print(f"❌ Test file not found: {test_file}")

    print("\n🎉 NASA POT10 Compliance System is operational!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)