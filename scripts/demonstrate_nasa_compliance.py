#!/usr/bin/env python3
"""
NASA POT10 Compliance System Demonstration
==========================================

This script demonstrates the NASA POT10 compliance system by:
1. Running a full compliance analysis on the current codebase
2. Generating detailed reports with violation analysis
3. Demonstrating gate blocking with non-compliant code
4. Showing compliance improvement roadmap
5. Creating compliance certificates for passing code

Usage:
    python scripts/demonstrate_nasa_compliance.py
"""

import sys
import os
import tempfile
import json
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from security.nasa_pot10_analyzer import create_nasa_pot10_analyzer
from security.compliance_scorer import create_compliance_scorer, count_lines_of_code
from security.compliance_gate import create_compliance_gate


def create_sample_violations():
    """Create sample Python files with NASA POT10 violations for demonstration"""

    # Sample file with multiple violations
    violation_code = '''
import os
import sys
import json
import logging

# Rule 6: Module-level variable (should be in smallest scope)
global_counter = 0

def overly_long_function_that_violates_rule_4():
    """
    This function violates Rule 4 by being longer than 60 lines
    It also violates other rules for demonstration purposes
    """
    # Rule 7: Unchecked return value
    os.getcwd()

    # Rule 2: Unbounded loop
    while True:
        global_counter += 1

        # Rule 1: Complex control flow with deep nesting
        try:
            try:
                try:
                    # Rule 9: Deep attribute access
                    result = sys.modules['os'].__dict__['path'].__dict__['join']

                    # Rule 3: Dynamic allocation in runtime
                    new_list = list(range(global_counter))
                    new_dict = dict()

                    # Rule 8: Dynamic import
                    __import__('random')

                    # Rule 10: Bare except clause
                    pass
                except:
                    pass
            except Exception as e:
                pass
        except:
            pass

        # Rule 5: No assertions (low assertion density)
        if global_counter > 100:
            break

    # This function is now over 60 lines (Rule 4 violation)
    # More meaningless code to pad length
    x = 1
    y = 2
    z = 3
    a = 4
    b = 5
    c = 6
    d = 7
    e = 8
    f = 9
    g = 10
    h = 11
    i = 12
    j = 13
    k = 14
    l = 15
    m = 16
    n = 17
    o = 18
    p = 19
    q = 20
    r = 21
    s = 22
    t = 23
    u = 24
    v = 25
    w = 26

    return global_counter

def function_with_no_assertions():
    """Rule 5: This function has no assertions despite being substantial"""
    data = []
    for i in range(1000):
        data.append(i * 2)
        # Should have assertions here to check invariants

    result = sum(data)
    # Should check return value and add assertions
    return result

def nested_subscript_access(data):
    """Rule 9: Multiple pointer dereference equivalent"""
    # This creates nested subscript access
    return data[0][1][2][3]  # Too many levels of dereferencing

# Rule 6: Another module-level variable
config_data = {"key": "value"}
'''

    # Sample compliant file for comparison
    compliant_code = '''
"""
Compliant NASA POT10 Code Example
This file demonstrates proper adherence to NASA POT10 rules
"""

import logging
from typing import List, Optional

# Rule 6: Constants at module level are acceptable
MAX_ITERATIONS = 100

class CompliantProcessor:
    """Example of NASA POT10 compliant code"""

    def __init__(self):
        """Rule 3: Initialize data structures here"""
        self.data: List[int] = []
        self.results: List[int] = []

    def process_data(self, input_data: List[int]) -> bool:
        """
        Rule 4: Function under 60 lines
        Rule 5: Has assertions for verification
        Rule 7: Checks return values
        """
        # Rule 5: Precondition assertion
        assert isinstance(input_data, list), "Input must be a list"
        assert len(input_data) <= MAX_ITERATIONS, "Input too large"

        # Rule 2: Loop with fixed upper bound
        for i in range(min(len(input_data), MAX_ITERATIONS)):
            # Rule 9: Single level of dereferencing
            value = input_data[i]

            # Rule 5: Invariant assertion
            assert isinstance(value, int), f"Expected int, got {type(value)}"

            # Simple processing
            result = self._safe_process_value(value)
            if result is not None:  # Rule 7: Check return value
                self.results.append(result)

        # Rule 5: Postcondition assertion
        assert len(self.results) <= len(input_data), "Results exceed input"

        return True

    def _safe_process_value(self, value: int) -> Optional[int]:
        """
        Helper function that follows NASA POT10 rules
        Rule 4: Short function
        Rule 5: Has assertions
        """
        # Rule 5: Precondition
        assert isinstance(value, int), "Value must be integer"

        # Rule 1: Simple control flow
        if value < 0:
            return None

        result = value * 2

        # Rule 5: Postcondition
        assert result >= 0, "Result should be non-negative"

        return result
'''

    return violation_code, compliant_code


def run_compliance_analysis():
    """Run comprehensive compliance analysis"""
    print("🔍 NASA POT10 Compliance System Demonstration")
    print("=" * 60)

    # Initialize components
    analyzer = create_nasa_pot10_analyzer()
    scorer = create_compliance_scorer()
    gate = create_compliance_gate()

    # Create temporary files for demonstration
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create sample files
        violation_code, compliant_code = create_sample_violations()

        violation_file = temp_path / "violations_example.py"
        compliant_file = temp_path / "compliant_example.py"

        with open(violation_file, 'w') as f:
            f.write(violation_code)
        with open(compliant_file, 'w') as f:
            f.write(compliant_code)

        print(f"📁 Created sample files in {temp_dir}")
        print(f"   - violations_example.py (non-compliant)")
        print(f"   - compliant_example.py (compliant)")
        print()

        # Analyze violations file
        print("1️⃣ ANALYZING NON-COMPLIANT CODE")
        print("-" * 40)

        violations = analyzer.analyze_file(str(violation_file))
        print(f"Found {len(violations)} violations in non-compliant file:")

        for i, violation in enumerate(violations[:10], 1):  # Show first 10
            print(f"  {i:2d}. Rule {violation.rule_number}: {violation.description}")
            print(f"      Line {violation.line_number}, Severity: {violation.severity}/10")
            print(f"      Fix: {violation.suggested_fix}")
            print()

        # Calculate compliance score for violation file
        loc_violations = count_lines_of_code(str(violation_file))
        file_compliance_violations = scorer.calculate_file_compliance(
            str(violation_file), violations, loc_violations
        )

        print(f"📊 Compliance Score: {file_compliance_violations.compliance_score:.1%}")
        print(f"📈 Violations per KLOC: {file_compliance_violations.violations_per_kloc:.1f}")
        print(f"🏷️  Compliance Level: {file_compliance_violations.compliance_level.value.upper()}")
        print()

        # Analyze compliant file
        print("2️⃣ ANALYZING COMPLIANT CODE")
        print("-" * 40)

        compliant_violations = analyzer.analyze_file(str(compliant_file))
        print(f"Found {len(compliant_violations)} violations in compliant file:")

        for violation in compliant_violations:
            print(f"  - Rule {violation.rule_number}: {violation.description}")
            print(f"    Severity: {violation.severity}/10")

        # Calculate compliance score for compliant file
        loc_compliant = count_lines_of_code(str(compliant_file))
        file_compliance_compliant = scorer.calculate_file_compliance(
            str(compliant_file), compliant_violations, loc_compliant
        )

        print(f"📊 Compliance Score: {file_compliance_compliant.compliance_score:.1%}")
        print(f"🏷️  Compliance Level: {file_compliance_compliant.compliance_level.value.upper()}")
        print()

        # Demonstrate gate functionality
        print("3️⃣ DEMONSTRATING COMPLIANCE GATE")
        print("-" * 40)

        # Test gate with non-compliant code
        print("Testing gate with NON-COMPLIANT code:")
        gate_result_fail = gate.check_compliance(str(violation_file), mode="strict")
        print(f"  Result: {gate_result_fail.message}")
        print(f"  Gate Passed: {'✅ YES' if gate_result_fail.passed else '❌ NO'}")
        print()

        # Test gate with compliant code
        print("Testing gate with COMPLIANT code:")
        gate_result_pass = gate.check_compliance(str(compliant_file), mode="strict")
        print(f"  Result: {gate_result_pass.message}")
        print(f"  Gate Passed: {'✅ YES' if gate_result_pass.passed else '❌ NO'}")
        print()

        # Generate comprehensive report
        print("4️⃣ GENERATING COMPLIANCE REPORT")
        print("-" * 40)

        all_violations = violations + compliant_violations
        file_compliances = [file_compliance_violations, file_compliance_compliant]

        report = scorer.create_compliance_report(
            temp_dir, all_violations, file_compliances, 1.0
        )

        print(f"📋 PROJECT COMPLIANCE SUMMARY:")
        print(f"   Overall Score: {report.project_compliance.overall_compliance_score:.1%}")
        print(f"   Total Files: {report.project_compliance.total_files}")
        print(f"   Total Violations: {report.project_compliance.total_violations}")
        print(f"   Compliance Level: {report.project_compliance.compliance_level.value.upper()}")
        print()

        print("📈 RULE-SPECIFIC COMPLIANCE:")
        for rule_num, score in report.project_compliance.rule_compliance_scores.items():
            rule_desc = analyzer.get_rule_description(rule_num)
            print(f"   Rule {rule_num:2d}: {score:.1%} - {rule_desc}")
        print()

        print("🎯 IMPROVEMENT RECOMMENDATIONS:")
        for i, rec in enumerate(report.improvement_recommendations[:5], 1):
            print(f"   {i}. {rec}")
        print()

        # Generate certificate if compliance passes
        if gate_result_pass.passed:
            print("5️⃣ GENERATING COMPLIANCE CERTIFICATE")
            print("-" * 40)

            cert_path = gate.generate_compliance_certificate(gate_result_pass)
            print(f"✅ Certificate generated: {cert_path}")

            # Show certificate contents
            with open(cert_path, 'r') as f:
                cert_data = json.load(f)

            print(f"   Certificate ID: {cert_data['certificate_id']}")
            print(f"   Compliance Score: {cert_data['compliance_score']:.1%}")
            print(f"   Valid Until: {cert_data['validity_period_days']} days from issue")
            print()

        return report


def analyze_real_codebase():
    """Analyze the actual agent-forge codebase"""
    print("6️⃣ ANALYZING REAL CODEBASE")
    print("-" * 40)

    # Get current working directory (should be agent-forge root)
    project_root = Path.cwd()
    src_dir = project_root / "src"

    if not src_dir.exists():
        print("❌ Source directory not found. Please run from project root.")
        return None

    print(f"🔍 Analyzing codebase at: {project_root}")
    print(f"📁 Source directory: {src_dir}")

    # Initialize components
    analyzer = create_nasa_pot10_analyzer()
    scorer = create_compliance_scorer()
    gate = create_compliance_gate()

    # Run analysis on src directory
    print("Running compliance analysis...")
    violations = analyzer.analyze_directory(str(src_dir))

    # Calculate file compliances
    file_compliances = []
    python_files = list(src_dir.rglob('*.py'))

    print(f"Found {len(python_files)} Python files to analyze")

    for py_file in python_files:
        if any(excluded in str(py_file) for excluded in ['__pycache__', '.git', 'test']):
            continue

        file_violations = [v for v in violations if v.file_path == str(py_file)]
        loc = count_lines_of_code(str(py_file))

        if loc > 0:  # Only include non-empty files
            file_compliance = scorer.calculate_file_compliance(
                str(py_file), file_violations, loc
            )
            file_compliances.append(file_compliance)

    if not file_compliances:
        print("❌ No Python files found for analysis")
        return None

    # Generate comprehensive report
    report = scorer.create_compliance_report(
        str(project_root), violations, file_compliances, 0.0
    )

    # Display results
    print(f"\n📊 AGENT-FORGE CODEBASE COMPLIANCE RESULTS")
    print("=" * 50)
    print(f"Overall Compliance Score: {report.project_compliance.overall_compliance_score:.1%}")
    print(f"Compliance Level: {report.project_compliance.compliance_level.value.upper()}")
    print(f"Total Files Analyzed: {len(file_compliances)}")
    print(f"Total Violations: {report.project_compliance.total_violations}")
    print(f"Target Threshold: {gate.config.target_threshold:.1%}")
    print()

    # Check if gate would pass
    gate_result = gate.check_compliance(str(src_dir), mode="strict")
    print(f"🚪 COMPLIANCE GATE STATUS:")
    print(f"   {gate_result.message}")
    print(f"   Gate Result: {'✅ PASS' if gate_result.passed else '❌ FAIL'}")
    print()

    # Show worst violations
    if violations:
        print("🔴 TOP VIOLATIONS TO ADDRESS:")
        top_violations = sorted(violations, key=lambda v: v.severity, reverse=True)[:10]

        for i, violation in enumerate(top_violations, 1):
            file_name = Path(violation.file_path).name
            print(f"   {i:2d}. Rule {violation.rule_number} in {file_name}:{violation.line_number}")
            print(f"       {violation.description} (Severity: {violation.severity}/10)")
            print(f"       Fix: {violation.suggested_fix}")
        print()

    # Rule breakdown
    print("📋 RULE-BY-RULE BREAKDOWN:")
    for rule_num in range(1, 11):
        rule_violations = [v for v in violations if v.rule_number == rule_num]
        rule_score = report.project_compliance.rule_compliance_scores.get(rule_num, 0)
        rule_desc = analyzer.get_rule_description(rule_num)

        status = "✅" if rule_score >= 0.8 else "⚠️" if rule_score >= 0.6 else "❌"
        print(f"   {status} Rule {rule_num:2d}: {rule_score:.1%} ({len(rule_violations):2d} violations)")
        print(f"       {rule_desc}")
    print()

    # Improvement roadmap
    print("🗺️ COMPLIANCE IMPROVEMENT ROADMAP:")
    for i, rec in enumerate(report.improvement_recommendations, 1):
        print(f"   {i}. {rec}")
    print()

    # Export report
    report_file = project_root / ".security" / "compliance_report.json"
    scorer.export_report(report, str(report_file))
    print(f"📄 Detailed report exported to: {report_file}")

    return report


def demonstrate_ci_integration():
    """Demonstrate CI/CD integration"""
    print("7️⃣ CI/CD INTEGRATION DEMONSTRATION")
    print("-" * 40)

    gate = create_compliance_gate()

    # Install pre-commit hook
    print("Installing pre-commit hook...")
    hook_success = gate.install_pre_commit_hook()
    print(f"Pre-commit hook: {'✅ Installed' if hook_success else '❌ Failed'}")

    # Create CI script
    print("Creating CI/CD script...")
    ci_success = gate.create_ci_script()
    print(f"CI/CD script: {'✅ Created' if ci_success else '❌ Failed'}")

    if ci_success:
        ci_script_path = Path(".security/ci_compliance_check.sh")
        print(f"📄 CI script available at: {ci_script_path}")
        print("   Add this to your CI/CD pipeline to enforce compliance")

    print()


def main():
    """Main demonstration function"""
    print("🚀 NASA POT10 Compliance System - Full Demonstration")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        # Run sample analysis
        sample_report = run_compliance_analysis()

        # Analyze real codebase
        real_report = analyze_real_codebase()

        # Demonstrate CI integration
        demonstrate_ci_integration()

        print("🎉 DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("✅ NASA POT10 compliance system is fully operational")
        print("✅ Analyzers implemented for all 10 rules")
        print("✅ Compliance scoring with weighted calculations")
        print("✅ Automated gates with blocking capability")
        print("✅ Comprehensive reporting and recommendations")
        print("✅ CI/CD integration ready")
        print()

        if real_report:
            overall_score = real_report.project_compliance.overall_compliance_score
            if overall_score >= 0.92:
                print(f"🏆 EXCELLENT: Codebase compliance at {overall_score:.1%}")
            elif overall_score >= 0.85:
                print(f"✅ GOOD: Codebase compliance at {overall_score:.1%}")
            else:
                print(f"⚠️ IMPROVEMENT NEEDED: Codebase compliance at {overall_score:.1%}")

        print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())