#!/usr/bin/env python3
"""
NASA POT10 Compliance Report Generator
=====================================

Simple standalone script to generate NASA POT10 compliance reports
"""

import os
import sys
import ast
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any

# Constants from base.py
NASA_POT10_TARGET_COMPLIANCE_THRESHOLD = 0.92
NASA_MAX_FUNCTION_LENGTH = 60
NASA_MIN_ASSERTION_DENSITY = 2.0

@dataclass
class Violation:
    rule_number: int
    file_path: str
    line_number: int
    function_name: str
    description: str
    severity: int
    code_snippet: str

class SimpleNASAAnalyzer:
    """Simplified NASA POT10 analyzer for demonstration"""

    def analyze_file(self, file_path: str) -> List[Violation]:
        """Analyze a Python file for NASA POT10 violations"""
        violations = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            tree = ast.parse(content)

            # Rule 4: Function length check
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if hasattr(node, 'end_lineno'):
                        func_length = node.end_lineno - node.lineno + 1
                    else:
                        func_length = 50  # Estimate

                    if func_length > NASA_MAX_FUNCTION_LENGTH:
                        violations.append(Violation(
                            rule_number=4,
                            file_path=file_path,
                            line_number=node.lineno,
                            function_name=node.name,
                            description=f"Function '{node.name}' has {func_length} lines (limit: {NASA_MAX_FUNCTION_LENGTH})",
                            severity=8,
                            code_snippet=f"def {node.name}(...): # {func_length} lines"
                        ))

            # Rule 7: Unchecked return values
            for node in ast.walk(tree):
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    func_name = "unknown"
                    if isinstance(node.value.func, ast.Name):
                        func_name = node.value.func.id
                    elif isinstance(node.value.func, ast.Attribute):
                        func_name = node.value.func.attr

                    if func_name not in ['print', 'append', 'extend']:
                        violations.append(Violation(
                            rule_number=7,
                            file_path=file_path,
                            line_number=node.lineno,
                            function_name="",
                            description=f"Return value of '{func_name}' not checked",
                            severity=7,
                            code_snippet=f"{func_name}(...)"
                        ))

            # Rule 10: Bare except clauses
            for i, line in enumerate(lines, 1):
                if 'except:' in line and 'except Exception:' not in line:
                    violations.append(Violation(
                        rule_number=10,
                        file_path=file_path,
                        line_number=i,
                        function_name="",
                        description="Bare except clause",
                        severity=7,
                        code_snippet=line.strip()
                    ))

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")

        return violations

    def analyze_directory(self, dir_path: str) -> List[Violation]:
        """Analyze all Python files in directory"""
        violations = []

        for py_file in Path(dir_path).rglob('*.py'):
            if any(skip in str(py_file) for skip in ['__pycache__', '.git', 'test']):
                continue

            file_violations = self.analyze_file(str(py_file))
            violations.extend(file_violations)

        return violations

def calculate_compliance_score(violations: List[Violation], total_files: int) -> float:
    """Calculate simple compliance score"""
    if total_files == 0:
        return 1.0

    # Simple scoring: reduce score based on violations
    violation_penalty = len(violations) * 0.05  # 5% penalty per violation
    score = max(0.0, 1.0 - violation_penalty)
    return score

def generate_compliance_report(project_path: str) -> Dict[str, Any]:
    """Generate comprehensive compliance report"""
    print(f"Analyzing project: {project_path}")

    analyzer = SimpleNASAAnalyzer()
    violations = analyzer.analyze_directory(project_path)

    # Count Python files
    python_files = list(Path(project_path).rglob('*.py'))
    python_files = [f for f in python_files if not any(skip in str(f) for skip in ['__pycache__', '.git'])]

    compliance_score = calculate_compliance_score(violations, len(python_files))

    # Group violations by rule
    rule_violations = {}
    for violation in violations:
        rule_num = violation.rule_number
        if rule_num not in rule_violations:
            rule_violations[rule_num] = []
        rule_violations[rule_num].append(violation)

    # Determine compliance level
    if compliance_score >= 0.95:
        level = "EXCELLENT"
    elif compliance_score >= NASA_POT10_TARGET_COMPLIANCE_THRESHOLD:
        level = "GOOD"
    elif compliance_score >= 0.85:
        level = "ACCEPTABLE"
    elif compliance_score >= 0.70:
        level = "POOR"
    else:
        level = "CRITICAL"

    report = {
        'timestamp': datetime.now().isoformat(),
        'project_path': project_path,
        'compliance_score': compliance_score,
        'compliance_percentage': f"{compliance_score:.1%}",
        'compliance_level': level,
        'target_threshold': NASA_POT10_TARGET_COMPLIANCE_THRESHOLD,
        'gate_passed': compliance_score >= NASA_POT10_TARGET_COMPLIANCE_THRESHOLD,
        'total_files': len(python_files),
        'total_violations': len(violations),
        'violations_by_rule': {str(k): len(v) for k, v in rule_violations.items()},
        'top_violations': [
            {
                'rule': v.rule_number,
                'file': v.file_path,
                'line': v.line_number,
                'description': v.description,
                'severity': v.severity
            }
            for v in sorted(violations, key=lambda x: x.severity, reverse=True)[:10]
        ]
    }

    return report

def main():
    """Main function"""
    print("NASA POT10 Compliance Report Generator")
    print("=" * 50)

    # Get project path
    project_root = Path.cwd()
    src_dir = project_root / "src"

    if src_dir.exists():
        target_dir = str(src_dir)
        print(f"Analyzing source directory: {target_dir}")
    else:
        target_dir = str(project_root)
        print(f"Analyzing project root: {target_dir}")

    # Generate report
    report = generate_compliance_report(target_dir)

    # Display summary
    print(f"\nCOMPLIANCE SUMMARY")
    print(f"==================")
    print(f"Overall Score: {report['compliance_percentage']}")
    print(f"Compliance Level: {report['compliance_level']}")
    print(f"Gate Status: {'PASS' if report['gate_passed'] else 'FAIL'}")
    print(f"Total Files: {report['total_files']}")
    print(f"Total Violations: {report['total_violations']}")
    print()

    # Show violations by rule
    if report['violations_by_rule']:
        print("VIOLATIONS BY RULE:")
        rule_descriptions = {
            1: "Control flow complexity",
            2: "Loop bounds",
            3: "Dynamic allocation",
            4: "Function length",
            5: "Assertion density",
            6: "Variable scope",
            7: "Return value checking",
            8: "Import/preprocessor usage",
            9: "Attribute/pointer dereferencing",
            10: "Compilation warnings"
        }

        for rule_num, count in report['violations_by_rule'].items():
            rule_desc = rule_descriptions.get(int(rule_num), "Unknown rule")
            print(f"  Rule {rule_num}: {count} violations - {rule_desc}")
        print()

    # Show top violations
    if report['top_violations']:
        print("TOP VIOLATIONS:")
        for i, violation in enumerate(report['top_violations'][:5], 1):
            file_name = Path(violation['file']).name
            print(f"  {i}. Rule {violation['rule']} in {file_name}:{violation['line']}")
            print(f"     {violation['description']} (severity: {violation['severity']}/10)")
        print()

    # Recommendations
    print("IMPROVEMENT RECOMMENDATIONS:")
    if report['compliance_level'] == "CRITICAL":
        print("  - URGENT: Address all high-severity violations immediately")
        print("  - Focus on Rules 4 (function length) and 7 (return checking)")
        print("  - Implement code review process before any deployments")
    elif report['compliance_level'] == "POOR":
        print("  - Add automated linting to CI/CD pipeline")
        print("  - Break down long functions into smaller components")
        print("  - Add explicit return value checking")
    elif report['compliance_level'] == "ACCEPTABLE":
        print("  - Fine-tune remaining violations to reach target threshold")
        print("  - Add more assertions for verification")
    else:
        print("  - Maintain current standards")
        print("  - Consider implementing compliance certificates")
    print()

    # Save report
    report_file = project_root / ".security" / "nasa_compliance_report.json"
    report_file.parent.mkdir(exist_ok=True)

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Detailed report saved to: {report_file}")

    # Exit with appropriate code
    return 0 if report['gate_passed'] else 1

if __name__ == "__main__":
    sys.exit(main())