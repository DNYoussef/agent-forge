"""
NASA POT10 Compliance Gate
==========================

Automated compliance gate system that enforces NASA POT10 standards
in CI/CD pipelines and pre-commit hooks. Blocks deployments and commits
that don't meet compliance thresholds.

Features:
- Pre-commit hook integration
- CI/CD pipeline gates
- Deployment blocking
- Compliance certificates
- Automated remediation suggestions
- Integration with external tools
"""

import os
import sys
import json
import subprocess
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

from .nasa_pot10_analyzer import create_nasa_pot10_analyzer, POTViolation
from .compliance_scorer import create_compliance_scorer, ComplianceReport, count_lines_of_code
from ..constants.base import (
    NASA_POT10_TARGET_COMPLIANCE_THRESHOLD,
    NASA_POT10_MINIMUM_COMPLIANCE_THRESHOLD
)


class GateResult:
    """Result of a compliance gate check"""

    def __init__(self, passed: bool, score: float, message: str,
                 violations: List[POTViolation], report: Optional[ComplianceReport] = None):
        self.passed = passed
        self.score = score
        self.message = message
        self.violations = violations
        self.report = report
        self.timestamp = datetime.now()


@dataclass
class GateConfiguration:
    """Configuration for compliance gates"""
    enabled: bool = True
    target_threshold: float = NASA_POT10_TARGET_COMPLIANCE_THRESHOLD
    minimum_threshold: float = NASA_POT10_MINIMUM_COMPLIANCE_THRESHOLD
    blocking_rules: List[int] = None  # Rules that block even at low severity
    warning_rules: List[int] = None   # Rules that only warn
    excluded_files: List[str] = None
    excluded_directories: List[str] = None
    max_violations_per_file: int = 10
    certificate_required: bool = True
    auto_fix_enabled: bool = False

    def __post_init__(self):
        if self.blocking_rules is None:
            # Critical rules that always block
            self.blocking_rules = [1, 2, 5, 7]  # Control flow, loops, assertions, return checking
        if self.warning_rules is None:
            # Less critical rules that warn but don't block
            self.warning_rules = [6, 8, 9, 10]  # Scope, preprocessor, pointers, warnings
        if self.excluded_files is None:
            self.excluded_files = [
                '*/test_*.py', '*/tests/*', '*/__pycache__/*',
                '*/build/*', '*/dist/*', '*/.venv/*'
            ]
        if self.excluded_directories is None:
            self.excluded_directories = [
                '__pycache__', '.git', '.pytest_cache', 'node_modules',
                '.venv', 'venv', 'build', 'dist', '.tox'
            ]


class ComplianceGate:
    """
    NASA POT10 Compliance Gate System

    Enforces compliance standards in development workflow
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_configuration(config_path)
        self.analyzer = create_nasa_pot10_analyzer()
        self.scorer = create_compliance_scorer()
        self.logger = self._setup_logging()

        # Ensure security directory exists
        self.security_dir = Path(".security")
        self.security_dir.mkdir(exist_ok=True)

    def _load_configuration(self, config_path: Optional[str]) -> GateConfiguration:
        """Load gate configuration from file or use defaults"""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                return GateConfiguration(**config_data)
            except Exception as e:
                logging.warning(f"Could not load config from {config_path}: {e}")

        return GateConfiguration()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for compliance gate"""
        logger = logging.getLogger('compliance_gate')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # File handler
            log_file = self.security_dir / "compliance_gate.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def check_compliance(self, target_path: str, mode: str = "strict") -> GateResult:
        """
        Check compliance for given path

        Args:
            target_path: Path to analyze (file or directory)
            mode: Analysis mode ("strict", "warning", "info")

        Returns:
            GateResult with pass/fail status and details
        """
        self.logger.info(f"Starting compliance check for {target_path} in {mode} mode")

        if not self.config.enabled:
            return GateResult(
                passed=True,
                score=1.0,
                message="Compliance gate disabled",
                violations=[]
            )

        try:
            start_time = datetime.now()

            # Analyze violations
            if Path(target_path).is_file():
                violations = self.analyzer.analyze_file(target_path)
                analyzed_files = [target_path]
            else:
                violations = self.analyzer.analyze_directory(
                    target_path, self.config.excluded_directories
                )
                analyzed_files = list(Path(target_path).rglob('*.py'))
                analyzed_files = [str(f) for f in analyzed_files
                                if not self._is_excluded_file(str(f))]

            # Filter violations based on mode and configuration
            filtered_violations = self._filter_violations(violations, mode)

            # Calculate compliance scores
            file_compliances = []
            for file_path in analyzed_files:
                file_violations = [v for v in filtered_violations if v.file_path == file_path]
                loc = count_lines_of_code(file_path)
                file_compliance = self.scorer.calculate_file_compliance(
                    file_path, file_violations, loc
                )
                file_compliances.append(file_compliance)

            analysis_duration = (datetime.now() - start_time).total_seconds()

            # Create comprehensive report
            report = self.scorer.create_compliance_report(
                target_path, filtered_violations, file_compliances, analysis_duration
            )

            # Determine gate result
            gate_result = self._evaluate_gate_result(report, mode)
            gate_result.report = report

            # Log results
            self.logger.info(f"Compliance check completed: {gate_result.message}")
            self.logger.info(f"Score: {gate_result.score:.1%}, Violations: {len(gate_result.violations)}")

            # Save gate result
            self._save_gate_result(gate_result, target_path)

            return gate_result

        except Exception as e:
            self.logger.error(f"Compliance check failed: {e}")
            return GateResult(
                passed=False,
                score=0.0,
                message=f"Compliance check failed: {str(e)}",
                violations=[]
            )

    def _filter_violations(self, violations: List[POTViolation], mode: str) -> List[POTViolation]:
        """Filter violations based on mode and configuration"""
        filtered = []

        for violation in violations:
            # Skip excluded files
            if self._is_excluded_file(violation.file_path):
                continue

            # Apply mode-specific filtering
            if mode == "info":
                # Include all violations for informational purposes
                filtered.append(violation)
            elif mode == "warning":
                # Include only blocking rules or high severity
                if (violation.rule_number in self.config.blocking_rules or
                    violation.severity >= 7):
                    filtered.append(violation)
            elif mode == "strict":
                # Include all violations except warning-only rules with low severity
                if not (violation.rule_number in self.config.warning_rules and
                       violation.severity < 5):
                    filtered.append(violation)

        return filtered

    def _is_excluded_file(self, file_path: str) -> bool:
        """Check if file should be excluded from analysis"""
        for pattern in self.config.excluded_files:
            if self._matches_pattern(file_path, pattern):
                return True
        return False

    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Simple pattern matching for file exclusions"""
        import fnmatch
        return fnmatch.fnmatch(path, pattern)

    def _evaluate_gate_result(self, report: ComplianceReport, mode: str) -> GateResult:
        """Evaluate whether gate should pass or fail"""
        score = report.project_compliance.overall_compliance_score
        violations = report.top_violations

        # Check critical violations (always block)
        critical_violations = [
            v for v in violations
            if v.rule_number in self.config.blocking_rules and v.severity >= 8
        ]

        if critical_violations and mode == "strict":
            return GateResult(
                passed=False,
                score=score,
                message=f"❌ BLOCKED: {len(critical_violations)} critical violations found",
                violations=violations
            )

        # Check compliance threshold
        if mode == "strict":
            threshold = self.config.target_threshold
        else:
            threshold = self.config.minimum_threshold

        if score >= threshold:
            return GateResult(
                passed=True,
                score=score,
                message=f"✅ PASS: Compliance at {score:.1%} (threshold: {threshold:.1%})",
                violations=violations
            )
        else:
            if mode == "warning":
                return GateResult(
                    passed=True,  # Pass with warning
                    score=score,
                    message=f"⚠️  WARNING: Below threshold at {score:.1%} (threshold: {threshold:.1%})",
                    violations=violations
                )
            else:
                return GateResult(
                    passed=False,
                    score=score,
                    message=f"❌ FAIL: Below threshold at {score:.1%} (threshold: {threshold:.1%})",
                    violations=violations
                )

    def _save_gate_result(self, gate_result: GateResult, target_path: str):
        """Save gate result to file"""
        result_file = self.security_dir / "latest_gate_result.json"

        result_data = {
            'timestamp': gate_result.timestamp.isoformat(),
            'target_path': target_path,
            'passed': gate_result.passed,
            'score': gate_result.score,
            'message': gate_result.message,
            'violation_count': len(gate_result.violations),
            'top_violations': [
                {
                    'rule': v.rule_number,
                    'type': v.violation_type.value,
                    'file': v.file_path,
                    'line': v.line_number,
                    'severity': v.severity,
                    'description': v.description
                }
                for v in gate_result.violations[:5]  # Top 5 violations
            ]
        }

        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)

    def install_pre_commit_hook(self, repo_path: str = ".") -> bool:
        """Install pre-commit hook for NASA POT10 compliance"""
        try:
            git_hooks_dir = Path(repo_path) / ".git" / "hooks"
            if not git_hooks_dir.exists():
                self.logger.error("Not a git repository or .git/hooks directory not found")
                return False

            pre_commit_hook = git_hooks_dir / "pre-commit"

            # Create pre-commit hook script
            hook_script = f'''#!/bin/bash
# NASA POT10 Compliance Pre-commit Hook
# Auto-generated by agent-forge compliance gate

echo "🔍 Running NASA POT10 compliance check..."

# Get list of staged Python files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep "\\.py$" || true)

if [ -z "$STAGED_FILES" ]; then
    echo "No Python files staged for commit"
    exit 0
fi

# Create temporary directory for analysis
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Copy staged files to temp directory
for FILE in $STAGED_FILES; do
    TEMP_FILE="$TEMP_DIR/$FILE"
    mkdir -p "$(dirname "$TEMP_FILE")"
    git show ":$FILE" > "$TEMP_FILE"
done

# Run compliance check
python -c "
import sys
sys.path.append('{Path(__file__).parent.parent.absolute()}')
from src.security.compliance_gate import ComplianceGate

gate = ComplianceGate()
result = gate.check_compliance('$TEMP_DIR', mode='strict')

print(result.message)
if result.violations:
    print('\\nTop violations:')
    for v in result.violations[:5]:
        print(f'  Rule {{v.rule_number}}: {{v.description}} ({{v.file_path}}:{{v.line_number}})')

if not result.passed:
    print('\\n💡 Run the following to see all violations:')
    print('python -m src.security.compliance_gate check .')
    sys.exit(1)
"

echo "✅ Compliance check passed"
'''

            # Write hook script
            with open(pre_commit_hook, 'w') as f:
                f.write(hook_script)

            # Make executable
            os.chmod(pre_commit_hook, 0o755)

            self.logger.info(f"Pre-commit hook installed at {pre_commit_hook}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to install pre-commit hook: {e}")
            return False

    def create_ci_script(self, output_path: str = ".security/ci_compliance_check.sh") -> bool:
        """Create CI/CD compliance check script"""
        try:
            script_path = Path(output_path)
            script_path.parent.mkdir(parents=True, exist_ok=True)

            ci_script = f'''#!/bin/bash
# NASA POT10 Compliance CI/CD Check
# Auto-generated by agent-forge compliance gate

set -e

echo "🔍 Running NASA POT10 compliance check in CI/CD..."

# Install dependencies if needed
if [ ! -f "requirements.txt" ] || ! pip show -q ast; then
    echo "Installing analysis dependencies..."
    pip install ast-deps || true
fi

# Run compliance check
python -c "
import sys
import os
sys.path.insert(0, os.getcwd())

from src.security.compliance_gate import ComplianceGate

gate = ComplianceGate()
result = gate.check_compliance('.', mode='strict')

print(f'Compliance Score: {{result.score:.1%}}')
print(f'Status: {{result.message}}')

if result.violations:
    print('\\nViolations found:')
    for i, v in enumerate(result.violations[:10], 1):
        print(f'{{i:2d}}. Rule {{v.rule_number}}: {{v.description}}')
        print(f'     File: {{v.file_path}}:{{v.line_number}}')
        print(f'     Severity: {{v.severity}}/10')
        print()

# Export results for CI/CD artifacts
if result.report:
    import json
    from datetime import datetime

    ci_report = {{
        'timestamp': datetime.now().isoformat(),
        'compliance_score': result.score,
        'passed': result.passed,
        'total_violations': len(result.violations),
        'gate_message': result.message,
        'threshold': gate.config.target_threshold
    }}

    with open('.security/ci_compliance_report.json', 'w') as f:
        json.dump(ci_report, f, indent=2)

    print('\\n📊 Detailed report saved to .security/ci_compliance_report.json')

if not result.passed:
    print('\\n❌ CI/CD GATE FAILURE: Compliance check failed')
    print('Fix violations before merging/deploying')
    sys.exit(1)
else:
    print('\\n✅ CI/CD GATE PASSED: Compliance check successful')
"

echo "Compliance check completed"
'''

            with open(script_path, 'w') as f:
                f.write(ci_script)

            os.chmod(script_path, 0o755)

            self.logger.info(f"CI/CD script created at {script_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create CI script: {e}")
            return False

    def generate_compliance_certificate(self, gate_result: GateResult,
                                      output_path: Optional[str] = None) -> str:
        """Generate compliance certificate for passed gates"""
        if not gate_result.passed:
            raise ValueError("Cannot generate certificate for failed compliance check")

        if output_path is None:
            output_path = self.security_dir / f"compliance_certificate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        certificate = {
            'certificate_id': f"NASA_POT10_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'standard': 'NASA Power of Ten Rules',
            'compliance_score': gate_result.score,
            'threshold_met': gate_result.score >= self.config.target_threshold,
            'certification_timestamp': gate_result.timestamp.isoformat(),
            'validity_period_days': 30,  # Certificates expire after 30 days
            'analysis_summary': {
                'violations_found': len(gate_result.violations),
                'critical_violations': len([v for v in gate_result.violations if v.severity >= 8]),
                'files_analyzed': len(set(v.file_path for v in gate_result.violations)) if gate_result.violations else 0
            },
            'compliance_statement': (
                f"This certificate confirms that the analyzed code meets NASA POT10 "
                f"compliance standards with a score of {gate_result.score:.1%} as of "
                f"{gate_result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}."
            ),
            'issuer': 'agent-forge NASA POT10 Compliance Gate',
            'signature': self._generate_certificate_signature(gate_result)
        }

        with open(output_path, 'w') as f:
            json.dump(certificate, f, indent=2)

        self.logger.info(f"Compliance certificate generated: {output_path}")
        return str(output_path)

    def _generate_certificate_signature(self, gate_result: GateResult) -> str:
        """Generate a simple signature for the certificate"""
        import hashlib

        # Create signature from key components
        signature_data = f"{gate_result.score}_{gate_result.timestamp.isoformat()}_{len(gate_result.violations)}"
        signature = hashlib.sha256(signature_data.encode()).hexdigest()[:16]
        return f"POT10_{signature}"

    def auto_fix_violations(self, violations: List[POTViolation]) -> Dict[str, Any]:
        """Attempt to automatically fix simple violations"""
        if not self.config.auto_fix_enabled:
            return {'status': 'disabled', 'fixes_applied': 0}

        fixes_applied = 0
        fixes_attempted = 0

        # Group violations by file
        file_violations = {}
        for violation in violations:
            if violation.file_path not in file_violations:
                file_violations[violation.file_path] = []
            file_violations[violation.file_path].append(violation)

        for file_path, file_viols in file_violations.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                original_content = content

                # Apply simple fixes
                for violation in file_viols:
                    fixes_attempted += 1

                    if violation.rule_number == 10:  # Compilation warnings
                        if 'except:' in violation.code_snippet:
                            content = content.replace('except:', 'except Exception:')
                            fixes_applied += 1

                    # Add more auto-fix rules here

                # Save if changes were made
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.logger.info(f"Auto-fixed violations in {file_path}")

            except Exception as e:
                self.logger.warning(f"Could not auto-fix {file_path}: {e}")

        return {
            'status': 'completed',
            'fixes_applied': fixes_applied,
            'fixes_attempted': fixes_attempted,
            'success_rate': fixes_applied / max(fixes_attempted, 1)
        }


def create_compliance_gate(config_path: Optional[str] = None) -> ComplianceGate:
    """Create and return a compliance gate instance"""
    return ComplianceGate(config_path)


# CLI Interface
def main():
    """CLI interface for compliance gate"""
    import argparse

    parser = argparse.ArgumentParser(description='NASA POT10 Compliance Gate')
    parser.add_argument('command', choices=['check', 'install-hook', 'create-ci', 'certificate'],
                       help='Command to execute')
    parser.add_argument('target', nargs='?', default='.',
                       help='Target path to analyze (default: current directory)')
    parser.add_argument('--mode', choices=['strict', 'warning', 'info'], default='strict',
                       help='Analysis mode (default: strict)')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    gate = create_compliance_gate(args.config)

    if args.command == 'check':
        result = gate.check_compliance(args.target, args.mode)
        print(result.message)

        if result.violations:
            print(f"\nFound {len(result.violations)} violations:")
            for i, violation in enumerate(result.violations[:10], 1):
                print(f"{i:2d}. Rule {violation.rule_number}: {violation.description}")
                print(f"     {violation.file_path}:{violation.line_number} (severity: {violation.severity}/10)")

        sys.exit(0 if result.passed else 1)

    elif args.command == 'install-hook':
        success = gate.install_pre_commit_hook(args.target)
        print("✅ Pre-commit hook installed" if success else "❌ Failed to install pre-commit hook")
        sys.exit(0 if success else 1)

    elif args.command == 'create-ci':
        output_path = args.output or ".security/ci_compliance_check.sh"
        success = gate.create_ci_script(output_path)
        print(f"✅ CI script created at {output_path}" if success else "❌ Failed to create CI script")
        sys.exit(0 if success else 1)

    elif args.command == 'certificate':
        result = gate.check_compliance(args.target, 'strict')
        if result.passed:
            cert_path = gate.generate_compliance_certificate(result, args.output)
            print(f"✅ Compliance certificate generated: {cert_path}")
        else:
            print("❌ Cannot generate certificate: compliance check failed")
            sys.exit(1)


if __name__ == "__main__":
    main()

"""
NASA POT10 Compliance Gate - Production Implementation
=====================================================

This module provides automated compliance gates for NASA POT10 standards with:

- Pre-commit hook integration to prevent non-compliant commits
- CI/CD pipeline gates that block deployments below threshold
- Compliance certificate generation for passed checks
- Configurable thresholds and violation filtering
- Automatic violation fixing for simple issues
- Integration with existing development workflows

The gate system enforces quality standards by:
1. Analyzing code changes before they're committed
2. Blocking CI/CD pipelines if compliance drops below threshold
3. Generating certificates for compliant code
4. Providing actionable feedback for violations

Usage:
    # Install pre-commit hook
    python -m src.security.compliance_gate install-hook

    # Check compliance
    python -m src.security.compliance_gate check .

    # Generate certificate
    python -m src.security.compliance_gate certificate
"""