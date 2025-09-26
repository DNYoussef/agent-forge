"""
NASA POT10 Compliance Scorer
============================

This module calculates compliance scores for NASA POT10 rules based on
violation analysis. It provides weighted scoring, trend tracking, and
detailed reporting capabilities.

Features:
- Weighted rule scoring based on criticality
- File-level and project-level compliance calculation
- Trend analysis over time
- Detailed violation reports
- Integration with automated gates
"""

import json
import sqlite3
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from enum import Enum

from .nasa_pot10_analyzer import POTViolation, POTViolationType, NASAPOT10Analyzer
from ..constants.base import (
    NASA_POT10_TARGET_COMPLIANCE_THRESHOLD,
    NASA_POT10_MINIMUM_COMPLIANCE_THRESHOLD
)


class ComplianceLevel(Enum):
    """NASA POT10 Compliance levels"""
    EXCELLENT = "excellent"      # >= 95%
    GOOD = "good"               # >= 92%
    ACCEPTABLE = "acceptable"   # >= 85%
    POOR = "poor"              # >= 70%
    CRITICAL = "critical"      # < 70%


@dataclass
class FileCompliance:
    """Compliance data for a single file"""
    file_path: str
    total_violations: int
    rule_violations: Dict[int, int]
    compliance_score: float
    compliance_level: ComplianceLevel
    lines_of_code: int
    violations_per_kloc: float
    timestamp: datetime


@dataclass
class ProjectCompliance:
    """Overall project compliance data"""
    project_path: str
    total_files: int
    analyzed_files: int
    total_violations: int
    overall_compliance_score: float
    compliance_level: ComplianceLevel
    rule_compliance_scores: Dict[int, float]
    file_compliances: List[FileCompliance]
    trend_data: List[Dict[str, Any]]
    timestamp: datetime
    analysis_duration_seconds: float


@dataclass
class ComplianceReport:
    """Comprehensive compliance report"""
    project_compliance: ProjectCompliance
    top_violations: List[POTViolation]
    improvement_recommendations: List[str]
    compliance_trend: List[Dict[str, Any]]
    gate_status: Dict[str, Any]


class ComplianceDatabase:
    """SQLite database for storing compliance history"""

    def __init__(self, db_path: str = ".security/compliance_history.db"):
        self.db_path = db_path
        self._ensure_database()

    def _ensure_database(self):
        """Create database tables if they don't exist"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    project_path TEXT NOT NULL,
                    compliance_score REAL NOT NULL,
                    total_violations INTEGER NOT NULL,
                    total_files INTEGER NOT NULL,
                    rule_scores TEXT NOT NULL,  -- JSON
                    metadata TEXT  -- JSON
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_compliance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    compliance_score REAL NOT NULL,
                    violations INTEGER NOT NULL,
                    lines_of_code INTEGER NOT NULL,
                    rule_violations TEXT NOT NULL  -- JSON
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_compliance_timestamp
                ON compliance_history(timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_compliance_timestamp
                ON file_compliance_history(timestamp)
            """)

    def store_compliance(self, project_compliance: ProjectCompliance):
        """Store project compliance data"""
        with sqlite3.connect(self.db_path) as conn:
            # Store project compliance
            conn.execute("""
                INSERT INTO compliance_history
                (timestamp, project_path, compliance_score, total_violations,
                 total_files, rule_scores, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                project_compliance.timestamp.isoformat(),
                project_compliance.project_path,
                project_compliance.overall_compliance_score,
                project_compliance.total_violations,
                project_compliance.total_files,
                json.dumps(project_compliance.rule_compliance_scores),
                json.dumps({
                    'analyzed_files': project_compliance.analyzed_files,
                    'compliance_level': project_compliance.compliance_level.value,
                    'analysis_duration': project_compliance.analysis_duration_seconds
                })
            ))

            # Store file compliance data
            for file_comp in project_compliance.file_compliances:
                conn.execute("""
                    INSERT INTO file_compliance_history
                    (timestamp, file_path, compliance_score, violations,
                     lines_of_code, rule_violations)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    file_comp.timestamp.isoformat(),
                    file_comp.file_path,
                    file_comp.compliance_score,
                    file_comp.total_violations,
                    file_comp.lines_of_code,
                    json.dumps(file_comp.rule_violations)
                ))

    def get_compliance_trend(self, project_path: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get compliance trend data for the last N days"""
        since_date = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, compliance_score, total_violations, rule_scores
                FROM compliance_history
                WHERE project_path = ? AND timestamp >= ?
                ORDER BY timestamp
            """, (project_path, since_date.isoformat()))

            trend_data = []
            for row in cursor.fetchall():
                trend_data.append({
                    'timestamp': row[0],
                    'compliance_score': row[1],
                    'total_violations': row[2],
                    'rule_scores': json.loads(row[3])
                })

            return trend_data

    def get_latest_compliance(self, project_path: str) -> Optional[Dict[str, Any]]:
        """Get the most recent compliance data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, compliance_score, total_violations,
                       total_files, rule_scores, metadata
                FROM compliance_history
                WHERE project_path = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (project_path,))

            row = cursor.fetchone()
            if row:
                return {
                    'timestamp': row[0],
                    'compliance_score': row[1],
                    'total_violations': row[2],
                    'total_files': row[3],
                    'rule_scores': json.loads(row[4]),
                    'metadata': json.loads(row[5])
                }
            return None


class ComplianceScorer:
    """
    NASA POT10 Compliance Scorer

    Calculates weighted compliance scores based on rule violations
    and provides detailed analysis and reporting.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.database = ComplianceDatabase()
        self.logger = logging.getLogger(__name__)

        # Rule weights based on criticality for mission-critical software
        self.rule_weights = self.config.get('rule_weights', {
            1: 0.15,  # Control flow - Critical for reliability
            2: 0.20,  # Loop bounds - Critical for determinism
            3: 0.10,  # Dynamic allocation - Important for predictability
            4: 0.10,  # Function length - Important for maintainability
            5: 0.15,  # Assertions - Critical for verification
            6: 0.05,  # Scope - Important for clarity
            7: 0.15,  # Return checking - Critical for error handling
            8: 0.05,  # Preprocessor - Important for simplicity
            9: 0.05,  # Pointer deref - Important for safety
            10: 0.05  # Warnings - Important for quality
        })

        # Severity multipliers
        self.severity_multipliers = {
            1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5,
            6: 0.6, 7: 0.7, 8: 0.8, 9: 0.9, 10: 1.0
        }

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load config from {config_path}: {e}")

        return {}

    def calculate_file_compliance(self, file_path: str, violations: List[POTViolation],
                                lines_of_code: int) -> FileCompliance:
        """Calculate compliance score for a single file"""

        # Count violations by rule
        rule_violations = {}
        total_violation_score = 0.0

        for violation in violations:
            rule_num = violation.rule_number
            rule_violations[rule_num] = rule_violations.get(rule_num, 0) + 1

            # Calculate weighted violation score
            rule_weight = self.rule_weights.get(rule_num, 0.1)
            severity_multiplier = self.severity_multipliers.get(violation.severity, 0.5)
            violation_score = rule_weight * severity_multiplier
            total_violation_score += violation_score

        # Calculate compliance score
        # Base score starts at 1.0, violations reduce it
        max_possible_score = sum(self.rule_weights.values())
        compliance_score = max(0.0, 1.0 - (total_violation_score / max_possible_score))

        # Determine compliance level
        compliance_level = self._get_compliance_level(compliance_score)

        # Calculate violations per KLOC
        violations_per_kloc = (len(violations) / max(lines_of_code, 1)) * 1000

        return FileCompliance(
            file_path=file_path,
            total_violations=len(violations),
            rule_violations=rule_violations,
            compliance_score=compliance_score,
            compliance_level=compliance_level,
            lines_of_code=lines_of_code,
            violations_per_kloc=violations_per_kloc,
            timestamp=datetime.now()
        )

    def calculate_project_compliance(self, project_path: str,
                                   file_compliances: List[FileCompliance],
                                   analysis_duration: float) -> ProjectCompliance:
        """Calculate overall project compliance"""

        if not file_compliances:
            return ProjectCompliance(
                project_path=project_path,
                total_files=0,
                analyzed_files=0,
                total_violations=0,
                overall_compliance_score=0.0,
                compliance_level=ComplianceLevel.CRITICAL,
                rule_compliance_scores={},
                file_compliances=[],
                trend_data=[],
                timestamp=datetime.now(),
                analysis_duration_seconds=analysis_duration
            )

        # Calculate aggregated metrics
        total_violations = sum(fc.total_violations for fc in file_compliances)
        total_lines = sum(fc.lines_of_code for fc in file_compliances)

        # Weighted average compliance score based on lines of code
        weighted_score_sum = sum(fc.compliance_score * fc.lines_of_code
                               for fc in file_compliances)
        overall_compliance_score = weighted_score_sum / max(total_lines, 1)

        # Calculate rule-specific compliance scores
        rule_compliance_scores = {}
        for rule_num in range(1, 11):
            rule_violations = sum(fc.rule_violations.get(rule_num, 0)
                                for fc in file_compliances)
            # Simple metric: files without violations for this rule
            compliant_files = sum(1 for fc in file_compliances
                                if fc.rule_violations.get(rule_num, 0) == 0)
            rule_compliance_scores[rule_num] = compliant_files / len(file_compliances)

        # Get trend data
        trend_data = self.database.get_compliance_trend(project_path)

        compliance_level = self._get_compliance_level(overall_compliance_score)

        project_compliance = ProjectCompliance(
            project_path=project_path,
            total_files=len(file_compliances),
            analyzed_files=len(file_compliances),
            total_violations=total_violations,
            overall_compliance_score=overall_compliance_score,
            compliance_level=compliance_level,
            rule_compliance_scores=rule_compliance_scores,
            file_compliances=file_compliances,
            trend_data=trend_data,
            timestamp=datetime.now(),
            analysis_duration_seconds=analysis_duration
        )

        # Store in database
        self.database.store_compliance(project_compliance)

        return project_compliance

    def _get_compliance_level(self, score: float) -> ComplianceLevel:
        """Determine compliance level from score"""
        if score >= 0.95:
            return ComplianceLevel.EXCELLENT
        elif score >= NASA_POT10_TARGET_COMPLIANCE_THRESHOLD:
            return ComplianceLevel.GOOD
        elif score >= NASA_POT10_MINIMUM_COMPLIANCE_THRESHOLD:
            return ComplianceLevel.ACCEPTABLE
        elif score >= 0.70:
            return ComplianceLevel.POOR
        else:
            return ComplianceLevel.CRITICAL

    def generate_improvement_recommendations(self, project_compliance: ProjectCompliance,
                                           violations: List[POTViolation]) -> List[str]:
        """Generate actionable improvement recommendations"""
        recommendations = []

        # Analyze most common violations
        violation_counts = {}
        for violation in violations:
            key = (violation.rule_number, violation.violation_type.value)
            violation_counts[key] = violation_counts.get(key, 0) + 1

        # Sort by frequency
        sorted_violations = sorted(violation_counts.items(),
                                 key=lambda x: x[1], reverse=True)

        # Generate recommendations for top issues
        for (rule_num, violation_type), count in sorted_violations[:5]:
            rule_weight = self.rule_weights.get(rule_num, 0.1)
            impact = count * rule_weight

            recommendations.append(
                f"Priority: Address {count} instances of Rule {rule_num} violations "
                f"({violation_type}) - Impact score: {impact:.2f}"
            )

        # Add rule-specific recommendations
        for rule_num, score in project_compliance.rule_compliance_scores.items():
            if score < 0.8:  # Poor compliance for this rule
                recommendations.append(self._get_rule_recommendation(rule_num, score))

        # Add general recommendations based on compliance level
        if project_compliance.compliance_level == ComplianceLevel.CRITICAL:
            recommendations.append(
                "CRITICAL: Implement immediate code review process focusing on "
                "Rules 1, 2, 5, and 7 (control flow, loop bounds, assertions, return checking)"
            )
        elif project_compliance.compliance_level == ComplianceLevel.POOR:
            recommendations.append(
                "Establish automated linting and static analysis in CI/CD pipeline"
            )

        return recommendations

    def _get_rule_recommendation(self, rule_num: int, score: float) -> str:
        """Get specific recommendation for a rule"""
        recommendations = {
            1: f"Rule 1 ({score:.1%} compliant): Simplify control flow - avoid deeply nested conditions and complex exception handling",
            2: f"Rule 2 ({score:.1%} compliant): Add explicit bounds to all loops - use range() with fixed limits, avoid while True",
            3: f"Rule 3 ({score:.1%} compliant): Pre-allocate data structures during initialization, avoid dynamic growth in runtime",
            4: f"Rule 4 ({score:.1%} compliant): Split functions > 60 lines into smaller, focused functions with single responsibilities",
            5: f"Rule 5 ({score:.1%} compliant): Add assertions for preconditions, postconditions, and invariants (target: 2% of lines)",
            6: f"Rule 6 ({score:.1%} compliant): Declare variables at smallest possible scope - avoid module-level variables",
            7: f"Rule 7 ({score:.1%} compliant): Always check return values - assign to variables or explicitly handle errors",
            8: f"Rule 8 ({score:.1%} compliant): Use only static imports - avoid dynamic imports and exec/eval statements",
            9: f"Rule 9 ({score:.1%} compliant): Limit attribute/subscript chains - store intermediate values in local variables",
            10: f"Rule 10 ({score:.1%} compliant): Enable all linting warnings and fix them - use strict static analysis tools"
        }
        return recommendations.get(rule_num, f"Rule {rule_num}: Compliance at {score:.1%}")

    def create_compliance_report(self, project_path: str, violations: List[POTViolation],
                               file_compliances: List[FileCompliance],
                               analysis_duration: float) -> ComplianceReport:
        """Create comprehensive compliance report"""

        # Calculate project compliance
        project_compliance = self.calculate_project_compliance(
            project_path, file_compliances, analysis_duration
        )

        # Get top violations by severity
        top_violations = sorted(violations, key=lambda v: v.severity, reverse=True)[:10]

        # Generate recommendations
        recommendations = self.generate_improvement_recommendations(
            project_compliance, violations
        )

        # Get compliance trend
        trend_data = self.database.get_compliance_trend(project_path, days=30)

        # Determine gate status
        gate_status = {
            'passed': project_compliance.overall_compliance_score >= NASA_POT10_TARGET_COMPLIANCE_THRESHOLD,
            'score': project_compliance.overall_compliance_score,
            'threshold': NASA_POT10_TARGET_COMPLIANCE_THRESHOLD,
            'minimum_threshold': NASA_POT10_MINIMUM_COMPLIANCE_THRESHOLD,
            'message': self._get_gate_message(project_compliance)
        }

        return ComplianceReport(
            project_compliance=project_compliance,
            top_violations=top_violations,
            improvement_recommendations=recommendations,
            compliance_trend=trend_data,
            gate_status=gate_status
        )

    def _get_gate_message(self, project_compliance: ProjectCompliance) -> str:
        """Get gate status message"""
        score = project_compliance.overall_compliance_score

        if score >= NASA_POT10_TARGET_COMPLIANCE_THRESHOLD:
            return f"✅ PASS: NASA POT10 compliance at {score:.1%} (target: {NASA_POT10_TARGET_COMPLIANCE_THRESHOLD:.1%})"
        elif score >= NASA_POT10_MINIMUM_COMPLIANCE_THRESHOLD:
            return f"⚠️  WARNING: Below target compliance at {score:.1%} (target: {NASA_POT10_TARGET_COMPLIANCE_THRESHOLD:.1%})"
        else:
            return f"❌ FAIL: Critical compliance failure at {score:.1%} (minimum: {NASA_POT10_MINIMUM_COMPLIANCE_THRESHOLD:.1%})"

    def export_report(self, report: ComplianceReport, output_path: str):
        """Export compliance report to JSON file"""
        # Convert dataclasses to dictionaries for JSON serialization
        report_dict = {
            'project_compliance': asdict(report.project_compliance),
            'top_violations': [asdict(v) for v in report.top_violations],
            'improvement_recommendations': report.improvement_recommendations,
            'compliance_trend': report.compliance_trend,
            'gate_status': report.gate_status,
            'export_timestamp': datetime.now().isoformat()
        }

        # Handle enum serialization
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, 'value'):  # Enum
                return obj.value
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=json_serializer)

    def get_compliance_summary(self, project_path: str) -> Dict[str, Any]:
        """Get a quick compliance summary"""
        latest = self.database.get_latest_compliance(project_path)
        if not latest:
            return {'status': 'no_data', 'message': 'No compliance data available'}

        return {
            'status': 'success',
            'compliance_score': latest['compliance_score'],
            'compliance_percentage': f"{latest['compliance_score']:.1%}",
            'total_violations': latest['total_violations'],
            'total_files': latest['total_files'],
            'timestamp': latest['timestamp'],
            'gate_passed': latest['compliance_score'] >= NASA_POT10_TARGET_COMPLIANCE_THRESHOLD,
            'level': self._get_compliance_level(latest['compliance_score']).value
        }


def count_lines_of_code(file_path: str) -> int:
    """Count non-empty, non-comment lines in a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        count = 0
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                count += 1
        return count
    except Exception:
        return 0


# Factory function for easy instantiation
def create_compliance_scorer(config_path: Optional[str] = None) -> ComplianceScorer:
    """Create and return a compliance scorer instance"""
    return ComplianceScorer(config_path)


if __name__ == "__main__":
    # Example usage
    from .nasa_pot10_analyzer import create_nasa_pot10_analyzer

    # Analyze a project
    analyzer = create_nasa_pot10_analyzer()
    scorer = create_compliance_scorer()

    project_path = "."
    violations = analyzer.analyze_directory(project_path)

    # Calculate file compliances
    file_compliances = []
    for file_path in Path(project_path).rglob('*.py'):
        file_violations = [v for v in violations if v.file_path == str(file_path)]
        loc = count_lines_of_code(str(file_path))
        file_compliance = scorer.calculate_file_compliance(str(file_path), file_violations, loc)
        file_compliances.append(file_compliance)

    # Create report
    report = scorer.create_compliance_report(project_path, violations, file_compliances, 0.0)

    print(f"Project Compliance: {report.project_compliance.overall_compliance_score:.1%}")
    print(f"Gate Status: {report.gate_status['message']}")

"""
NASA POT10 Compliance Scorer - Production Implementation
=======================================================

This module provides comprehensive compliance scoring for NASA POT10 rules with:

- Weighted rule scoring based on mission-critical importance
- File and project-level compliance calculation
- Historical trend tracking with SQLite database
- Detailed violation analysis and reporting
- Actionable improvement recommendations
- Integration with automated quality gates

The scorer uses a sophisticated weighted scoring system where violations
are penalized based on both their rule importance and severity level.
Rule 2 (loop bounds) and Rule 5 (assertions) carry the highest weights
due to their critical importance for mission-critical software.

Usage:
    scorer = create_compliance_scorer()
    report = scorer.create_compliance_report(project_path, violations, file_compliances, duration)
    gate_passed = report.gate_status['passed']
"""