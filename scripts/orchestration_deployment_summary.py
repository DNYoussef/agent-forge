#!/usr/bin/env python3
"""
Orchestration Deployment Summary Generator
Creates comprehensive production readiness report for Agent Forge 8-phase pipeline orchestration system.
"""

import os
import sys
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class OrchestrationDeploymentAnalyzer:
    """Analyzes orchestration system deployment readiness and generates comprehensive report."""

    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent
        self.report_timestamp = datetime.now().isoformat()
        self.deployment_id = f"orch-deploy-{int(time.time())}"

    def analyze_orchestration_components(self) -> Dict[str, Any]:
        """Analyze all orchestration system components."""
        print("[ANALYZING] Orchestration components...")

        components = {
            "phase_orchestrator": {
                "path": "src/orchestration/phase_orchestrator.py",
                "status": "unknown",
                "features": [],
                "size_loc": 0,
                "complexity_score": 0
            },
            "pipeline_controller": {
                "path": "src/orchestration/pipeline_controller.py",
                "status": "unknown",
                "features": [],
                "size_loc": 0,
                "complexity_score": 0
            },
            "phase_validators": {
                "path": "src/orchestration/phase_validators.py",
                "status": "unknown",
                "features": [],
                "size_loc": 0,
                "complexity_score": 0
            },
            "integration_tests": {
                "path": "tests/integration/test_full_pipeline.py",
                "status": "unknown",
                "features": [],
                "size_loc": 0,
                "complexity_score": 0
            },
            "main_pipeline": {
                "path": "main_pipeline.py",
                "status": "unknown",
                "features": [],
                "size_loc": 0,
                "complexity_score": 0
            }
        }

        for component_name, component_info in components.items():
            file_path = self.base_dir / component_info["path"]
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8')
                    component_info["status"] = "available"
                    component_info["size_loc"] = len(content.splitlines())

                    # Analyze features based on content
                    features = self._analyze_component_features(content, component_name)
                    component_info["features"] = features
                    component_info["complexity_score"] = self._calculate_complexity_score(content)

                except Exception as e:
                    component_info["status"] = f"error: {str(e)}"
            else:
                component_info["status"] = "missing"

        return components

    def _analyze_component_features(self, content: str, component_name: str) -> List[str]:
        """Analyze features present in component code."""
        features = []

        # Common orchestration features
        feature_patterns = {
            "async_execution": ["async def", "await ", "asyncio"],
            "state_management": ["State", "PENDING", "RUNNING", "COMPLETED"],
            "dependency_resolution": ["dependencies", "topological", "dependency_graph"],
            "checkpoint_support": ["checkpoint", "resume", "save_checkpoint"],
            "error_handling": ["try:", "except", "Error", "Exception"],
            "progress_tracking": ["progress", "callback", "monitor"],
            "resource_management": ["memory", "cpu", "gpu", "resource"],
            "validation": ["validate", "ValidationResult", "validator"],
            "parallel_execution": ["parallel", "concurrent", "ThreadPool"],
            "performance_monitoring": ["metrics", "performance", "benchmark"]
        }

        # Component-specific features
        if component_name == "phase_orchestrator":
            feature_patterns.update({
                "phase_transitions": ["transition", "next_phase", "phase_state"],
                "data_flow": ["data_flow", "phase_output", "input_data"],
                "recovery_mechanisms": ["recover", "retry", "fallback"]
            })
        elif component_name == "pipeline_controller":
            feature_patterns.update({
                "master_control": ["execute_full_pipeline", "controller", "master"],
                "resource_allocation": ["allocate", "resources", "constraints"],
                "distributed_execution": ["distributed", "worker", "cluster"]
            })
        elif component_name == "phase_validators":
            feature_patterns.update({
                "format_validation": ["format", "schema", "structure"],
                "compatibility_checks": ["compatible", "compatibility", "version"],
                "quality_gates": ["quality", "gate", "threshold"]
            })

        for feature_name, patterns in feature_patterns.items():
            if any(pattern in content for pattern in patterns):
                features.append(feature_name)

        return features

    def _calculate_complexity_score(self, content: str) -> int:
        """Calculate complexity score based on code patterns."""
        score = 0

        # Base complexity indicators
        score += content.count("class ") * 10
        score += content.count("def ") * 5
        score += content.count("async def ") * 8
        score += content.count("if ") * 2
        score += content.count("for ") * 3
        score += content.count("while ") * 3
        score += content.count("try:") * 5
        score += content.count("except") * 4

        return score

    def analyze_phase_status(self) -> Dict[str, Any]:
        """Analyze current phase implementation status."""
        print("[ANALYZING] Phase implementation status...")

        phase_report_path = self.base_dir / "phase_validation_report.json"
        if phase_report_path.exists():
            try:
                with open(phase_report_path, 'r', encoding='utf-8') as f:
                    phase_data = json.load(f)

                return {
                    "validation_timestamp": phase_data.get("validation_info", {}).get("timestamp", "unknown"),
                    "total_phases": phase_data.get("validation_info", {}).get("total_phases", 0),
                    "health_score": phase_data.get("system_status", {}).get("health_score", 0),
                    "operational_phases": phase_data.get("system_status", {}).get("operational_phases", 0),
                    "status_distribution": phase_data.get("system_status", {}).get("status_distribution", {}),
                    "ready_for_production": phase_data.get("system_status", {}).get("ready_for_production", False),
                    "phase_details": phase_data.get("phase_details", {}),
                    "recommendations": phase_data.get("recommendations", [])
                }
            except Exception as e:
                return {"error": f"Failed to load phase report: {str(e)}"}
        else:
            return {"error": "Phase validation report not found"}

    def analyze_integration_evidence(self) -> Dict[str, Any]:
        """Analyze integration test results and evidence."""
        print("[ANALYZING] Integration test evidence...")

        evidence = {
            "test_files_detected": 0,
            "integration_test_results": {},
            "checkpoint_demo_results": {},
            "performance_metrics": {}
        }

        # Check for test files
        test_dirs = ["tests", "test", "testing"]
        for test_dir in test_dirs:
            test_path = self.base_dir / test_dir
            if test_path.exists():
                test_files = list(test_path.rglob("*.py"))
                evidence["test_files_detected"] += len(test_files)

        # Check integration test results
        integration_test_path = self.base_dir / "tests" / "integration" / "test_full_pipeline.py"
        if integration_test_path.exists():
            content = integration_test_path.read_text(encoding='utf-8')
            evidence["integration_test_results"] = {
                "available": True,
                "test_count": content.count("async def test_"),
                "mock_controllers": "MockPhaseController" in content,
                "async_support": "async def" in content
            }

        # Check checkpoint demo results
        checkpoint_demo_path = self.base_dir / "checkpoint_demo"
        if checkpoint_demo_path.exists():
            demo_files = list(checkpoint_demo_path.glob("*.json"))
            if demo_files:
                try:
                    latest_demo = max(demo_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_demo, 'r', encoding='utf-8') as f:
                        demo_data = json.load(f)
                    evidence["checkpoint_demo_results"] = {
                        "available": True,
                        "resilience_score": demo_data.get("summary", {}).get("overall_resilience_score", 0),
                        "scenarios_tested": len(demo_data.get("scenarios", {})),
                        "all_passed": demo_data.get("summary", {}).get("all_scenarios_passed", False)
                    }
                except Exception as e:
                    evidence["checkpoint_demo_results"] = {"error": str(e)}

        # Check performance metrics
        perf_report_path = self.base_dir / "performance_analysis_report.json"
        if perf_report_path.exists():
            try:
                with open(perf_report_path, 'r', encoding='utf-8') as f:
                    perf_data = json.load(f)
                evidence["performance_metrics"] = {
                    "available": True,
                    "orchestration_score": perf_data.get("orchestration_capabilities", {}).get("total_score", 0),
                    "integration_score": perf_data.get("integration_evidence", {}).get("integration_score", 0),
                    "features_implemented": perf_data.get("orchestration_capabilities", {}).get("features_implemented", 0)
                }
            except Exception as e:
                evidence["performance_metrics"] = {"error": str(e)}

        return evidence

    def assess_production_readiness(self, components: Dict, phase_status: Dict, integration_evidence: Dict) -> Dict[str, Any]:
        """Assess overall production readiness."""
        print("[ASSESSING] Production readiness...")

        readiness = {
            "overall_score": 0,
            "component_readiness": {},
            "phase_readiness": {},
            "integration_readiness": {},
            "deployment_blockers": [],
            "recommendations": [],
            "deployment_strategy": "unknown"
        }

        # Component readiness assessment
        component_scores = {}
        for name, info in components.items():
            if info["status"] == "available":
                feature_count = len(info["features"])
                complexity_normalized = min(info["complexity_score"] / 100, 1.0)
                component_scores[name] = min(feature_count * 10 + complexity_normalized * 20, 100)
            else:
                component_scores[name] = 0
                readiness["deployment_blockers"].append(f"Missing component: {name}")

        readiness["component_readiness"] = component_scores
        avg_component_score = sum(component_scores.values()) / len(component_scores) if component_scores else 0

        # Phase readiness assessment
        if "error" not in phase_status:
            operational_ratio = phase_status.get("operational_phases", 0) / max(phase_status.get("total_phases", 1), 1)
            phase_score = operational_ratio * 100
            readiness["phase_readiness"] = {
                "score": phase_score,
                "operational_phases": phase_status.get("operational_phases", 0),
                "total_phases": phase_status.get("total_phases", 0),
                "health_score": phase_status.get("health_score", 0) * 100
            }

            if phase_score < 50:
                readiness["deployment_blockers"].append(f"Insufficient operational phases: {phase_status.get('operational_phases', 0)}/{phase_status.get('total_phases', 0)}")
        else:
            phase_score = 0
            readiness["phase_readiness"] = {"error": phase_status["error"]}
            readiness["deployment_blockers"].append("Phase validation failed")

        # Integration readiness assessment
        integration_score = 0
        if integration_evidence.get("integration_test_results", {}).get("available", False):
            integration_score += 30
        if integration_evidence.get("checkpoint_demo_results", {}).get("available", False):
            integration_score += 25
            if integration_evidence.get("checkpoint_demo_results", {}).get("all_passed", False):
                integration_score += 25
        if integration_evidence.get("performance_metrics", {}).get("available", False):
            integration_score += 20

        readiness["integration_readiness"] = {
            "score": integration_score,
            "test_files": integration_evidence.get("test_files_detected", 0),
            "evidence_complete": integration_score >= 80
        }

        # Overall score calculation
        readiness["overall_score"] = (avg_component_score * 0.4 + phase_score * 0.3 + integration_score * 0.3)

        # Deployment strategy determination
        if readiness["overall_score"] >= 80 and not readiness["deployment_blockers"]:
            readiness["deployment_strategy"] = "production_ready"
        elif readiness["overall_score"] >= 60:
            readiness["deployment_strategy"] = "staging_deployment"
        elif readiness["overall_score"] >= 40:
            readiness["deployment_strategy"] = "development_deployment"
        else:
            readiness["deployment_strategy"] = "not_ready"

        # Generate recommendations
        if avg_component_score < 70:
            readiness["recommendations"].append("Enhance orchestration component features and error handling")
        if phase_score < 50:
            readiness["recommendations"].append("Implement missing execute methods in phases before production deployment")
        if integration_score < 80:
            readiness["recommendations"].append("Complete integration testing and validation framework")

        return readiness

    def generate_deployment_plan(self, readiness: Dict[str, Any]) -> Dict[str, Any]:
        """Generate deployment plan based on readiness assessment."""
        print("[GENERATING] Deployment plan...")

        deployment_plan = {
            "strategy": readiness["deployment_strategy"],
            "phases": [],
            "timeline": {},
            "prerequisites": [],
            "validation_gates": [],
            "rollback_plan": {}
        }

        if readiness["deployment_strategy"] == "production_ready":
            deployment_plan["phases"] = [
                "Pre-deployment validation",
                "Orchestration system deployment",
                "Phase integration testing",
                "Performance validation",
                "Production cutover",
                "Post-deployment monitoring"
            ]
            deployment_plan["timeline"] = {
                "estimated_duration": "2-3 days",
                "validation_time": "4-6 hours",
                "deployment_window": "2-4 hours"
            }
        elif readiness["deployment_strategy"] == "staging_deployment":
            deployment_plan["phases"] = [
                "Staging environment setup",
                "Limited orchestration deployment",
                "Functional testing",
                "Performance baseline",
                "Production readiness assessment"
            ]
            deployment_plan["timeline"] = {
                "estimated_duration": "1-2 weeks",
                "staging_duration": "3-5 days",
                "testing_duration": "2-3 days"
            }
        elif readiness["deployment_strategy"] == "development_deployment":
            deployment_plan["phases"] = [
                "Development environment deployment",
                "Component integration",
                "Basic functionality testing",
                "Issue identification and resolution"
            ]
            deployment_plan["timeline"] = {
                "estimated_duration": "2-4 weeks",
                "development_duration": "1-2 weeks",
                "testing_duration": "1-2 weeks"
            }
        else:
            deployment_plan["phases"] = [
                "Complete missing components",
                "Implement phase execute methods",
                "Basic integration testing",
                "Readiness re-assessment"
            ]
            deployment_plan["timeline"] = {
                "estimated_duration": "4-8 weeks",
                "development_duration": "3-6 weeks",
                "testing_duration": "1-2 weeks"
            }

        # Common prerequisites
        deployment_plan["prerequisites"] = [
            "Python 3.8+ environment",
            "Required dependencies installed",
            "Proper environment configuration",
            "Access to model checkpoints and data"
        ]

        # Validation gates
        deployment_plan["validation_gates"] = [
            "All orchestration components available",
            "Integration tests passing",
            "Performance benchmarks met",
            "Error handling validated"
        ]

        # Rollback plan
        deployment_plan["rollback_plan"] = {
            "triggers": ["Critical failures", "Performance degradation", "Data corruption"],
            "procedures": ["Stop pipeline execution", "Restore previous checkpoint", "Validate system state"],
            "recovery_time": "15-30 minutes"
        }

        return deployment_plan

    def create_comprehensive_report(self) -> Dict[str, Any]:
        """Create comprehensive orchestration deployment report."""
        print(f"[GENERATING] Comprehensive orchestration deployment report...")
        print(f"[INFO] Report ID: {self.deployment_id}")

        # Analyze all components
        components = self.analyze_orchestration_components()
        phase_status = self.analyze_phase_status()
        integration_evidence = self.analyze_integration_evidence()
        readiness = self.assess_production_readiness(components, phase_status, integration_evidence)
        deployment_plan = self.generate_deployment_plan(readiness)

        # Generate executive summary
        executive_summary = {
            "deployment_id": self.deployment_id,
            "analysis_timestamp": self.report_timestamp,
            "overall_readiness_score": readiness["overall_score"],
            "deployment_strategy": readiness["deployment_strategy"],
            "components_available": sum(1 for c in components.values() if c["status"] == "available"),
            "total_components": len(components),
            "operational_phases": phase_status.get("operational_phases", 0) if "error" not in phase_status else 0,
            "total_phases": phase_status.get("total_phases", 0) if "error" not in phase_status else 8,
            "integration_score": readiness["integration_readiness"]["score"],
            "deployment_blockers": len(readiness["deployment_blockers"]),
            "production_ready": readiness["deployment_strategy"] == "production_ready"
        }

        # Compile complete report
        complete_report = {
            "executive_summary": executive_summary,
            "orchestration_components": components,
            "phase_implementation_status": phase_status,
            "integration_evidence": integration_evidence,
            "production_readiness": readiness,
            "deployment_plan": deployment_plan,
            "recommendations": {
                "immediate_actions": readiness["recommendations"],
                "deployment_blockers": readiness["deployment_blockers"],
                "next_steps": deployment_plan["phases"]
            },
            "metadata": {
                "generator": "OrchestrationDeploymentAnalyzer",
                "version": "1.0.0",
                "generation_timestamp": self.report_timestamp,
                "base_directory": str(self.base_dir),
                "deployment_id": self.deployment_id
            }
        }

        return complete_report

def main():
    """Main function to generate orchestration deployment report."""
    print("=" * 80)
    print("AGENT FORGE ORCHESTRATION DEPLOYMENT SUMMARY")
    print("=" * 80)

    try:
        # Initialize analyzer
        analyzer = OrchestrationDeploymentAnalyzer()

        # Generate comprehensive report
        report = analyzer.create_comprehensive_report()

        # Save detailed JSON report
        report_path = analyzer.base_dir / f"orchestration_deployment_report_{int(time.time())}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Display executive summary
        summary = report["executive_summary"]
        print(f"\n[EXECUTIVE SUMMARY]")
        print(f"Deployment ID: {summary['deployment_id']}")
        print(f"Overall Readiness Score: {summary['overall_readiness_score']:.1f}/100")
        print(f"Deployment Strategy: {summary['deployment_strategy'].replace('_', ' ').title()}")
        print(f"Components Available: {summary['components_available']}/{summary['total_components']}")
        print(f"Operational Phases: {summary['operational_phases']}/{summary['total_phases']}")
        print(f"Integration Score: {summary['integration_score']}/100")
        print(f"Deployment Blockers: {summary['deployment_blockers']}")
        print(f"Production Ready: {'Yes' if summary['production_ready'] else 'No'}")

        # Display component status
        print(f"\n[COMPONENT STATUS]")
        for name, info in report["orchestration_components"].items():
            status_indicator = "[OK]" if info["status"] == "available" else "[MISSING]"
            feature_count = len(info["features"])
            print(f"{status_indicator} {name}: {info['size_loc']} LOC, {feature_count} features")

        # Display deployment recommendations
        recommendations = report["recommendations"]
        if recommendations["deployment_blockers"]:
            print(f"\n[DEPLOYMENT BLOCKERS]")
            for blocker in recommendations["deployment_blockers"]:
                print(f"[CRITICAL] {blocker}")

        if recommendations["immediate_actions"]:
            print(f"\n[IMMEDIATE ACTIONS]")
            for action in recommendations["immediate_actions"]:
                print(f"[ACTION] {action}")

        # Display next steps
        deployment_plan = report["deployment_plan"]
        print(f"\n[DEPLOYMENT PLAN]")
        print(f"Strategy: {deployment_plan['strategy'].replace('_', ' ').title()}")
        print(f"Estimated Duration: {deployment_plan['timeline'].get('estimated_duration', 'Unknown')}")
        print(f"Next Steps:")
        for i, phase in enumerate(deployment_plan["phases"], 1):
            print(f"  {i}. {phase}")

        print(f"\n[REPORT SAVED]")
        print(f"Detailed report: {report_path}")
        print(f"Generated at: {summary['analysis_timestamp']}")

        # Return summary for testing
        return {
            "report_path": str(report_path),
            "summary": summary,
            "success": True
        }

    except Exception as e:
        print(f"[ERROR] Failed to generate deployment report: {str(e)}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    result = main()
    if result.get("success"):
        print(f"\n[SUCCESS] Orchestration deployment analysis completed")
    else:
        print(f"\n[FAILED] Analysis failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)