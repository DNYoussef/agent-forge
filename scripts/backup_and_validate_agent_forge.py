#!/usr/bin/env python3
"""
Agent Forge Backup and Validation Script
Provides comprehensive backup procedures and validation tests for the Agent Forge integration.
"""

import os
import sys
import shutil
import json
import logging
import unittest
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add agent_forge to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'agent_forge'))

import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AgentForgeBackupManager:
    """Manages backups of Agent Forge components."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "backups" / "agent_forge"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self, description: str = "Automated backup") -> str:
        """Create a timestamped backup of all Agent Forge components."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"agent_forge_backup_{timestamp}"
        backup_path = self.backup_dir / backup_name

        try:
            logger.info(f"Creating backup: {backup_name}")

            # Create backup directory
            backup_path.mkdir(exist_ok=True)

            # Copy agent_forge directory
            agent_forge_src = self.project_root / "agent_forge"
            if agent_forge_src.exists():
                shutil.copytree(agent_forge_src, backup_path / "agent_forge")

            # Create backup manifest
            manifest = {
                "backup_name": backup_name,
                "timestamp": timestamp,
                "description": description,
                "files_backed_up": [],
                "backup_size_bytes": 0
            }

            # Calculate backup size and file list
            for root, dirs, files in os.walk(backup_path):
                for file in files:
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(backup_path)
                    manifest["files_backed_up"].append(str(relative_path))
                    manifest["backup_size_bytes"] += file_path.stat().st_size

            # Save manifest
            with open(backup_path / "backup_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            logger.info(f"Backup created successfully: {backup_path}")
            logger.info(f"Files backed up: {len(manifest['files_backed_up'])}")
            logger.info(f"Backup size: {manifest['backup_size_bytes'] / 1024:.2f} KB")

            return str(backup_path)

        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            raise

    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups."""
        backups = []
        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir():
                manifest_file = backup_dir / "backup_manifest.json"
                if manifest_file.exists():
                    try:
                        with open(manifest_file) as f:
                            manifest = json.load(f)
                            backups.append(manifest)
                    except Exception as e:
                        logger.warning(f"Could not read manifest for {backup_dir.name}: {e}")

        return sorted(backups, key=lambda x: x["timestamp"], reverse=True)

    def restore_backup(self, backup_name: str) -> None:
        """Restore from a specific backup."""
        backup_path = self.backup_dir / backup_name
        if not backup_path.exists():
            raise ValueError(f"Backup not found: {backup_name}")

        try:
            logger.info(f"Restoring backup: {backup_name}")

            # Remove existing agent_forge directory
            agent_forge_target = self.project_root / "agent_forge"
            if agent_forge_target.exists():
                shutil.rmtree(agent_forge_target)

            # Restore from backup
            agent_forge_backup = backup_path / "agent_forge"
            if agent_forge_backup.exists():
                shutil.copytree(agent_forge_backup, agent_forge_target)

            logger.info(f"Backup restored successfully from: {backup_name}")

        except Exception as e:
            logger.error(f"Restore failed: {str(e)}")
            raise


class AgentForgeValidator:
    """Comprehensive validation tests for Agent Forge components."""

    def __init__(self):
        self.test_results = []

    def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation tests."""
        logger.info("Starting Agent Forge validation tests...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "tests_passed": 0,
            "tests_failed": 0,
            "test_details": []
        }

        # List of validation tests
        validation_tests = [
            self._test_imports,
            self._test_grokfast_optimizer,
            self._test_cognate_model_creator,
            self._test_model_training,
            self._test_api_server_imports,
            self._test_progress_callbacks
        ]

        for test in validation_tests:
            try:
                test_name = test.__name__
                logger.info(f"Running test: {test_name}")

                test_result = test()
                test_result["test_name"] = test_name
                test_result["status"] = "PASSED" if test_result["success"] else "FAILED"

                results["test_details"].append(test_result)

                if test_result["success"]:
                    results["tests_passed"] += 1
                else:
                    results["tests_failed"] += 1

                logger.info(f"Test {test_name}: {test_result['status']}")

            except Exception as e:
                error_result = {
                    "test_name": test.__name__,
                    "status": "ERROR",
                    "success": False,
                    "message": f"Test execution failed: {str(e)}",
                    "details": {}
                }
                results["test_details"].append(error_result)
                results["tests_failed"] += 1
                logger.error(f"Test {test.__name__} error: {str(e)}")

        results["total_tests"] = results["tests_passed"] + results["tests_failed"]
        results["success_rate"] = results["tests_passed"] / results["total_tests"] if results["total_tests"] > 0 else 0

        logger.info(f"Validation completed: {results['tests_passed']}/{results['total_tests']} tests passed")

        return results

    def _test_imports(self) -> Dict[str, Any]:
        """Test that all components can be imported successfully."""
        try:
            # Test individual imports
            from agent_forge.phases.cognate_pretrain.grokfast_enhanced import EnhancedGrokFastOptimizer
            from agent_forge.phases.cognate_pretrain.cognate_creator import CognateModelCreator
            from agent_forge.api.python_bridge_server import app

            return {
                "success": True,
                "message": "All imports successful",
                "details": {
                    "grokfast_optimizer": "OK",
                    "cognate_creator": "OK",
                    "bridge_server": "OK"
                }
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Import failed: {str(e)}",
                "details": {"error": str(e)}
            }

    def _test_grokfast_optimizer(self) -> Dict[str, Any]:
        """Test Grokfast optimizer functionality."""
        try:
            from agent_forge.phases.cognate_pretrain.grokfast_enhanced import EnhancedGrokFastOptimizer

            # Create simple model
            model = torch.nn.Linear(10, 5)
            base_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            grokfast_optimizer = EnhancedGrokFastOptimizer(base_optimizer, alpha=0.9, lambda_=0.1)

            # Test basic operations
            model.zero_grad()

            # Create dummy loss
            x = torch.randn(32, 10)
            y = torch.randn(32, 5)
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()

            # Test step
            grokfast_optimizer.step(model)
            grokfast_optimizer.zero_grad()

            return {
                "success": True,
                "message": "Grokfast optimizer working correctly",
                "details": {
                    "alpha": grokfast_optimizer.alpha,
                    "lambda": grokfast_optimizer.lambda_,
                    "step_count": grokfast_optimizer.step_count
                }
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Grokfast optimizer test failed: {str(e)}",
                "details": {"error": str(e)}
            }

    def _test_cognate_model_creator(self) -> Dict[str, Any]:
        """Test CognateModelCreator functionality."""
        try:
            from agent_forge.phases.cognate_pretrain.cognate_creator import CognateModelCreator

            # Create model creator
            creator = CognateModelCreator(
                vocab_size=1000,
                d_model=64,
                nhead=4,
                num_layers=2,
                grokfast_enabled=True
            )

            # Test model creation
            model = creator.create_model()
            model_info = creator.get_model_info()

            return {
                "success": True,
                "message": "CognateModelCreator working correctly",
                "details": {
                    "total_parameters": model_info["total_parameters"],
                    "grokfast_enabled": model_info["grokfast_config"]["enabled"],
                    "model_device": str(creator.device)
                }
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"CognateModelCreator test failed: {str(e)}",
                "details": {"error": str(e)}
            }

    def _test_model_training(self) -> Dict[str, Any]:
        """Test basic model training functionality."""
        try:
            from agent_forge.phases.cognate_pretrain.cognate_creator import CognateModelCreator, create_sample_training_data

            # Create small model for testing
            creator = CognateModelCreator(
                vocab_size=100,
                d_model=32,
                nhead=2,
                num_layers=1,
                grokfast_enabled=True
            )

            # Generate minimal training data
            training_data = create_sample_training_data(
                vocab_size=100,
                num_samples=10,
                seq_length=16
            )

            # Test short training run
            training_stats = creator.train(
                train_data=training_data,
                epochs=1,
                batch_size=5
            )

            return {
                "success": True,
                "message": "Model training working correctly",
                "details": {
                    "epochs": training_stats["epochs"],
                    "total_steps": training_stats["total_steps"],
                    "final_loss": training_stats["final_loss"],
                    "grokfast_enabled": training_stats["grokfast_enabled"]
                }
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Model training test failed: {str(e)}",
                "details": {"error": str(e)}
            }

    def _test_api_server_imports(self) -> Dict[str, Any]:
        """Test API server can be imported and initialized."""
        try:
            from agent_forge.api.python_bridge_server import app
            from fastapi.testclient import TestClient

            # Create test client
            client = TestClient(app)

            # Test root endpoint
            response = client.get("/")

            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "message": "API server working correctly",
                    "details": {
                        "api_version": data.get("version", "unknown"),
                        "status": data.get("status", "unknown")
                    }
                }
            else:
                return {
                    "success": False,
                    "message": f"API server returned status code: {response.status_code}",
                    "details": {"status_code": response.status_code}
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"API server test failed: {str(e)}",
                "details": {"error": str(e)}
            }

    def _test_progress_callbacks(self) -> Dict[str, Any]:
        """Test progress callback functionality."""
        try:
            from agent_forge.phases.cognate_pretrain.cognate_creator import CognateModelCreator, create_sample_training_data

            # Create model
            creator = CognateModelCreator(
                vocab_size=50,
                d_model=32,
                nhead=2,
                num_layers=1
            )

            # Test progress callback
            callback_calls = []

            def test_callback(step: int, loss: float, perplexity: float):
                callback_calls.append({"step": step, "loss": loss, "perplexity": perplexity})

            # Generate minimal training data
            training_data = create_sample_training_data(
                vocab_size=50,
                num_samples=5,
                seq_length=8
            )

            # Train with callback
            creator.train(
                train_data=training_data,
                epochs=1,
                batch_size=2,
                progress_callback=test_callback
            )

            return {
                "success": len(callback_calls) > 0,
                "message": f"Progress callbacks working - {len(callback_calls)} calls received",
                "details": {
                    "callback_calls": len(callback_calls),
                    "first_call": callback_calls[0] if callback_calls else None
                }
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Progress callback test failed: {str(e)}",
                "details": {"error": str(e)}
            }


def main():
    """Main backup and validation script."""
    import argparse

    parser = argparse.ArgumentParser(description="Agent Forge Backup and Validation")
    parser.add_argument("--project-root", default=os.path.dirname(os.path.dirname(__file__)),
                       help="Project root directory")
    parser.add_argument("--backup", action="store_true", help="Create backup before validation")
    parser.add_argument("--validate", action="store_true", help="Run validation tests")
    parser.add_argument("--list-backups", action="store_true", help="List available backups")
    parser.add_argument("--restore", help="Restore from specific backup")

    args = parser.parse_args()

    project_root = Path(args.project_root)
    backup_manager = AgentForgeBackupManager(str(project_root))

    try:
        if args.list_backups:
            backups = backup_manager.list_backups()
            print(f"Available backups ({len(backups)}):")
            for backup in backups:
                print(f"  {backup['backup_name']} - {backup['timestamp']} - {backup['description']}")

        elif args.restore:
            backup_manager.restore_backup(args.restore)
            print(f"Restored from backup: {args.restore}")

        elif args.backup:
            backup_path = backup_manager.create_backup("Manual backup before validation")
            print(f"Backup created: {backup_path}")

        if args.validate or (not any([args.backup, args.list_backups, args.restore])):
            validator = AgentForgeValidator()
            results = validator.run_all_validations()

            # Print summary
            print(f"\n=== VALIDATION RESULTS ===")
            print(f"Tests passed: {results['tests_passed']}/{results['total_tests']}")
            print(f"Success rate: {results['success_rate']:.1%}")

            # Print detailed results
            for test in results['test_details']:
                status_icon = "✓" if test['success'] else "✗"
                print(f"{status_icon} {test['test_name']}: {test['message']}")

            # Exit with appropriate code
            exit_code = 0 if results['tests_failed'] == 0 else 1
            sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()


# Version & Run Log Footer
"""
## Version & Run Log
| Version | Timestamp | Agent/Model | Change Summary | Artifacts | Status | Notes | Cost | Hash |
|--------:|-----------|-------------|----------------|-----------|--------|-------|------|------|
| 1.0.0   | 2025-09-25T11:15:00-04:00 | backend-dev@claude-4 | Complete backup and validation script | backup_and_validate_agent_forge.py | OK | Full test suite with backup procedures | 0.00 | d0e6f3b |

### Receipt
- status: OK
- reason_if_blocked: --
- run_id: backup-validation-001
- inputs: ["all_agent_forge_components"]
- tools_used: ["Write"]
- versions: {"model":"claude-4","prompt":"v1.0"}
"""