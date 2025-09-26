"""
Resource Manager - Hardware Detection and Resource Optimization

Provides hardware detection and resource management for parallel training.
"""

import psutil
import threading
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class SystemResources:
    """System resource information"""
    cpu_count: int
    memory_gb: float
    available_memory_gb: float
    cpu_usage_percent: float
    recommended_workers: int
    memory_per_worker_gb: float


class ResourceManager:
    """
    Hardware detection and resource management for Agent Forge training

    Provides conservative resource allocation and monitoring capabilities.
    """

    def __init__(self):
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._resource_data = {}
        self._lock = threading.Lock()

    def detect_system_resources(self) -> SystemResources:
        """
        Detect current system resources and provide recommendations

        Returns:
            SystemResources with current state and recommendations
        """
        try:
            # CPU information
            cpu_count = psutil.cpu_count()
            cpu_usage = psutil.cpu_percent(interval=1.0)

            # Memory information
            memory = psutil.virtual_memory()
            total_memory_gb = memory.total / (1024**3)
            available_memory_gb = memory.available / (1024**3)

            # Conservative worker calculation
            # Rule: 1 worker per 8GB RAM, max half of CPUs
            max_workers_by_memory = max(1, int(available_memory_gb // 8))
            max_workers_by_cpu = max(1, cpu_count // 2)
            recommended_workers = min(max_workers_by_memory, max_workers_by_cpu, 6)  # Cap at 6

            memory_per_worker = available_memory_gb / recommended_workers if recommended_workers > 0 else available_memory_gb

            return SystemResources(
                cpu_count=cpu_count,
                memory_gb=total_memory_gb,
                available_memory_gb=available_memory_gb,
                cpu_usage_percent=cpu_usage,
                recommended_workers=recommended_workers,
                memory_per_worker_gb=memory_per_worker
            )

        except Exception as e:
            print(f"Resource detection error: {e}")
            # Safe fallback
            return SystemResources(
                cpu_count=1,
                memory_gb=4.0,
                available_memory_gb=2.0,
                cpu_usage_percent=50.0,
                recommended_workers=1,
                memory_per_worker_gb=2.0
            )

    def validate_worker_count(self, requested_workers: int) -> Tuple[int, str]:
        """
        Validate and adjust requested worker count based on system resources

        Args:
            requested_workers: Number of workers requested

        Returns:
            Tuple of (validated_workers, reason)
        """
        resources = self.detect_system_resources()

        if requested_workers <= 0:
            return 1, "Invalid worker count, using minimum of 1"

        if requested_workers <= resources.recommended_workers:
            return requested_workers, "Requested worker count is safe"

        # Check specific constraints
        if resources.available_memory_gb < requested_workers * 8:
            safe_workers = max(1, int(resources.available_memory_gb // 8))
            return safe_workers, f"Memory constrained: {resources.available_memory_gb:.1f}GB available, need {requested_workers * 8}GB"

        if requested_workers > resources.cpu_count // 2:
            safe_workers = max(1, resources.cpu_count // 2)
            return safe_workers, f"CPU constrained: {resources.cpu_count} CPUs, max recommended {safe_workers}"

        if resources.cpu_usage_percent > 80:
            return 1, f"High CPU usage: {resources.cpu_usage_percent:.1f}%, limiting to 1 worker"

        return resources.recommended_workers, "Using system-recommended worker count"

    def start_monitoring(self, interval: float = 5.0) -> None:
        """
        Start monitoring system resources in background thread

        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        print(f"Started resource monitoring (interval: {interval}s)")

    def stop_monitoring(self) -> None:
        """Stop resource monitoring"""
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        print("Stopped resource monitoring")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics"""
        with self._lock:
            return dict(self._resource_data)

    def _monitor_loop(self, interval: float) -> None:
        """Background monitoring loop"""
        while self._monitoring:
            try:
                resources = self.detect_system_resources()

                with self._lock:
                    self._resource_data = {
                        'timestamp': time.time(),
                        'cpu_count': resources.cpu_count,
                        'memory_gb': resources.memory_gb,
                        'available_memory_gb': resources.available_memory_gb,
                        'cpu_usage_percent': resources.cpu_usage_percent,
                        'recommended_workers': resources.recommended_workers,
                        'memory_per_worker_gb': resources.memory_per_worker_gb
                    }

                time.sleep(interval)

            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(interval)

    def generate_resource_report(self) -> Dict[str, Any]:
        """Generate comprehensive resource report"""
        resources = self.detect_system_resources()
        current_metrics = self.get_current_metrics()

        return {
            'system_info': {
                'cpu_count': resources.cpu_count,
                'total_memory_gb': resources.memory_gb,
                'available_memory_gb': resources.available_memory_gb
            },
            'current_usage': {
                'cpu_usage_percent': resources.cpu_usage_percent,
                'memory_usage_percent': ((resources.memory_gb - resources.available_memory_gb) / resources.memory_gb) * 100
            },
            'recommendations': {
                'parallel_workers': resources.recommended_workers,
                'memory_per_worker_gb': resources.memory_per_worker_gb,
                'training_mode_suggestion': 'parallel' if resources.recommended_workers > 1 else 'series'
            },
            'constraints': {
                'memory_limited': resources.available_memory_gb < 16,
                'cpu_limited': resources.cpu_count < 4,
                'high_usage': resources.cpu_usage_percent > 70
            },
            'monitoring_active': self._monitoring,
            'last_update': current_metrics.get('timestamp', time.time())
        }


# Utility functions for external use
def get_optimal_workers(requested: int = 3) -> Tuple[int, str]:
    """Quick utility to get optimal worker count"""
    manager = ResourceManager()
    return manager.validate_worker_count(requested)


def get_resource_summary() -> Dict[str, Any]:
    """Quick utility to get resource summary"""
    manager = ResourceManager()
    return manager.generate_resource_report()


if __name__ == "__main__":
    # Demo resource management
    print("=== Agent Forge Resource Manager Demo ===")

    manager = ResourceManager()

    # Test resource detection
    print("\n1. System Resource Detection:")
    resources = manager.detect_system_resources()
    print(f"CPUs: {resources.cpu_count}, Memory: {resources.memory_gb:.1f}GB")
    print(f"Available: {resources.available_memory_gb:.1f}GB, CPU Usage: {resources.cpu_usage_percent:.1f}%")
    print(f"Recommended Workers: {resources.recommended_workers}")

    # Test worker validation
    print("\n2. Worker Validation:")
    test_workers = [1, 2, 4, 8, 16]
    for workers in test_workers:
        validated, reason = manager.validate_worker_count(workers)
        print(f"Requested: {workers} -> Validated: {validated} ({reason})")

    # Test resource report
    print("\n3. Resource Report:")
    report = manager.generate_resource_report()
    print(f"Suggested Mode: {report['recommendations']['training_mode_suggestion']}")
    print(f"Memory Limited: {report['constraints']['memory_limited']}")
    print(f"CPU Limited: {report['constraints']['cpu_limited']}")

    print("\n=== Demo Complete ===")

# Version & Run Log Footer
"""
Version & Run Log

| Version | Timestamp | Agent/Model | Change Summary | Artifacts | Status | Notes | Cost | Hash |
|--------:|-----------|-------------|----------------|-----------|--------|-------|------|------|
| 1.0.0   | 2025-01-25T15:20:03-05:00 | system@architect | Create ResourceManager for hardware detection | resource_manager.py | OK | Conservative resource allocation with monitoring | 0.00 | b2f9d4c |

Receipt:
- status: OK
- reason_if_blocked: --
- run_id: agent-forge-002
- inputs: ["hardware-requirements", "resource-constraints"]
- tools_used: ["Write"]
- versions: {"model":"claude-sonnet-4","design":"resource-management-v1"}
"""