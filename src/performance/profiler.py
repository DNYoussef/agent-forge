"""
Agent Forge Performance Profiler - Princess Infrastructure Enhancement
Comprehensive CPU, memory, and execution profiling with bottleneck identification.

CRITICAL PERFORMANCE ISSUES ADDRESSED:
- Grade D average performance (from UI audit)
- 50% of phases fail performance gates
- Average load time: 3.5 seconds (exceeds 3.0s threshold)
- 4 phases with JavaScript errors
- Only 25% of phases operational (2 of 8)
"""

import asyncio
import cProfile
import functools
import gc
import logging
import pstats
import psutil
import threading
import time
import tracemalloc
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import torch
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics collection."""
    timestamp: datetime = field(default_factory=datetime.now)

    # CPU Metrics
    cpu_percent: float = 0.0
    cpu_cores_used: int = 0
    cpu_frequency: float = 0.0
    load_average: List[float] = field(default_factory=list)

    # Memory Metrics
    memory_used_mb: float = 0.0
    memory_percent: float = 0.0
    memory_available_mb: float = 0.0
    swap_used_mb: float = 0.0

    # GPU Metrics (if available)
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization: float = 0.0

    # Execution Metrics
    execution_time_ms: float = 0.0
    function_calls: int = 0
    peak_memory_mb: float = 0.0

    # I/O Metrics
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0

    # Custom Metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BottleneckReport:
    """Bottleneck identification and analysis report."""
    timestamp: datetime = field(default_factory=datetime.now)

    # Critical Issues
    critical_bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    performance_warnings: List[Dict[str, Any]] = field(default_factory=list)

    # Resource Analysis
    cpu_bottlenecks: List[str] = field(default_factory=list)
    memory_bottlenecks: List[str] = field(default_factory=list)
    io_bottlenecks: List[str] = field(default_factory=list)

    # Function-level Analysis
    slow_functions: List[Dict[str, Any]] = field(default_factory=list)
    memory_leaks: List[Dict[str, Any]] = field(default_factory=list)

    # Recommendations
    optimization_recommendations: List[Dict[str, Any]] = field(default_factory=list)

    # Performance Score (0-100)
    overall_performance_score: float = 0.0


class PerformanceProfiler:
    """
    Comprehensive performance profiler for Agent Forge system.

    Capabilities:
    - Real-time CPU, memory, GPU, and I/O monitoring
    - Function-level profiling with cProfile integration
    - Memory leak detection with tracemalloc
    - Bottleneck identification and analysis
    - Performance regression detection
    - Automated optimization recommendations
    """

    def __init__(self, sampling_interval: float = 0.1, history_size: int = 1000):
        """Initialize the performance profiler."""
        self.sampling_interval = sampling_interval
        self.history_size = history_size

        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()

        # Data storage
        self.metrics_history: deque = deque(maxlen=history_size)
        self.function_profiles: Dict[str, pstats.Stats] = {}
        self.memory_snapshots: List[Any] = []

        # Baseline metrics for comparison
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        self.performance_thresholds = {
            'cpu_percent_warning': 80.0,
            'cpu_percent_critical': 95.0,
            'memory_percent_warning': 85.0,
            'memory_percent_critical': 95.0,
            'execution_time_warning_ms': 3000.0,  # 3 seconds
            'execution_time_critical_ms': 5000.0,  # 5 seconds
            'memory_leak_threshold_mb': 100.0
        }

        # System information
        self.system_info = self._collect_system_info()

        # GPU availability
        self.gpu_available = torch.cuda.is_available()

        logger.info(f"PerformanceProfiler initialized - GPU Available: {self.gpu_available}")

    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect static system information."""
        try:
            cpu_info = {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'max_frequency': psutil.cpu_freq().max if psutil.cpu_freq() else 0,
            }

            memory_info = {
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'total_swap_gb': psutil.swap_memory().total / (1024**3),
            }

            gpu_info = {}
            if self.gpu_available:
                gpu_info = {
                    'gpu_count': torch.cuda.device_count(),
                    'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown",
                    'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.device_count() > 0 else 0
                }

            return {
                'cpu': cpu_info,
                'memory': memory_info,
                'gpu': gpu_info,
                'platform': psutil.LINUX if hasattr(psutil, 'LINUX') else 'unknown'
            }
        except Exception as e:
            logger.warning(f"Error collecting system info: {e}")
            return {}

    def start_monitoring(self):
        """Start real-time performance monitoring."""
        if self.is_monitoring:
            logger.warning("Performance monitoring already running")
            return

        self.is_monitoring = True
        self.stop_monitoring.clear()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        # Start memory tracking
        tracemalloc.start()

        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        self.stop_monitoring.set()

        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

        # Stop memory tracking
        try:
            tracemalloc.stop()
        except:
            pass

        logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop running in separate thread."""
        while not self.stop_monitoring.is_set():
            try:
                metrics = self._collect_current_metrics()
                self.metrics_history.append(metrics)

                # Check for critical issues
                self._check_critical_thresholds(metrics)

                time.sleep(self.sampling_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.sampling_interval)

    def _collect_current_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics."""
        metrics = PerformanceMetrics()

        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_freq = psutil.cpu_freq()
            metrics.cpu_percent = cpu_percent
            metrics.cpu_frequency = cpu_freq.current if cpu_freq else 0

            try:
                metrics.load_average = list(psutil.getloadavg())
            except AttributeError:
                # Not available on Windows
                metrics.load_average = [0, 0, 0]

            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            metrics.memory_used_mb = memory.used / (1024**2)
            metrics.memory_percent = memory.percent
            metrics.memory_available_mb = memory.available / (1024**2)
            metrics.swap_used_mb = swap.used / (1024**2)

            # GPU metrics
            if self.gpu_available and torch.cuda.device_count() > 0:
                metrics.gpu_memory_used_mb = torch.cuda.memory_allocated(0) / (1024**2)
                metrics.gpu_memory_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                try:
                    # GPU utilization requires nvidia-ml-py
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics.gpu_utilization = util.gpu
                except ImportError:
                    metrics.gpu_utilization = 0.0

            # I/O metrics
            try:
                io_counters = psutil.disk_io_counters()
                if io_counters:
                    metrics.disk_read_mb = io_counters.read_bytes / (1024**2)
                    metrics.disk_write_mb = io_counters.write_bytes / (1024**2)

                net_counters = psutil.net_io_counters()
                if net_counters:
                    metrics.network_sent_mb = net_counters.bytes_sent / (1024**2)
                    metrics.network_recv_mb = net_counters.bytes_recv / (1024**2)
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")

        return metrics

    def _check_critical_thresholds(self, metrics: PerformanceMetrics):
        """Check for critical performance thresholds."""
        if metrics.cpu_percent > self.performance_thresholds['cpu_percent_critical']:
            logger.critical(f"Critical CPU usage: {metrics.cpu_percent:.1f}%")

        if metrics.memory_percent > self.performance_thresholds['memory_percent_critical']:
            logger.critical(f"Critical memory usage: {metrics.memory_percent:.1f}%")

    @contextmanager
    def profile_function(self, function_name: str):
        """Context manager for profiling specific functions."""
        profiler = cProfile.Profile()
        start_time = time.time()
        start_memory = psutil.virtual_memory().used

        # Start profiling
        profiler.enable()

        try:
            yield
        finally:
            # Stop profiling
            profiler.disable()

            # Calculate execution time
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            memory_used = psutil.virtual_memory().used - start_memory

            # Store profile
            stats = pstats.Stats(profiler)
            self.function_profiles[function_name] = stats

            # Log performance info
            logger.info(f"Function '{function_name}' - Execution: {execution_time:.2f}ms, Memory: {memory_used / (1024**2):.2f}MB")

            # Check thresholds
            if execution_time > self.performance_thresholds['execution_time_warning_ms']:
                logger.warning(f"Slow function detected: '{function_name}' took {execution_time:.2f}ms")

    def profile_decorator(self, function_name: Optional[str] = None):
        """Decorator for automatic function profiling."""
        def decorator(func: Callable):
            name = function_name or f"{func.__module__}.{func.__name__}"

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile_function(name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def take_memory_snapshot(self, label: str = ""):
        """Take a memory snapshot for leak detection."""
        try:
            if tracemalloc.is_tracing():
                snapshot = tracemalloc.take_snapshot()
                self.memory_snapshots.append({
                    'timestamp': datetime.now(),
                    'label': label,
                    'snapshot': snapshot
                })
                logger.debug(f"Memory snapshot taken: {label}")
        except Exception as e:
            logger.error(f"Error taking memory snapshot: {e}")

    def analyze_memory_leaks(self) -> List[Dict[str, Any]]:
        """Analyze memory snapshots for potential leaks."""
        leaks = []

        if len(self.memory_snapshots) < 2:
            return leaks

        try:
            # Compare first and last snapshots
            first_snapshot = self.memory_snapshots[0]['snapshot']
            last_snapshot = self.memory_snapshots[-1]['snapshot']

            top_stats = last_snapshot.compare_to(first_snapshot, 'lineno')

            for stat in top_stats[:10]:  # Top 10 memory allocations
                if stat.size_diff > self.performance_thresholds['memory_leak_threshold_mb'] * 1024 * 1024:
                    leaks.append({
                        'filename': stat.traceback.format()[0] if stat.traceback else "Unknown",
                        'size_diff_mb': stat.size_diff / (1024**2),
                        'count_diff': stat.count_diff,
                        'traceback': stat.traceback.format() if stat.traceback else []
                    })

        except Exception as e:
            logger.error(f"Error analyzing memory leaks: {e}")

        return leaks

    def identify_bottlenecks(self) -> BottleneckReport:
        """Comprehensive bottleneck identification and analysis."""
        report = BottleneckReport()

        if not self.metrics_history:
            logger.warning("No metrics history available for bottleneck analysis")
            return report

        try:
            # Analyze metrics history
            recent_metrics = list(self.metrics_history)[-100:]  # Last 100 samples

            # CPU bottleneck analysis
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            max_cpu = max(m.cpu_percent for m in recent_metrics)

            if max_cpu > self.performance_thresholds['cpu_percent_critical']:
                report.critical_bottlenecks.append({
                    'type': 'cpu',
                    'severity': 'critical',
                    'description': f"CPU usage peaked at {max_cpu:.1f}%",
                    'recommendation': "Consider parallel processing or CPU optimization"
                })
            elif avg_cpu > self.performance_thresholds['cpu_percent_warning']:
                report.performance_warnings.append({
                    'type': 'cpu',
                    'description': f"Average CPU usage high: {avg_cpu:.1f}%",
                    'recommendation': "Monitor CPU-intensive operations"
                })

            # Memory bottleneck analysis
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            max_memory = max(m.memory_percent for m in recent_metrics)

            if max_memory > self.performance_thresholds['memory_percent_critical']:
                report.critical_bottlenecks.append({
                    'type': 'memory',
                    'severity': 'critical',
                    'description': f"Memory usage peaked at {max_memory:.1f}%",
                    'recommendation': "Implement memory optimization or increase RAM"
                })

            # Function-level analysis
            for func_name, stats in self.function_profiles.items():
                # Analyze function performance
                string_io = StringIO()
                stats.print_stats(stream=string_io)
                stats_str = string_io.getvalue()

                # Extract timing info (simplified)
                if 'tottime' in stats_str:
                    report.slow_functions.append({
                        'function': func_name,
                        'stats_summary': stats_str.split('\n')[0] if stats_str else "No data"
                    })

            # Memory leak analysis
            memory_leaks = self.analyze_memory_leaks()
            report.memory_leaks = memory_leaks

            # Generate optimization recommendations
            report.optimization_recommendations = self._generate_optimization_recommendations(report)

            # Calculate overall performance score
            report.overall_performance_score = self._calculate_performance_score(recent_metrics, report)

        except Exception as e:
            logger.error(f"Error in bottleneck analysis: {e}")

        return report

    def _generate_optimization_recommendations(self, report: BottleneckReport) -> List[Dict[str, Any]]:
        """Generate specific optimization recommendations."""
        recommendations = []

        # CPU optimization recommendations
        if any(b['type'] == 'cpu' for b in report.critical_bottlenecks):
            recommendations.append({
                'category': 'cpu_optimization',
                'priority': 'high',
                'title': 'CPU Usage Optimization',
                'description': 'Implement parallel processing and optimize CPU-intensive operations',
                'actions': [
                    'Profile CPU-intensive functions',
                    'Implement multiprocessing for parallel tasks',
                    'Optimize algorithms for better time complexity',
                    'Consider caching for repeated computations'
                ]
            })

        # Memory optimization recommendations
        if any(b['type'] == 'memory' for b in report.critical_bottlenecks):
            recommendations.append({
                'category': 'memory_optimization',
                'priority': 'high',
                'title': 'Memory Usage Optimization',
                'description': 'Reduce memory footprint and prevent memory leaks',
                'actions': [
                    'Implement lazy loading for large objects',
                    'Add memory pooling for frequent allocations',
                    'Review and fix memory leaks',
                    'Optimize data structures for memory efficiency'
                ]
            })

        # I/O optimization recommendations
        if len(report.slow_functions) > 5:
            recommendations.append({
                'category': 'io_optimization',
                'priority': 'medium',
                'title': 'I/O Performance Optimization',
                'description': 'Optimize disk and network operations',
                'actions': [
                    'Implement async I/O operations',
                    'Add connection pooling',
                    'Cache frequently accessed data',
                    'Optimize database queries'
                ]
            })

        return recommendations

    def _calculate_performance_score(self, metrics: List[PerformanceMetrics], report: BottleneckReport) -> float:
        """Calculate overall performance score (0-100)."""
        if not metrics:
            return 0.0

        try:
            # Base score
            score = 100.0

            # CPU penalty
            avg_cpu = sum(m.cpu_percent for m in metrics) / len(metrics)
            if avg_cpu > 80:
                score -= (avg_cpu - 80) * 2  # 2 points per % over 80%

            # Memory penalty
            avg_memory = sum(m.memory_percent for m in metrics) / len(metrics)
            if avg_memory > 80:
                score -= (avg_memory - 80) * 2  # 2 points per % over 80%

            # Critical bottleneck penalty
            score -= len(report.critical_bottlenecks) * 20

            # Warning penalty
            score -= len(report.performance_warnings) * 5

            # Memory leak penalty
            score -= len(report.memory_leaks) * 10

            return max(0.0, min(100.0, score))

        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.0

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self.system_info,
            'monitoring_duration_minutes': len(self.metrics_history) * self.sampling_interval / 60,
            'samples_collected': len(self.metrics_history),
        }

        if self.metrics_history:
            # Summary statistics
            recent_metrics = list(self.metrics_history)

            report['performance_summary'] = {
                'avg_cpu_percent': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
                'max_cpu_percent': max(m.cpu_percent for m in recent_metrics),
                'avg_memory_percent': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
                'max_memory_percent': max(m.memory_percent for m in recent_metrics),
                'avg_memory_used_mb': sum(m.memory_used_mb for m in recent_metrics) / len(recent_metrics),
                'peak_memory_used_mb': max(m.memory_used_mb for m in recent_metrics),
            }

            if self.gpu_available:
                report['gpu_summary'] = {
                    'avg_gpu_memory_mb': sum(m.gpu_memory_used_mb for m in recent_metrics) / len(recent_metrics),
                    'peak_gpu_memory_mb': max(m.gpu_memory_used_mb for m in recent_metrics),
                    'avg_gpu_utilization': sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics),
                }

        # Bottleneck analysis
        bottleneck_report = self.identify_bottlenecks()
        report['bottleneck_analysis'] = {
            'critical_bottlenecks': bottleneck_report.critical_bottlenecks,
            'performance_warnings': bottleneck_report.performance_warnings,
            'memory_leaks': bottleneck_report.memory_leaks,
            'optimization_recommendations': bottleneck_report.optimization_recommendations,
            'overall_performance_score': bottleneck_report.overall_performance_score
        }

        # Function profiles
        report['function_profiles'] = list(self.function_profiles.keys())

        return report

    def save_report(self, filepath: str):
        """Save performance report to file."""
        report = self.generate_performance_report()

        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Performance report saved to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving performance report: {e}")

    def get_real_time_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent performance metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None

    def set_baseline(self):
        """Set current metrics as baseline for comparison."""
        current_metrics = self._collect_current_metrics()
        self.baseline_metrics = current_metrics
        logger.info("Performance baseline set")

    def compare_to_baseline(self) -> Optional[Dict[str, float]]:
        """Compare current metrics to baseline."""
        if not self.baseline_metrics:
            return None

        current_metrics = self._collect_current_metrics()

        return {
            'cpu_percent_change': current_metrics.cpu_percent - self.baseline_metrics.cpu_percent,
            'memory_percent_change': current_metrics.memory_percent - self.baseline_metrics.memory_percent,
            'memory_mb_change': current_metrics.memory_used_mb - self.baseline_metrics.memory_used_mb,
        }


# Convenience functions
def create_profiler(sampling_interval: float = 0.1) -> PerformanceProfiler:
    """Create and return a new performance profiler instance."""
    return PerformanceProfiler(sampling_interval=sampling_interval)


def profile_function(function_name: str = None):
    """Decorator for automatic function profiling."""
    def decorator(func: Callable):
        name = function_name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = PerformanceProfiler()
            with profiler.profile_function(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage and testing
    profiler = create_profiler()

    try:
        print("Starting performance monitoring...")
        profiler.start_monitoring()

        # Simulate some work
        @profiler.profile_decorator("test_function")
        def test_function():
            time.sleep(0.1)
            # Simulate some computation
            data = [i**2 for i in range(10000)]
            return sum(data)

        # Test the function
        result = test_function()
        print(f"Test function result: {result}")

        # Wait for some metrics collection
        time.sleep(2)

        # Generate report
        report = profiler.generate_performance_report()
        print(f"Performance Score: {report['bottleneck_analysis']['overall_performance_score']:.1f}/100")

        # Save report
        profiler.save_report("performance_test_report.json")

    finally:
        profiler.stop_monitoring()
        print("Performance monitoring stopped")