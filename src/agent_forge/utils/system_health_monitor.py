"""
System Health Monitoring Dashboard for Agent Forge.
Monitors system resources and predicts training failures before they occur.
"""

import os
import psutil
import logging
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import torch
import numpy as np


@dataclass
class SystemHealthMetrics:
    """System health metrics snapshot"""
    timestamp: float
    gpu_memory: Dict[str, float]
    disk_space: Dict[str, float]
    cpu_usage: Dict[str, float]
    memory_usage: Dict[str, float]
    network_connectivity: Dict[str, bool]
    checkpoint_integrity: Dict[str, bool]
    temperature: Dict[str, float]
    prediction_score: float


class SystemHealthMonitor:
    """Monitor system resources and training health"""

    def __init__(self, monitoring_interval: int = 60):
        self.monitoring_interval = monitoring_interval
        self.logger = logging.getLogger(__name__)

        # Historical data storage
        self.health_history: List[SystemHealthMetrics] = []
        self.max_history_size = 1000

        # Monitoring state
        self.monitoring_active = False
        self.last_check_time = 0

        # Failure prediction model (simple heuristic-based)
        self.failure_indicators = {
            'gpu_memory_critical': 0.95,  # >95% usage
            'disk_space_critical': 0.95,  # >95% usage
            'cpu_usage_critical': 0.90,   # >90% usage
            'memory_usage_critical': 0.90, # >90% usage
            'temperature_critical': 85,    # >85°C
        }

        self.logger.info("SystemHealthMonitor initialized")

    def check_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health check"""

        try:
            current_time = time.time()

            # GPU Memory Check
            gpu_memory = self._check_gpu_memory()

            # Disk Space Check
            disk_space = self._check_disk_space()

            # CPU Usage Check
            cpu_usage = self._check_cpu_usage()

            # Memory Usage Check
            memory_usage = self._check_memory_usage()

            # Network Connectivity Check
            network_connectivity = self._check_network()

            # Checkpoint Integrity Check
            checkpoint_integrity = self._verify_checkpoints()

            # Temperature Check (if available)
            temperature = self._check_temperature()

            # Create health snapshot
            health_snapshot = SystemHealthMetrics(
                timestamp=current_time,
                gpu_memory=gpu_memory,
                disk_space=disk_space,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                network_connectivity=network_connectivity,
                checkpoint_integrity=checkpoint_integrity,
                temperature=temperature,
                prediction_score=0.0  # Will be calculated
            )

            # Calculate failure prediction
            health_snapshot.prediction_score = self.predict_failure_risk(asdict(health_snapshot))

            # Store in history
            self.health_history.append(health_snapshot)

            # Keep history manageable
            if len(self.health_history) > self.max_history_size:
                self.health_history = self.health_history[-self.max_history_size:]

            self.last_check_time = current_time

            return asdict(health_snapshot)

        except Exception as e:
            self.logger.error(f"System health check failed: {e}")
            return self._get_fallback_health_status()

    def _check_gpu_memory(self) -> Dict[str, float]:
        """Check GPU memory usage"""
        gpu_info = {
            'available_gb': 0.0,
            'used_gb': 0.0,
            'total_gb': 0.0,
            'utilization_percent': 0.0
        }

        try:
            if torch.cuda.is_available():
                # Get GPU memory info
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated(0)
                cached_memory = torch.cuda.memory_reserved(0)

                gpu_info.update({
                    'total_gb': total_memory / (1024**3),
                    'used_gb': allocated_memory / (1024**3),
                    'cached_gb': cached_memory / (1024**3),
                    'available_gb': (total_memory - allocated_memory) / (1024**3),
                    'utilization_percent': (allocated_memory / total_memory) * 100
                })

                # Try to get GPU utilization if nvidia-smi available
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                         capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        gpu_info['gpu_utilization_percent'] = float(result.stdout.strip())
                except:
                    gpu_info['gpu_utilization_percent'] = 0.0

            else:
                self.logger.warning("CUDA not available - using CPU fallback")

        except Exception as e:
            self.logger.error(f"GPU memory check failed: {e}")

        return gpu_info

    def _check_disk_space(self) -> Dict[str, float]:
        """Check disk space availability"""
        disk_info = {
            'total_gb': 0.0,
            'used_gb': 0.0,
            'available_gb': 0.0,
            'utilization_percent': 0.0
        }

        try:
            # Check current working directory disk space
            disk_usage = psutil.disk_usage('.')

            disk_info.update({
                'total_gb': disk_usage.total / (1024**3),
                'used_gb': disk_usage.used / (1024**3),
                'available_gb': disk_usage.free / (1024**3),
                'utilization_percent': (disk_usage.used / disk_usage.total) * 100
            })

        except Exception as e:
            self.logger.error(f"Disk space check failed: {e}")

        return disk_info

    def _check_cpu_usage(self) -> Dict[str, float]:
        """Check CPU usage and load"""
        cpu_info = {
            'current_percent': 0.0,
            'avg_1min': 0.0,
            'avg_5min': 0.0,
            'avg_15min': 0.0,
            'core_count': 0,
            'frequency_mhz': 0.0
        }

        try:
            # Current CPU usage
            cpu_info['current_percent'] = psutil.cpu_percent(interval=1)

            # Load averages (Unix systems)
            try:
                load_avg = os.getloadavg()
                cpu_info.update({
                    'avg_1min': load_avg[0],
                    'avg_5min': load_avg[1],
                    'avg_15min': load_avg[2]
                })
            except:
                # Windows doesn't have loadavg
                pass

            # CPU info
            cpu_info['core_count'] = psutil.cpu_count()

            # CPU frequency
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                cpu_info['frequency_mhz'] = cpu_freq.current

        except Exception as e:
            self.logger.error(f"CPU usage check failed: {e}")

        return cpu_info

    def _check_memory_usage(self) -> Dict[str, float]:
        """Check system memory usage"""
        memory_info = {
            'total_gb': 0.0,
            'available_gb': 0.0,
            'used_gb': 0.0,
            'utilization_percent': 0.0,
            'swap_total_gb': 0.0,
            'swap_used_gb': 0.0,
            'swap_percent': 0.0
        }

        try:
            # Virtual memory
            virtual_mem = psutil.virtual_memory()
            memory_info.update({
                'total_gb': virtual_mem.total / (1024**3),
                'available_gb': virtual_mem.available / (1024**3),
                'used_gb': virtual_mem.used / (1024**3),
                'utilization_percent': virtual_mem.percent
            })

            # Swap memory
            swap_mem = psutil.swap_memory()
            memory_info.update({
                'swap_total_gb': swap_mem.total / (1024**3),
                'swap_used_gb': swap_mem.used / (1024**3),
                'swap_percent': swap_mem.percent
            })

        except Exception as e:
            self.logger.error(f"Memory usage check failed: {e}")

        return memory_info

    def _check_network(self) -> Dict[str, bool]:
        """Check network connectivity"""
        network_info = {
            'internet_available': False,
            'dns_resolution': False,
            'local_network': False
        }

        try:
            import socket
            import subprocess

            # Check internet connectivity
            try:
                socket.create_connection(("8.8.8.8", 53), timeout=3)
                network_info['internet_available'] = True
            except:
                pass

            # Check DNS resolution
            try:
                socket.gethostbyname("google.com")
                network_info['dns_resolution'] = True
            except:
                pass

            # Check local network (ping gateway)
            try:
                result = subprocess.run(['ping', '-c', '1', '8.8.8.8'],
                                      capture_output=True, timeout=5)
                network_info['local_network'] = result.returncode == 0
            except:
                pass

        except Exception as e:
            self.logger.error(f"Network check failed: {e}")

        return network_info

    def _verify_checkpoints(self) -> Dict[str, bool]:
        """Verify checkpoint integrity"""
        checkpoint_info = {
            'checkpoints_exist': False,
            'integrity_verified': False,
            'recent_checkpoint': False
        }

        try:
            # Check for checkpoint directory
            checkpoint_dir = Path("checkpoints")
            if checkpoint_dir.exists():
                checkpoint_info['checkpoints_exist'] = True

                # Check for recent checkpoints (within last hour)
                current_time = time.time()
                recent_checkpoints = []

                for checkpoint_file in checkpoint_dir.rglob("*.pt"):
                    try:
                        file_time = checkpoint_file.stat().st_mtime
                        if current_time - file_time < 3600:  # 1 hour
                            recent_checkpoints.append(checkpoint_file)
                    except:
                        continue

                checkpoint_info['recent_checkpoint'] = len(recent_checkpoints) > 0

                # Basic integrity check (file size > 0)
                valid_checkpoints = 0
                total_checkpoints = 0

                for checkpoint_file in checkpoint_dir.rglob("*.pt"):
                    total_checkpoints += 1
                    try:
                        if checkpoint_file.stat().st_size > 0:
                            valid_checkpoints += 1
                    except:
                        continue

                if total_checkpoints > 0:
                    checkpoint_info['integrity_verified'] = (valid_checkpoints / total_checkpoints) > 0.8

        except Exception as e:
            self.logger.error(f"Checkpoint verification failed: {e}")

        return checkpoint_info

    def _check_temperature(self) -> Dict[str, float]:
        """Check system temperatures if available"""
        temp_info = {
            'cpu_temp_c': 0.0,
            'gpu_temp_c': 0.0,
            'system_temp_c': 0.0
        }

        try:
            # Try to get temperature sensors
            temps = psutil.sensors_temperatures()

            if temps:
                # CPU temperature
                for name, entries in temps.items():
                    if 'cpu' in name.lower() or 'core' in name.lower():
                        if entries:
                            temp_info['cpu_temp_c'] = entries[0].current
                            break

                # System temperature
                if 'acpi' in temps:
                    temp_info['system_temp_c'] = temps['acpi'][0].current

            # GPU temperature (try nvidia-smi)
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    temp_info['gpu_temp_c'] = float(result.stdout.strip())
            except:
                pass

        except Exception as e:
            self.logger.error(f"Temperature check failed: {e}")

        return temp_info

    def predict_failure_risk(self, metrics: Dict[str, Any]) -> float:
        """Predict likelihood of training failure based on current metrics"""

        risk_score = 0.0
        max_risk = 0.0

        try:
            # GPU memory risk
            gpu_util = metrics.get('gpu_memory', {}).get('utilization_percent', 0)
            if gpu_util > self.failure_indicators['gpu_memory_critical'] * 100:
                risk_score += 0.3
            max_risk += 0.3

            # Disk space risk
            disk_util = metrics.get('disk_space', {}).get('utilization_percent', 0)
            if disk_util > self.failure_indicators['disk_space_critical'] * 100:
                risk_score += 0.2
            max_risk += 0.2

            # CPU usage risk
            cpu_util = metrics.get('cpu_usage', {}).get('current_percent', 0)
            if cpu_util > self.failure_indicators['cpu_usage_critical'] * 100:
                risk_score += 0.2
            max_risk += 0.2

            # Memory usage risk
            mem_util = metrics.get('memory_usage', {}).get('utilization_percent', 0)
            if mem_util > self.failure_indicators['memory_usage_critical'] * 100:
                risk_score += 0.15
            max_risk += 0.15

            # Temperature risk
            cpu_temp = metrics.get('temperature', {}).get('cpu_temp_c', 0)
            gpu_temp = metrics.get('temperature', {}).get('gpu_temp_c', 0)
            if cpu_temp > self.failure_indicators['temperature_critical'] or \
               gpu_temp > self.failure_indicators['temperature_critical']:
                risk_score += 0.1
            max_risk += 0.1

            # Network connectivity risk
            network = metrics.get('network_connectivity', {})
            if not network.get('internet_available', True):
                risk_score += 0.05
            max_risk += 0.05

            # Normalize risk score
            if max_risk > 0:
                risk_score = risk_score / max_risk

        except Exception as e:
            self.logger.error(f"Risk prediction failed: {e}")
            risk_score = 0.5  # Default moderate risk

        return min(1.0, max(0.0, risk_score))

    def recommend_actions(self, health_status: Dict[str, Any]) -> List[str]:
        """Recommend preventive actions based on health status"""

        recommendations = []

        try:
            # GPU memory recommendations
            gpu_util = health_status.get('gpu_memory', {}).get('utilization_percent', 0)
            if gpu_util > 85:
                recommendations.extend([
                    "Reduce batch size to lower GPU memory usage",
                    "Enable gradient checkpointing",
                    "Clear CUDA cache regularly",
                    "Consider using mixed precision training"
                ])

            # Disk space recommendations
            disk_util = health_status.get('disk_space', {}).get('utilization_percent', 0)
            if disk_util > 90:
                recommendations.extend([
                    "Clean up old checkpoint files",
                    "Enable checkpoint compression",
                    "Move large files to external storage",
                    "Implement automatic cleanup policies"
                ])

            # CPU usage recommendations
            cpu_util = health_status.get('cpu_usage', {}).get('current_percent', 0)
            if cpu_util > 85:
                recommendations.extend([
                    "Reduce number of data loading workers",
                    "Optimize data preprocessing pipeline",
                    "Consider using GPU for data augmentation",
                    "Monitor for CPU-bound operations"
                ])

            # Memory recommendations
            mem_util = health_status.get('memory_usage', {}).get('utilization_percent', 0)
            if mem_util > 85:
                recommendations.extend([
                    "Reduce dataset caching",
                    "Use streaming data loading",
                    "Implement memory-mapped datasets",
                    "Monitor for memory leaks"
                ])

            # Temperature recommendations
            cpu_temp = health_status.get('temperature', {}).get('cpu_temp_c', 0)
            gpu_temp = health_status.get('temperature', {}).get('gpu_temp_c', 0)
            if cpu_temp > 80 or gpu_temp > 80:
                recommendations.extend([
                    "Improve system cooling",
                    "Reduce training intensity temporarily",
                    "Check for dust in cooling systems",
                    "Monitor thermal throttling"
                ])

            # Network recommendations
            network = health_status.get('network_connectivity', {})
            if not network.get('internet_available', True):
                recommendations.extend([
                    "Enable offline mode if available",
                    "Cache required data locally",
                    "Check network configuration",
                    "Use local mirrors for downloads"
                ])

            # General recommendations if high risk
            risk_score = health_status.get('prediction_score', 0)
            if risk_score > 0.7:
                recommendations.extend([
                    "Save checkpoints more frequently",
                    "Enable automatic recovery mode",
                    "Monitor system continuously",
                    "Prepare fallback configurations"
                ])

        except Exception as e:
            self.logger.error(f"Action recommendation failed: {e}")
            recommendations.append("Monitor system health manually")

        return recommendations

    def start_continuous_monitoring(self):
        """Start continuous health monitoring"""
        self.monitoring_active = True
        self.logger.info("Continuous health monitoring started")

    def stop_continuous_monitoring(self):
        """Stop continuous health monitoring"""
        self.monitoring_active = False
        self.logger.info("Continuous health monitoring stopped")

    def get_health_trend(self, hours: int = 24) -> Dict[str, List[float]]:
        """Get health trends over specified time period"""

        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [m for m in self.health_history if m.timestamp > cutoff_time]

        trends = {
            'timestamps': [],
            'gpu_memory_util': [],
            'disk_space_util': [],
            'cpu_usage': [],
            'memory_usage': [],
            'risk_score': []
        }

        for metric in recent_metrics:
            trends['timestamps'].append(metric.timestamp)
            trends['gpu_memory_util'].append(metric.gpu_memory.get('utilization_percent', 0))
            trends['disk_space_util'].append(metric.disk_space.get('utilization_percent', 0))
            trends['cpu_usage'].append(metric.cpu_usage.get('current_percent', 0))
            trends['memory_usage'].append(metric.memory_usage.get('utilization_percent', 0))
            trends['risk_score'].append(metric.prediction_score)

        return trends

    def _get_fallback_health_status(self) -> Dict[str, Any]:
        """Get fallback health status when monitoring fails"""
        return {
            'timestamp': time.time(),
            'gpu_memory': {'available_gb': 0, 'utilization_percent': 0},
            'disk_space': {'available_gb': 0, 'utilization_percent': 0},
            'cpu_usage': {'current_percent': 0},
            'memory_usage': {'utilization_percent': 0},
            'network_connectivity': {'internet_available': False},
            'checkpoint_integrity': {'checkpoints_exist': False},
            'temperature': {'cpu_temp_c': 0, 'gpu_temp_c': 0},
            'prediction_score': 0.5,
            'monitoring_error': True
        }

    def export_health_report(self, filepath: str):
        """Export comprehensive health report"""
        try:
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'monitoring_duration_hours': (time.time() - self.health_history[0].timestamp) / 3600 if self.health_history else 0,
                'total_snapshots': len(self.health_history),
                'current_health': self.check_system_health(),
                'health_trends': self.get_health_trend(24),
                'failure_indicators': self.failure_indicators,
                'recommendations': self.recommend_actions(self.check_system_health()) if self.health_history else []
            }

            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)

            self.logger.info(f"Health report exported to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to export health report: {e}")


# Export the monitor class
__all__ = ['SystemHealthMonitor', 'SystemHealthMetrics']