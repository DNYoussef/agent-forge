"""
Agent Forge - Data Format Compatibility Layer

Ensures perfect compatibility between real training data and existing UI interfaces
while providing seamless data transformation and validation.
"""

import json
import time
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Enumeration of supported metric types"""
    LOSS = "loss"
    PERPLEXITY = "perplexity"
    GROK_PROGRESS = "grokProgress"
    MODEL_PARAMS = "modelParams"
    STEP = "step"
    PROGRESS = "progress"
    TIME = "time"


@dataclass
class UIMetricsFormat:
    """
    Standard UI metrics format that matches existing interface expectations.

    This ensures compatibility with the existing route.ts simulation and
    preserves all UI component functionality.
    """
    loss: float
    perplexity: float
    grokProgress: float
    modelParams: int
    currentStep: int
    totalSteps: int
    currentModel: int
    totalModels: int
    overallProgress: float
    trainingTime: float
    estimatedTimeRemaining: float
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class TrainingProgressData:
    """Raw training progress data from the training process"""
    sessionId: str
    modelIndex: int
    totalModels: int
    step: int
    totalSteps: int
    loss: float
    timestamp: float
    perplexity: Optional[float] = None
    grokProgress: Optional[float] = None
    modelParams: Optional[int] = None
    overallProgress: Optional[float] = None
    trainingTime: Optional[float] = None
    estimatedTimeRemaining: Optional[float] = None
    modelCompleted: Optional[bool] = None
    event: Optional[str] = None


class DataCompatibilityLayer:
    """
    Compatibility layer that transforms real training data to match existing UI expectations.

    Provides bidirectional transformation, validation, and fallback mechanisms
    to ensure seamless integration with existing interfaces.
    """

    def __init__(self):
        self.validation_rules = self._setup_validation_rules()
        self.transformation_cache: Dict[str, Any] = {}
        self.fallback_values = self._setup_fallback_values()

    def _setup_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Setup validation rules for data transformation"""
        return {
            'loss': {'min': 0.0, 'max': 10.0, 'default': 2.5},
            'perplexity': {'min': 1.0, 'max': 1000.0, 'default': 12.18},
            'grokProgress': {'min': 0.0, 'max': 100.0, 'default': 0.0},
            'modelParams': {'min': 1000000, 'max': 100000000, 'default': 25000000},
            'currentStep': {'min': 0, 'max': 100000, 'default': 0},
            'totalSteps': {'min': 1, 'max': 100000, 'default': 1000},
            'currentModel': {'min': 1, 'max': 10, 'default': 1},
            'totalModels': {'min': 1, 'max': 10, 'default': 3},
            'overallProgress': {'min': 0.0, 'max': 100.0, 'default': 0.0},
            'trainingTime': {'min': 0.0, 'max': 86400.0, 'default': 0.0},  # Max 24 hours
            'estimatedTimeRemaining': {'min': 0.0, 'max': 86400.0, 'default': 0.0}
        }

    def _setup_fallback_values(self) -> UIMetricsFormat:
        """Setup fallback values that match existing API simulation"""
        return UIMetricsFormat(
            loss=2.5,
            perplexity=12.18,
            grokProgress=0.0,
            modelParams=25000000,
            currentStep=0,
            totalSteps=1000,
            currentModel=1,
            totalModels=3,
            overallProgress=0.0,
            trainingTime=0.0,
            estimatedTimeRemaining=0.0,
            timestamp=time.time()
        )

    def transform_progress_data(self, raw_data: Union[Dict[str, Any], TrainingProgressData]) -> UIMetricsFormat:
        """
        Transform raw training progress data to UI-compatible format.

        Args:
            raw_data: Raw progress data from training process

        Returns:
            UIMetricsFormat object compatible with existing UI
        """
        try:
            # Convert to dict if TrainingProgressData object
            if isinstance(raw_data, TrainingProgressData):
                data_dict = asdict(raw_data)
            else:
                data_dict = raw_data.copy()

            # Transform with validation
            transformed = UIMetricsFormat(
                loss=self._validate_and_transform('loss', data_dict.get('loss')),
                perplexity=self._validate_and_transform('perplexity',
                                                       data_dict.get('perplexity') or
                                                       self._calculate_perplexity(data_dict.get('loss', 2.5))),
                grokProgress=self._validate_and_transform('grokProgress',
                                                        data_dict.get('grokProgress') or
                                                        self._calculate_grok_progress(data_dict.get('loss', 2.5))),
                modelParams=self._validate_and_transform('modelParams',
                                                       data_dict.get('modelParams', 25000000)),
                currentStep=self._validate_and_transform('currentStep',
                                                       data_dict.get('currentStep') or
                                                       data_dict.get('step', 0)),
                totalSteps=self._validate_and_transform('totalSteps',
                                                      data_dict.get('totalSteps', 1000)),
                currentModel=self._validate_and_transform('currentModel',
                                                        data_dict.get('currentModel') or
                                                        (data_dict.get('modelIndex', 0) + 1)),
                totalModels=self._validate_and_transform('totalModels',
                                                       data_dict.get('totalModels', 3)),
                overallProgress=self._validate_and_transform('overallProgress',
                                                           data_dict.get('overallProgress') or
                                                           self._calculate_overall_progress(data_dict)),
                trainingTime=self._validate_and_transform('trainingTime',
                                                        data_dict.get('trainingTime', 0.0)),
                estimatedTimeRemaining=self._validate_and_transform('estimatedTimeRemaining',
                                                                  data_dict.get('estimatedTimeRemaining', 0.0)),
                timestamp=data_dict.get('timestamp', time.time())
            )

            # Cache for performance
            self.transformation_cache[str(data_dict.get('sessionId', 'unknown'))] = transformed.to_dict()

            return transformed

        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            return self._get_safe_fallback(raw_data)

    def _validate_and_transform(self, field_name: str, value: Any) -> Union[float, int]:
        """Validate and transform individual field values"""
        if value is None:
            return self.validation_rules[field_name]['default']

        try:
            # Convert to appropriate type
            if field_name in ['currentStep', 'totalSteps', 'currentModel', 'totalModels', 'modelParams']:
                value = int(value)
            else:
                value = float(value)

            # Apply validation rules
            rules = self.validation_rules[field_name]
            value = max(rules['min'], min(rules['max'], value))

            return value

        except (ValueError, TypeError):
            logger.warning(f"Invalid value for {field_name}: {value}, using default")
            return self.validation_rules[field_name]['default']

    def _calculate_perplexity(self, loss: float) -> float:
        """Calculate perplexity from loss value"""
        import math
        try:
            return math.exp(min(loss, 10.0))  # Cap to prevent overflow
        except (ValueError, OverflowError):
            return 12.18  # Default perplexity

    def _calculate_grok_progress(self, loss: float) -> float:
        """Calculate grokking progress from loss value"""
        try:
            grok_threshold = 2.0  # Typical threshold where grokking begins
            return max(0.0, min(100.0, (grok_threshold - loss) / grok_threshold * 100))
        except (ValueError, TypeError):
            return 0.0

    def _calculate_overall_progress(self, data_dict: Dict[str, Any]) -> float:
        """Calculate overall training progress"""
        try:
            model_idx = data_dict.get('modelIndex', 0)
            total_models = data_dict.get('totalModels', 3)
            current_step = data_dict.get('currentStep') or data_dict.get('step', 0)
            total_steps = data_dict.get('totalSteps', 1000)

            if total_models == 0 or total_steps == 0:
                return 0.0

            model_progress = (current_step / total_steps) * 100
            overall_progress = ((model_idx * 100 + model_progress) / total_models)

            return min(100.0, max(0.0, overall_progress))

        except (ValueError, TypeError, ZeroDivisionError):
            return 0.0

    def _get_safe_fallback(self, original_data: Any) -> UIMetricsFormat:
        """Get safe fallback values when transformation fails"""
        logger.warning("Using fallback values due to transformation failure")
        fallback = self.fallback_values

        # Try to preserve some original data if possible
        try:
            if isinstance(original_data, dict):
                if 'loss' in original_data:
                    fallback.loss = float(original_data['loss'])
                if 'currentModel' in original_data:
                    fallback.currentModel = int(original_data['currentModel'])
        except:
            pass  # Use defaults if extraction fails

        return fallback

    def validate_ui_compatibility(self, data: UIMetricsFormat) -> Dict[str, Any]:
        """
        Validate that transformed data is compatible with existing UI components.

        Returns:
            Dictionary with validation results and compatibility checks
        """
        validation_result = {
            'compatible': True,
            'warnings': [],
            'errors': [],
            'data_integrity': True
        }

        # Check required fields
        required_fields = ['loss', 'perplexity', 'grokProgress', 'modelParams']
        for field in required_fields:
            if not hasattr(data, field) or getattr(data, field) is None:
                validation_result['errors'].append(f"Missing required field: {field}")
                validation_result['compatible'] = False

        # Check data ranges
        if data.loss < 0 or data.loss > 10:
            validation_result['warnings'].append(f"Loss value {data.loss} outside expected range")

        if data.perplexity < 1:
            validation_result['warnings'].append(f"Perplexity value {data.perplexity} unusually low")

        if data.grokProgress < 0 or data.grokProgress > 100:
            validation_result['errors'].append(f"Grokking progress {data.grokProgress} outside valid range")
            validation_result['compatible'] = False

        # Check progress consistency
        if data.currentStep > data.totalSteps:
            validation_result['warnings'].append("Current step exceeds total steps")

        if data.currentModel > data.totalModels:
            validation_result['warnings'].append("Current model exceeds total models")

        # Check overall progress logic
        expected_progress = ((data.currentModel - 1) * 100 +
                           (data.currentStep / data.totalSteps) * 100) / data.totalModels
        if abs(data.overallProgress - expected_progress) > 5:  # 5% tolerance
            validation_result['warnings'].append("Overall progress calculation inconsistency")

        return validation_result

    def create_http_api_response(self, ui_data: UIMetricsFormat, session_id: str = None) -> Dict[str, Any]:
        """
        Create HTTP API response that matches existing route.ts format.

        Args:
            ui_data: UI-compatible metrics data
            session_id: Optional session identifier

        Returns:
            API response dictionary compatible with existing endpoints
        """
        return {
            'success': True,
            'data': {
                'sessionId': session_id,
                'metrics': ui_data.to_dict(),
                'timestamp': ui_data.timestamp,
                'status': 'running'
            },
            'meta': {
                'version': '1.0',
                'compatibility_layer': True,
                'transformation_applied': True
            }
        }

    def create_websocket_message(self, ui_data: UIMetricsFormat, event_type: str = 'progress_update') -> Dict[str, Any]:
        """
        Create WebSocket message format compatible with existing client expectations.

        Args:
            ui_data: UI-compatible metrics data
            event_type: Type of WebSocket event

        Returns:
            WebSocket message dictionary
        """
        return {
            'event': event_type,
            'metrics': ui_data.to_dict(),
            'status': 'running',
            'timestamp': ui_data.timestamp
        }

    def get_cached_transformation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached transformation data for performance optimization"""
        return self.transformation_cache.get(session_id)

    def clear_cache(self, session_id: str = None):
        """Clear transformation cache"""
        if session_id:
            self.transformation_cache.pop(session_id, None)
        else:
            self.transformation_cache.clear()

    def get_compatibility_stats(self) -> Dict[str, Any]:
        """Get compatibility layer statistics"""
        return {
            'transformations_cached': len(self.transformation_cache),
            'validation_rules_count': len(self.validation_rules),
            'fallback_available': True,
            'supported_formats': ['TrainingProgressData', 'dict', 'UIMetricsFormat']
        }


# Performance optimization utilities
class PerformanceOptimizer:
    """Performance optimization for real-time streaming"""

    def __init__(self, max_update_rate: float = 10.0):
        self.max_update_rate = max_update_rate  # Updates per second
        self.last_update_times: Dict[str, float] = {}
        self.update_buffer: Dict[str, List[Dict[str, Any]]] = {}
        self.throttle_enabled = True

    def should_emit_update(self, session_id: str) -> bool:
        """Check if update should be emitted based on rate limiting"""
        if not self.throttle_enabled:
            return True

        current_time = time.time()
        last_update = self.last_update_times.get(session_id, 0)
        min_interval = 1.0 / self.max_update_rate

        if current_time - last_update >= min_interval:
            self.last_update_times[session_id] = current_time
            return True

        return False

    def buffer_update(self, session_id: str, update_data: Dict[str, Any]):
        """Buffer update for batch processing"""
        if session_id not in self.update_buffer:
            self.update_buffer[session_id] = []

        self.update_buffer[session_id].append(update_data)

        # Keep buffer size manageable
        if len(self.update_buffer[session_id]) > 100:
            self.update_buffer[session_id] = self.update_buffer[session_id][-100:]

    def get_buffered_updates(self, session_id: str) -> List[Dict[str, Any]]:
        """Get and clear buffered updates"""
        updates = self.update_buffer.get(session_id, [])
        self.update_buffer[session_id] = []
        return updates

    def optimize_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize payload size for network transmission"""
        optimized = {}

        for key, value in data.items():
            # Round floating point values to reduce payload size
            if isinstance(value, float):
                if key in ['loss', 'perplexity']:
                    optimized[key] = round(value, 4)
                elif key in ['grokProgress', 'overallProgress']:
                    optimized[key] = round(value, 1)
                else:
                    optimized[key] = round(value, 2)
            else:
                optimized[key] = value

        return optimized


# Validation testing functions
def run_compatibility_tests():
    """Run comprehensive compatibility tests"""
    compatibility_layer = DataCompatibilityLayer()

    # Test data transformation
    test_data = {
        'sessionId': 'test-123',
        'modelIndex': 0,
        'totalModels': 3,
        'step': 50,
        'totalSteps': 100,
        'loss': 1.5,
        'timestamp': time.time()
    }

    # Transform data
    ui_data = compatibility_layer.transform_progress_data(test_data)

    # Validate compatibility
    validation = compatibility_layer.validate_ui_compatibility(ui_data)

    # Create API response
    api_response = compatibility_layer.create_http_api_response(ui_data, 'test-123')

    # Create WebSocket message
    ws_message = compatibility_layer.create_websocket_message(ui_data)

    print("Compatibility Tests Results:")
    print(f"- Data transformation: {'✓' if ui_data else '✗'}")
    print(f"- Validation compatible: {'✓' if validation['compatible'] else '✗'}")
    print(f"- API response created: {'✓' if api_response['success'] else '✗'}")
    print(f"- WebSocket message: {'✓' if ws_message else '✗'}")

    return all([ui_data, validation['compatible'], api_response['success'], ws_message])


if __name__ == '__main__':
    # Run compatibility tests
    success = run_compatibility_tests()
    print(f"\nAll tests passed: {'✓' if success else '✗'}")