"""Agent Forge - Parallel/Series Training Configuration System"""

__version__ = "1.0.0"
__author__ = "SPEK Platform"

from .phases.cognate_pretrain.cognate_creator import CognateCreator
from .utils.resource_manager import ResourceManager
from .utils.progress_aggregator import ProgressAggregator

__all__ = ['CognateCreator', 'ResourceManager', 'ProgressAggregator']
