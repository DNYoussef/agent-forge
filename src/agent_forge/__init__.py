"""Agent Forge - Parallel/Series Training Configuration System"""

__version__ = "1.0.0"
__author__ = "SPEK Platform"

# Import available components (removed broken cognate_creator import)
try:
    from .utils.resource_manager import ResourceManager
    from .utils.progress_aggregator import ProgressAggregator
    __all__ = ['ResourceManager', 'ProgressAggregator']
except ImportError:
    # Fallback if utils are not available
    __all__ = []
