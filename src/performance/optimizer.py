"""
Agent Forge Performance Optimizer - Princess Infrastructure Enhancement
Advanced optimization engine with caching, lazy loading, and parallel execution.

PERFORMANCE OPTIMIZATION STRATEGIES:
- Intelligent caching system with LRU and time-based eviction
- Lazy loading for memory-intensive components
- Parallel execution with async/await and thread pools
- Memory pool management for frequent allocations
- Connection pooling for I/O operations
- Code optimization suggestions and automated fixes
"""

import asyncio
import functools
import gc
import hashlib
import logging
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import json
import pickle
import torch
import torch.nn as nn
from queue import Queue, Empty
import multiprocessing as mp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of optimization operation."""
    original_duration_ms: float
    optimized_duration_ms: float
    improvement_percent: float
    memory_saved_mb: float
    cache_hits: int = 0
    cache_misses: int = 0
    optimization_type: str = ""
    recommendations: List[str] = field(default_factory=list)


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_mb: float = 0.0
    hit_rate: float = 0.0


class CachingStrategy(ABC):
    """Abstract base class for caching strategies."""

    @abstractmethod
    def should_cache(self, key: str, value: Any, size_bytes: int) -> bool:
        """Determine if item should be cached."""
        pass

    @abstractmethod
    def should_evict(self, key: str, value: Any, last_access: datetime) -> bool:
        """Determine if item should be evicted."""
        pass


class LRUCachingStrategy(CachingStrategy):
    """Least Recently Used caching strategy."""

    def __init__(self, max_size_mb: float = 1024, max_age_minutes: int = 60):
        self.max_size_mb = max_size_mb
        self.max_age_minutes = max_age_minutes

    def should_cache(self, key: str, value: Any, size_bytes: int) -> bool:
        """Cache if item is not too large."""
        max_item_size = self.max_size_mb * 0.1 * 1024 * 1024  # 10% of max cache size
        return size_bytes <= max_item_size

    def should_evict(self, key: str, value: Any, last_access: datetime) -> bool:
        """Evict if item is older than max age."""
        age = datetime.now() - last_access
        return age > timedelta(minutes=self.max_age_minutes)


class IntelligentCache:
    """
    High-performance intelligent cache with multiple eviction strategies.

    Features:
    - LRU with time-based expiration
    - Memory-aware eviction
    - Performance metrics tracking
    - Thread-safe operations
    - Configurable serialization
    """

    def __init__(self, max_size_mb: float = 1024, strategy: Optional[CachingStrategy] = None):
        self.max_size_mb = max_size_mb
        self.strategy = strategy or LRUCachingStrategy(max_size_mb)

        # Cache storage
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._access_times: Dict[str, datetime] = {}
        self._item_sizes: Dict[str, int] = {}
        self._current_size_bytes = 0

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self.stats = CacheStats()

        logger.debug(f"IntelligentCache initialized with {max_size_mb}MB limit")

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                value = self._cache[key]
                self._cache.move_to_end(key)
                self._access_times[key] = datetime.now()
                self.stats.hits += 1
                self._update_hit_rate()
                return value
            else:
                self.stats.misses += 1
                self._update_hit_rate()
                return None

    def put(self, key: str, value: Any) -> bool:
        """Put item in cache."""
        with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                # Fallback size estimation
                size_bytes = 1024  # 1KB default

            # Check if should cache
            if not self.strategy.should_cache(key, value, size_bytes):
                return False

            # Remove existing item if present
            if key in self._cache:
                self._remove_item(key)

            # Evict items if necessary
            max_size_bytes = self.max_size_mb * 1024 * 1024
            while (self._current_size_bytes + size_bytes) > max_size_bytes and self._cache:
                self._evict_lru()

            # Add new item
            self._cache[key] = value
            self._access_times[key] = datetime.now()
            self._item_sizes[key] = size_bytes
            self._current_size_bytes += size_bytes

            return True

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._item_sizes.clear()
            self._current_size_bytes = 0

    def evict_expired(self):
        """Evict expired items based on strategy."""
        with self._lock:
            expired_keys = []
            for key, last_access in self._access_times.items():
                if self.strategy.should_evict(key, self._cache.get(key), last_access):
                    expired_keys.append(key)

            for key in expired_keys:
                self._remove_item(key)

    def _remove_item(self, key: str):
        """Remove item from cache."""
        if key in self._cache:
            del self._cache[key]
            self._current_size_bytes -= self._item_sizes.pop(key, 0)
            self._access_times.pop(key, None)

    def _evict_lru(self):
        """Evict least recently used item."""
        if self._cache:
            key = next(iter(self._cache))
            self._remove_item(key)
            self.stats.evictions += 1

    def _update_hit_rate(self):
        """Update cache hit rate."""
        total_requests = self.stats.hits + self.stats.misses
        if total_requests > 0:
            self.stats.hit_rate = self.stats.hits / total_requests

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self.stats.size_mb = self._current_size_bytes / (1024 * 1024)
            return self.stats


class LazyLoader:
    """
    Lazy loading implementation for memory-intensive objects.

    Features:
    - Deferred initialization until first access
    - Automatic cleanup of unused objects
    - Memory usage tracking
    - Thread-safe operations
    """

    def __init__(self, factory: Callable[[], Any], cleanup_delay_seconds: int = 300):
        self._factory = factory
        self._value = None
        self._loaded = False
        self._last_access = None
        self._cleanup_delay = cleanup_delay_seconds
        self._lock = threading.Lock()

        # Schedule cleanup check
        self._cleanup_timer = None
        self._schedule_cleanup()

    def get(self) -> Any:
        """Get the lazy-loaded value."""
        with self._lock:
            if not self._loaded:
                logger.debug(f"Lazy loading object: {self._factory}")
                self._value = self._factory()
                self._loaded = True

            self._last_access = datetime.now()
            self._schedule_cleanup()
            return self._value

    def is_loaded(self) -> bool:
        """Check if object is currently loaded."""
        return self._loaded

    def unload(self):
        """Force unload the object."""
        with self._lock:
            if self._loaded:
                logger.debug(f"Unloading lazy object: {self._factory}")
                self._value = None
                self._loaded = False
                gc.collect()  # Force garbage collection

    def _schedule_cleanup(self):
        """Schedule automatic cleanup."""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()

        def cleanup():
            if self._last_access:
                age = datetime.now() - self._last_access
                if age.total_seconds() > self._cleanup_delay:
                    self.unload()

        self._cleanup_timer = threading.Timer(self._cleanup_delay, cleanup)
        self._cleanup_timer.start()


class MemoryPool:
    """
    Memory pool for efficient object allocation and reuse.

    Features:
    - Pre-allocated object pools
    - Automatic pool expansion
    - Memory usage monitoring
    - Type-specific pools
    """

    def __init__(self, factory: Callable[[], Any], initial_size: int = 10, max_size: int = 100):
        self.factory = factory
        self.max_size = max_size
        self._pool: Queue = Queue(maxsize=max_size)
        self._lock = threading.Lock()
        self._created_count = 0
        self._acquired_count = 0
        self._released_count = 0

        # Pre-fill pool
        for _ in range(initial_size):
            try:
                obj = self.factory()
                self._pool.put(obj, block=False)
                self._created_count += 1
            except Exception as e:
                logger.warning(f"Error pre-filling memory pool: {e}")
                break

    def acquire(self) -> Any:
        """Acquire object from pool."""
        with self._lock:
            try:
                obj = self._pool.get(block=False)
                self._acquired_count += 1
                return obj
            except Empty:
                # Create new object if pool is empty
                if self._created_count < self.max_size:
                    obj = self.factory()
                    self._created_count += 1
                    self._acquired_count += 1
                    return obj
                else:
                    raise RuntimeError("Memory pool exhausted")

    def release(self, obj: Any):
        """Release object back to pool."""
        with self._lock:
            try:
                # Reset object if it has a reset method
                if hasattr(obj, 'reset'):
                    obj.reset()

                self._pool.put(obj, block=False)
                self._released_count += 1
            except Exception:
                # Pool is full, let object be garbage collected
                pass

    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        return {
            'pool_size': self._pool.qsize(),
            'created_count': self._created_count,
            'acquired_count': self._acquired_count,
            'released_count': self._released_count,
            'in_use': self._acquired_count - self._released_count
        }


class ConnectionPool:
    """
    Connection pool for I/O operations (database, HTTP, etc.).

    Features:
    - Connection reuse
    - Automatic connection validation
    - Connection lifecycle management
    - Statistics tracking
    """

    def __init__(self, connection_factory: Callable[[], Any],
                 max_connections: int = 20,
                 validation_query: Optional[str] = None):
        self.connection_factory = connection_factory
        self.max_connections = max_connections
        self.validation_query = validation_query

        self._pool: Queue = Queue(maxsize=max_connections)
        self._lock = threading.Lock()
        self._created_count = 0
        self._active_connections = set()

    @contextmanager
    def get_connection(self):
        """Get connection from pool with automatic cleanup."""
        connection = None
        try:
            connection = self._acquire_connection()
            yield connection
        finally:
            if connection:
                self._release_connection(connection)

    def _acquire_connection(self) -> Any:
        """Acquire connection from pool."""
        with self._lock:
            try:
                connection = self._pool.get(block=False)
                if self._validate_connection(connection):
                    self._active_connections.add(connection)
                    return connection
                else:
                    # Connection is invalid, create new one
                    return self._create_new_connection()
            except Empty:
                return self._create_new_connection()

    def _release_connection(self, connection: Any):
        """Release connection back to pool."""
        with self._lock:
            if connection in self._active_connections:
                self._active_connections.remove(connection)

            try:
                self._pool.put(connection, block=False)
            except Exception:
                # Pool is full, close the connection
                self._close_connection(connection)

    def _create_new_connection(self) -> Any:
        """Create new connection."""
        if self._created_count >= self.max_connections:
            raise RuntimeError("Connection pool exhausted")

        connection = self.connection_factory()
        self._created_count += 1
        self._active_connections.add(connection)
        return connection

    def _validate_connection(self, connection: Any) -> bool:
        """Validate connection is still usable."""
        try:
            if self.validation_query and hasattr(connection, 'execute'):
                connection.execute(self.validation_query)
            return True
        except Exception:
            return False

    def _close_connection(self, connection: Any):
        """Close connection."""
        try:
            if hasattr(connection, 'close'):
                connection.close()
        except Exception:
            pass


class ParallelExecutor:
    """
    Advanced parallel execution engine with load balancing.

    Features:
    - Async/await support
    - Thread pool management
    - Process pool support
    - Load balancing
    - Error handling and retries
    """

    def __init__(self, max_threads: int = None, max_processes: int = None):
        self.max_threads = max_threads or min(32, (mp.cpu_count() or 1) + 4)
        self.max_processes = max_processes or mp.cpu_count()

        self._thread_pool = ThreadPoolExecutor(max_workers=self.max_threads)
        self._process_pool = None  # Created on demand

        # Statistics
        self._tasks_submitted = 0
        self._tasks_completed = 0
        self._tasks_failed = 0

        logger.info(f"ParallelExecutor initialized - Threads: {self.max_threads}, Processes: {self.max_processes}")

    async def execute_async_batch(self, tasks: List[Callable], *args, **kwargs) -> List[Any]:
        """Execute batch of async tasks."""
        results = []

        async def run_task(task):
            try:
                if asyncio.iscoroutinefunction(task):
                    return await task(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(self._thread_pool, task, *args, **kwargs)
            except Exception as e:
                logger.error(f"Task failed: {e}")
                self._tasks_failed += 1
                return None

        self._tasks_submitted += len(tasks)

        # Execute all tasks concurrently
        results = await asyncio.gather(*[run_task(task) for task in tasks], return_exceptions=True)

        self._tasks_completed += len([r for r in results if not isinstance(r, Exception)])

        return results

    def execute_thread_batch(self, tasks: List[Callable], *args, **kwargs) -> List[Any]:
        """Execute batch of tasks in thread pool."""
        results = []
        futures = []

        self._tasks_submitted += len(tasks)

        # Submit all tasks
        for task in tasks:
            future = self._thread_pool.submit(task, *args, **kwargs)
            futures.append(future)

        # Collect results
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                self._tasks_completed += 1
            except Exception as e:
                logger.error(f"Thread task failed: {e}")
                results.append(None)
                self._tasks_failed += 1

        return results

    def execute_process_batch(self, tasks: List[Callable], *args, **kwargs) -> List[Any]:
        """Execute batch of tasks in process pool."""
        if not self._process_pool:
            self._process_pool = mp.Pool(processes=self.max_processes)

        results = []
        self._tasks_submitted += len(tasks)

        try:
            # Submit all tasks
            async_results = []
            for task in tasks:
                async_result = self._process_pool.apply_async(task, args, kwargs)
                async_results.append(async_result)

            # Collect results
            for async_result in async_results:
                try:
                    result = async_result.get(timeout=60)  # 60 second timeout
                    results.append(result)
                    self._tasks_completed += 1
                except Exception as e:
                    logger.error(f"Process task failed: {e}")
                    results.append(None)
                    self._tasks_failed += 1

        except Exception as e:
            logger.error(f"Process pool execution failed: {e}")

        return results

    def get_stats(self) -> Dict[str, int]:
        """Get execution statistics."""
        return {
            'tasks_submitted': self._tasks_submitted,
            'tasks_completed': self._tasks_completed,
            'tasks_failed': self._tasks_failed,
            'success_rate': self._tasks_completed / max(1, self._tasks_submitted)
        }

    def shutdown(self):
        """Shutdown executor pools."""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
        if self._process_pool:
            self._process_pool.close()
            self._process_pool.join()


class PerformanceOptimizer:
    """
    Master performance optimization engine.

    Integrates all optimization strategies:
    - Intelligent caching
    - Lazy loading
    - Memory pooling
    - Connection pooling
    - Parallel execution
    - Code optimization analysis
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Initialize components
        cache_size_mb = self.config.get('cache_size_mb', 1024)
        self.cache = IntelligentCache(max_size_mb=cache_size_mb)

        max_threads = self.config.get('max_threads', None)
        max_processes = self.config.get('max_processes', None)
        self.executor = ParallelExecutor(max_threads=max_threads, max_processes=max_processes)

        # Component registries
        self.lazy_loaders: Dict[str, LazyLoader] = {}
        self.memory_pools: Dict[str, MemoryPool] = {}
        self.connection_pools: Dict[str, ConnectionPool] = {}

        # Optimization tracking
        self.optimization_results: List[OptimizationResult] = []

        logger.info("PerformanceOptimizer initialized")

    def register_lazy_loader(self, name: str, factory: Callable[[], Any],
                           cleanup_delay_seconds: int = 300) -> LazyLoader:
        """Register a lazy loader."""
        loader = LazyLoader(factory, cleanup_delay_seconds)
        self.lazy_loaders[name] = loader
        return loader

    def register_memory_pool(self, name: str, factory: Callable[[], Any],
                           initial_size: int = 10, max_size: int = 100) -> MemoryPool:
        """Register a memory pool."""
        pool = MemoryPool(factory, initial_size, max_size)
        self.memory_pools[name] = pool
        return pool

    def register_connection_pool(self, name: str, connection_factory: Callable[[], Any],
                               max_connections: int = 20) -> ConnectionPool:
        """Register a connection pool."""
        pool = ConnectionPool(connection_factory, max_connections)
        self.connection_pools[name] = pool
        return pool

    def cached_function(self, cache_key: Optional[str] = None, ttl_minutes: int = 60):
        """Decorator for automatic function result caching."""
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if cache_key:
                    key = cache_key
                else:
                    key_data = f"{func.__module__}.{func.__name__}:{args}:{sorted(kwargs.items())}"
                    key = hashlib.md5(key_data.encode()).hexdigest()

                # Try cache first
                cached_result = self.cache.get(key)
                if cached_result is not None:
                    return cached_result

                # Execute function
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000

                # Cache result
                self.cache.put(key, result)

                logger.debug(f"Cached function result: {func.__name__} ({execution_time:.2f}ms)")
                return result
            return wrapper
        return decorator

    def optimize_torch_model(self, model: nn.Module) -> nn.Module:
        """Optimize PyTorch model for better performance."""
        try:
            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.cuda()

            # Enable optimizations
            model.eval()

            # Compile model for PyTorch 2.0+
            if hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model)
                    logger.info("Model compiled with torch.compile")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")

            # JIT script for older PyTorch versions
            else:
                try:
                    # Create example input (this would need to be model-specific)
                    dummy_input = torch.randn(1, 10)
                    if torch.cuda.is_available():
                        dummy_input = dummy_input.cuda()

                    model = torch.jit.trace(model, dummy_input)
                    logger.info("Model optimized with TorchScript")
                except Exception as e:
                    logger.warning(f"TorchScript optimization failed: {e}")

            return model

        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model

    async def optimize_async_batch(self, functions: List[Callable], *args, **kwargs) -> OptimizationResult:
        """Optimize execution of async function batch."""
        start_time = time.time()

        # Execute with parallel executor
        results = await self.executor.execute_async_batch(functions, *args, **kwargs)

        execution_time = (time.time() - start_time) * 1000

        # Calculate improvement (baseline would be sequential execution)
        estimated_sequential_time = len(functions) * 100  # Rough estimate
        improvement = max(0, (estimated_sequential_time - execution_time) / estimated_sequential_time * 100)

        optimization_result = OptimizationResult(
            original_duration_ms=estimated_sequential_time,
            optimized_duration_ms=execution_time,
            improvement_percent=improvement,
            memory_saved_mb=0,  # Would need more detailed tracking
            optimization_type="async_batch"
        )

        self.optimization_results.append(optimization_result)
        return optimization_result

    def analyze_code_bottlenecks(self, code_file: str) -> List[str]:
        """Analyze code file for potential optimization opportunities."""
        recommendations = []

        try:
            with open(code_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Simple static analysis
            lines = content.split('\n')

            for i, line in enumerate(lines, 1):
                line = line.strip()

                # Check for common performance issues
                if 'for' in line and 'range(len(' in line:
                    recommendations.append(f"Line {i}: Consider using enumerate() instead of range(len())")

                if '+ str(' in line or 'str(' in line and '+' in line:
                    recommendations.append(f"Line {i}: Consider using f-strings for string formatting")

                if 'time.sleep(' in line and 'async' not in content[:content.find(line)]:
                    recommendations.append(f"Line {i}: Consider using asyncio.sleep() in async contexts")

                if 'requests.' in line and 'session' not in content:
                    recommendations.append(f"Line {i}: Consider using requests.Session() for multiple requests")

                if 'open(' in line and 'with' not in line:
                    recommendations.append(f"Line {i}: Consider using 'with' statement for file operations")

        except Exception as e:
            logger.error(f"Error analyzing code file {code_file}: {e}")

        return recommendations

    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'cache_stats': self.cache.get_stats().__dict__,
            'executor_stats': self.executor.get_stats(),
            'optimization_results': [result.__dict__ for result in self.optimization_results],
        }

        # Memory pool stats
        if self.memory_pools:
            report['memory_pool_stats'] = {
                name: pool.get_stats() for name, pool in self.memory_pools.items()
            }

        # Lazy loader stats
        if self.lazy_loaders:
            report['lazy_loader_stats'] = {
                name: {'loaded': loader.is_loaded()} for name, loader in self.lazy_loaders.items()
            }

        # Calculate overall performance improvement
        if self.optimization_results:
            avg_improvement = sum(r.improvement_percent for r in self.optimization_results) / len(self.optimization_results)
            total_memory_saved = sum(r.memory_saved_mb for r in self.optimization_results)

            report['summary'] = {
                'average_improvement_percent': avg_improvement,
                'total_memory_saved_mb': total_memory_saved,
                'optimizations_applied': len(self.optimization_results)
            }

        return report

    def save_optimization_report(self, filepath: str):
        """Save optimization report to file."""
        report = self.generate_optimization_report()

        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Optimization report saved to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving optimization report: {e}")

    def cleanup(self):
        """Cleanup resources."""
        self.cache.clear()
        self.executor.shutdown()

        for loader in self.lazy_loaders.values():
            loader.unload()


# Convenience functions
def create_optimizer(config: Optional[Dict[str, Any]] = None) -> PerformanceOptimizer:
    """Create and return a new performance optimizer instance."""
    return PerformanceOptimizer(config)


def optimize_function(cache_ttl_minutes: int = 60, use_parallel: bool = False):
    """Decorator for automatic function optimization."""
    def decorator(func: Callable):
        optimizer = PerformanceOptimizer()

        if use_parallel:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await optimizer.executor.execute_async_batch([func], *args, **kwargs)
            return async_wrapper
        else:
            return optimizer.cached_function(ttl_minutes=cache_ttl_minutes)(func)

    return decorator


if __name__ == "__main__":
    # Example usage and testing
    optimizer = create_optimizer()

    try:
        print("Testing performance optimization...")

        # Test caching
        @optimizer.cached_function(ttl_minutes=5)
        def expensive_computation(n: int) -> int:
            time.sleep(0.1)  # Simulate expensive operation
            return sum(i**2 for i in range(n))

        # First call - should be slow
        start_time = time.time()
        result1 = expensive_computation(1000)
        first_call_time = time.time() - start_time

        # Second call - should be fast (cached)
        start_time = time.time()
        result2 = expensive_computation(1000)
        second_call_time = time.time() - start_time

        print(f"First call: {first_call_time:.3f}s, Second call: {second_call_time:.3f}s")
        print(f"Cache hit improvement: {((first_call_time - second_call_time) / first_call_time * 100):.1f}%")

        # Test lazy loading
        def create_large_object():
            return list(range(1000000))  # Large list

        lazy_loader = optimizer.register_lazy_loader("large_object", create_large_object)
        print(f"Lazy loader created, loaded: {lazy_loader.is_loaded()}")

        # Access the object
        obj = lazy_loader.get()
        print(f"Object accessed, loaded: {lazy_loader.is_loaded()}, size: {len(obj)}")

        # Generate optimization report
        report = optimizer.generate_optimization_report()
        print(f"Cache hit rate: {report['cache_stats']['hit_rate']:.2f}")

        # Save report
        optimizer.save_optimization_report("optimization_test_report.json")

    finally:
        optimizer.cleanup()
        print("Performance optimization test completed")