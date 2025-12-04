"""
Performance monitoring and optimization module for the Email Agent system.
Tracks system performance, identifies bottlenecks, and provides optimization recommendations.
"""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import logging
from collections import deque, defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    timestamp: str
    operation: str
    duration_ms: float
    memory_usage_mb: float
    cpu_percent: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class SystemHealth:
    """System health snapshot."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    active_threads: int
    queue_size: int
    response_time_avg: float
    error_rate: float


class PerformanceTracker:
    """Tracks and analyzes system performance metrics."""
    
    def __init__(self, max_metrics: int = 1000, logger: Optional[logging.Logger] = None):
        self.max_metrics = max_metrics
        self.logger = logger or logging.getLogger(__name__)
        self.metrics = deque(maxlen=max_metrics)
        self.operation_stats = defaultdict(list)
        self.lock = threading.Lock()
        self.start_time = time.time()
        
    def record_metric(self, operation: str, duration_ms: float, success: bool = True, error_message: str = None):
        """Record a performance metric."""
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            operation=operation,
            duration_ms=duration_ms,
            memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            cpu_percent=psutil.cpu_percent(),
            success=success,
            error_message=error_message
        )
        
        with self.lock:
            self.metrics.append(metric)
            self.operation_stats[operation].append(metric)
            
            # Keep only recent metrics per operation
            if len(self.operation_stats[operation]) > 100:
                self.operation_stats[operation] = self.operation_stats[operation][-100:]
    
    def get_operation_stats(self, operation: str, hours: int = 1) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_metrics = [
                m for m in self.operation_stats[operation]
                if datetime.fromisoformat(m.timestamp) > cutoff_time
            ]
        
        if not recent_metrics:
            return {'operation': operation, 'no_data': True}
        
        durations = [m.duration_ms for m in recent_metrics]
        success_count = sum(1 for m in recent_metrics if m.success)
        
        return {
            'operation': operation,
            'count': len(recent_metrics),
            'success_rate': success_count / len(recent_metrics),
            'avg_duration_ms': sum(durations) / len(durations),
            'min_duration_ms': min(durations),
            'max_duration_ms': max(durations),
            'p95_duration_ms': sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
            'avg_memory_mb': sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics),
            'error_rate': (len(recent_metrics) - success_count) / len(recent_metrics)
        }
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health snapshot."""
        return SystemHealth(
            timestamp=datetime.now().isoformat(),
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=psutil.virtual_memory().percent,
            disk_usage_percent=psutil.disk_usage('/').percent,
            active_threads=threading.active_count(),
            queue_size=len(self.metrics),
            response_time_avg=self._calculate_avg_response_time(),
            error_rate=self._calculate_error_rate()
        )
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time for recent operations."""
        recent_time = datetime.now() - timedelta(minutes=5)
        
        with self.lock:
            recent_metrics = [
                m for m in self.metrics
                if datetime.fromisoformat(m.timestamp) > recent_time
            ]
        
        if not recent_metrics:
            return 0.0
        
        return sum(m.duration_ms for m in recent_metrics) / len(recent_metrics)
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate for recent operations."""
        recent_time = datetime.now() - timedelta(minutes=5)
        
        with self.lock:
            recent_metrics = [
                m for m in self.metrics
                if datetime.fromisoformat(m.timestamp) > recent_time
            ]
        
        if not recent_metrics:
            return 0.0
        
        error_count = sum(1 for m in recent_metrics if not m.success)
        return error_count / len(recent_metrics)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        health = self.get_system_health()
        
        # Get stats for all operations
        operation_stats = {}
        with self.lock:
            for operation in self.operation_stats.keys():
                operation_stats[operation] = self.get_operation_stats(operation)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(operation_stats)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(health, operation_stats, bottlenecks)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': time.time() - self.start_time,
            'system_health': asdict(health),
            'operation_stats': operation_stats,
            'bottlenecks': bottlenecks,
            'recommendations': recommendations
        }
    
    def _identify_bottlenecks(self, operation_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        for operation, stats in operation_stats.items():
            if stats.get('no_data'):
                continue
            
            # High average duration
            if stats['avg_duration_ms'] > 5000:  # 5 seconds
                bottlenecks.append({
                    'type': 'slow_operation',
                    'operation': operation,
                    'avg_duration_ms': stats['avg_duration_ms'],
                    'severity': 'high' if stats['avg_duration_ms'] > 10000 else 'medium'
                })
            
            # High error rate
            if stats['error_rate'] > 0.1:  # 10%
                bottlenecks.append({
                    'type': 'high_error_rate',
                    'operation': operation,
                    'error_rate': stats['error_rate'],
                    'severity': 'high' if stats['error_rate'] > 0.2 else 'medium'
                })
            
            # High P95 latency
            if stats['p95_duration_ms'] > 10000:  # 10 seconds
                bottlenecks.append({
                    'type': 'high_p95_latency',
                    'operation': operation,
                    'p95_duration_ms': stats['p95_duration_ms'],
                    'severity': 'medium'
                })
        
        return bottlenecks
    
    def _generate_recommendations(self, health: SystemHealth, operation_stats: Dict[str, Any], bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # System resource recommendations
        if health.cpu_percent > 80:
            recommendations.append("High CPU usage detected. Consider scaling up or optimizing CPU-intensive operations.")
        
        if health.memory_percent > 85:
            recommendations.append("High memory usage detected. Consider implementing memory optimization or scaling up.")
        
        if health.disk_usage_percent > 90:
            recommendations.append("High disk usage detected. Consider cleaning up old files or expanding storage.")
        
        # Operation-specific recommendations
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'slow_operation':
                recommendations.append(f"Operation '{bottleneck['operation']}' is slow (avg: {bottleneck['avg_duration_ms']:.0f}ms). Consider optimization or caching.")
            
            elif bottleneck['type'] == 'high_error_rate':
                recommendations.append(f"Operation '{bottleneck['operation']}' has high error rate ({bottleneck['error_rate']:.1%}). Review error handling and retry logic.")
            
            elif bottleneck['type'] == 'high_p95_latency':
                recommendations.append(f"Operation '{bottleneck['operation']}' has high P95 latency ({bottleneck['p95_duration_ms']:.0f}ms). Consider async processing.")
        
        # General recommendations
        if health.active_threads > 50:
            recommendations.append("High thread count detected. Consider using async/await patterns or connection pooling.")
        
        if health.response_time_avg > 3000:
            recommendations.append("High average response time. Consider implementing caching or optimizing database queries.")
        
        return recommendations


class PerformanceOptimizer:
    """Automatic performance optimization system."""
    
    def __init__(self, tracker: PerformanceTracker, logger: Optional[logging.Logger] = None):
        self.tracker = tracker
        self.logger = logger or logging.getLogger(__name__)
        self.optimization_history = []
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def optimize_operation(self, operation: str) -> Dict[str, Any]:
        """Apply optimizations for a specific operation."""
        stats = self.tracker.get_operation_stats(operation)
        
        if stats.get('no_data'):
            return {'operation': operation, 'status': 'no_data'}
        
        optimizations_applied = []
        
        # Apply caching for slow operations
        if stats['avg_duration_ms'] > 2000:
            optimizations_applied.append('caching_recommended')
        
        # Apply async processing for high-latency operations
        if stats['p95_duration_ms'] > 5000:
            optimizations_applied.append('async_processing_recommended')
        
        # Apply retry logic for high-error operations
        if stats['error_rate'] > 0.05:
            optimizations_applied.append('retry_logic_recommended')
        
        optimization_result = {
            'operation': operation,
            'timestamp': datetime.now().isoformat(),
            'optimizations_applied': optimizations_applied,
            'before_stats': stats
        }
        
        self.optimization_history.append(optimization_result)
        return optimization_result
    
    def auto_optimize(self) -> List[Dict[str, Any]]:
        """Automatically optimize all operations based on performance data."""
        results = []
        
        # Get all operations with performance issues
        report = self.tracker.get_performance_report()
        
        for bottleneck in report['bottlenecks']:
            if bottleneck['severity'] == 'high':
                result = self.optimize_operation(bottleneck['operation'])
                results.append(result)
        
        return results


def performance_monitor(operation: str):
    """Decorator to monitor performance of functions."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error_message = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                
                # Record metric if tracker is available
                if hasattr(wrapper, '_tracker'):
                    wrapper._tracker.record_metric(
                        operation=operation,
                        duration_ms=duration_ms,
                        success=success,
                        error_message=error_message
                    )
        
        return wrapper
    return decorator


class AsyncPerformanceTracker:
    """Async version of performance tracker for high-throughput scenarios."""
    
    def __init__(self, max_metrics: int = 10000, logger: Optional[logging.Logger] = None):
        self.max_metrics = max_metrics
        self.logger = logger or logging.getLogger(__name__)
        self.metrics_queue = asyncio.Queue(maxsize=max_metrics)
        self.metrics_store = deque(maxlen=max_metrics)
        self.processing_task = None
        self.lock = asyncio.Lock()
    
    async def start(self):
        """Start the async performance tracker."""
        self.processing_task = asyncio.create_task(self._process_metrics())
    
    async def stop(self):
        """Stop the async performance tracker."""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
    
    async def record_metric_async(self, operation: str, duration_ms: float, success: bool = True):
        """Record a performance metric asynchronously."""
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            operation=operation,
            duration_ms=duration_ms,
            memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            cpu_percent=psutil.cpu_percent(),
            success=success
        )
        
        try:
            await self.metrics_queue.put(metric)
        except asyncio.QueueFull:
            self.logger.warning("Metrics queue is full, dropping metric")
    
    async def _process_metrics(self):
        """Process metrics from the queue."""
        while True:
            try:
                metric = await self.metrics_queue.get()
                async with self.lock:
                    self.metrics_store.append(metric)
                self.metrics_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing metric: {e}")
    
    async def get_metrics_async(self, operation: str = None, hours: int = 1) -> List[PerformanceMetric]:
        """Get metrics asynchronously."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        async with self.lock:
            filtered_metrics = [
                m for m in self.metrics_store
                if datetime.fromisoformat(m.timestamp) > cutoff_time
                and (operation is None or m.operation == operation)
            ]
        
        return filtered_metrics


# Global performance tracker instance
performance_tracker = PerformanceTracker()


def setup_performance_monitoring():
    """Setup global performance monitoring."""
    # Attach tracker to the performance_monitor decorator
    performance_monitor._tracker = performance_tracker
    
    # Start background monitoring
    def monitor_system():
        while True:
            try:
                health = performance_tracker.get_system_health()
                
                # Log warnings for critical conditions
                if health.cpu_percent > 90:
                    performance_tracker.logger.warning(f"Critical CPU usage: {health.cpu_percent}%")
                
                if health.memory_percent > 95:
                    performance_tracker.logger.warning(f"Critical memory usage: {health.memory_percent}%")
                
                if health.error_rate > 0.2:
                    performance_tracker.logger.warning(f"High error rate: {health.error_rate:.1%}")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                performance_tracker.logger.error(f"Error in system monitoring: {e}")
                time.sleep(60)
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_system, daemon=True)
    monitor_thread.start()
    
    return performance_tracker


# Cache implementation for performance optimization
class PerformanceCache:
    """Simple in-memory cache for performance optimization."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if time.time() - self.access_times[key] > self.ttl_seconds:
                del self.cache[key]
                del self.access_times[key]
                return None
            
            self.access_times[key] = time.time()
            return self.cache[key]
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        with self.lock:
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)


# Global cache instance
performance_cache = PerformanceCache()


def cached(ttl_seconds: int = 3600):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = performance_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            performance_cache.set(cache_key, result)
            
            return result
        
        return wrapper
    return decorator
