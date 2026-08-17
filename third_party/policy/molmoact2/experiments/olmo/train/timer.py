"""Utilities for timing and profiling training code."""

import time
import logging
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Optional
import torch
import torch.distributed as dist

log = logging.getLogger(__name__)


class Timer:
    """A timer for measuring elapsed time with optional synchronization for distributed training."""
    
    def __init__(self, name: str = "", synchronize: bool = False, device: Optional[torch.device] = None):
        self.name = name
        self.synchronize = synchronize
        self.device = device
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0
        
    def __enter__(self):
        if self.synchronize and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        if self.synchronize and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self.elapsed = time.perf_counter() - self.start_time
        self.start_time = None


class TimerManager:
    """
    Manager for multiple timers with automatic logging and statistics.
    
    Example usage:
        timer_mgr = TimerManager()
        
        with timer_mgr.timer("forward_pass"):
            model(batch)
            
        with timer_mgr.timer("backward_pass", synchronize=True):
            loss.backward()
            
        # Log accumulated statistics
        timer_mgr.log_stats()
        
        # Get metrics for wandb
        metrics = timer_mgr.get_metrics()
    """
    
    def __init__(self, synchronize: bool = False, device: Optional[torch.device] = None, reduce_across_ranks: bool = False, enabled: bool = True):
        self.synchronize = synchronize
        self.device = device
        self.reduce_across_ranks = reduce_across_ranks
        self.enabled = enabled
        self.timings: Dict[str, list] = defaultdict(list)
        self.current_timers: Dict[str, float] = {}
        
    @contextmanager
    def timer(self, name: str, synchronize: Optional[bool] = None):
        """Context manager for timing a code block."""
        if not self.enabled:
            # When disabled, just yield without any timing overhead
            yield
            return
            
        if synchronize is None:
            synchronize = self.synchronize
            
        if synchronize and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            
        start_time = time.perf_counter()
        try:
            yield
        finally:
            if synchronize and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - start_time
            self.timings[name].append(elapsed)
    
    def get_last(self, name: str) -> Optional[float]:
        """Get the most recent timing for a given timer name."""
        if not self.enabled:
            return None
        if name in self.timings and self.timings[name]:
            return self.timings[name][-1]
        return None
            
    def reset(self):
        """Reset all accumulated timings."""
        self.timings.clear()
        self.current_timers.clear()
        
    def get_stats(self, prefix: str = "timing/") -> Dict[str, float]:
        """
        Get statistics for all timers.
        
        Returns a dictionary with mean, min, max, and sum for each timer.
        """
        if not self.enabled:
            return {}
            
        stats = {}
        for name, times in self.timings.items():
            if not times:
                continue
                
            # Calculate statistics
            mean_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            total_time = sum(times)
            
            # Optionally reduce across ranks
            if self.reduce_across_ranks and dist.is_initialized():
                mean_tensor = torch.tensor(mean_time, device=self.device)
                dist.all_reduce(mean_tensor, op=dist.ReduceOp.AVG)
                mean_time = mean_tensor.item()
            
            stats[f"{prefix}{name}_mean"] = mean_time
            stats[f"{prefix}{name}_min"] = min_time
            stats[f"{prefix}{name}_max"] = max_time
            stats[f"{prefix}{name}_total"] = total_time
            stats[f"{prefix}{name}_count"] = len(times)
            
        return stats
    
    def get_metrics(self, prefix: str = "timing/", only_mean: bool = True) -> Dict[str, float]:
        """
        Get metrics suitable for logging to wandb.
        
        Args:
            prefix: Prefix for metric names
            only_mean: If True, only return mean times (cleaner logs)
        """
        if not self.enabled:
            return {}
            
        if only_mean:
            metrics = {}
            for name, times in self.timings.items():
                if times:
                    mean_time = sum(times) / len(times)
                    if self.reduce_across_ranks and dist.is_initialized():
                        mean_tensor = torch.tensor(mean_time, device=self.device)
                        dist.all_reduce(mean_tensor, op=dist.ReduceOp.AVG)
                        mean_time = mean_tensor.item()
                    metrics[f"{prefix}{name}"] = mean_time
            return metrics
        else:
            return self.get_stats(prefix)
    
    def log_stats(self, step: Optional[int] = None):
        """Log timing statistics to the console."""
        if not self.enabled or not self.timings:
            return
            
        stats = self.get_stats(prefix="")
        
        header = "Timing Statistics"
        if step is not None:
            header += f" (step {step})"
            
        log.info(header)
        
        # Group by timer name
        timer_names = sorted(set(k.rsplit("_", 1)[0] for k in stats.keys()))
        for name in timer_names:
            mean = stats.get(f"{name}_mean", 0)
            min_t = stats.get(f"{name}_min", 0)
            max_t = stats.get(f"{name}_max", 0)
            count = stats.get(f"{name}_count", 0)
            log.info(f"  {name}: mean={mean*1000:.2f}ms, min={min_t*1000:.2f}ms, max={max_t*1000:.2f}ms, count={count}")


class SectionTimer:
    """
    A simple section timer for quick timing of code blocks.
    
    Example:
        timer = SectionTimer("data_loading")
        timer.start()
        # ... code ...
        timer.stop()
        print(f"Data loading took {timer.elapsed:.2f}s")
    """
    
    def __init__(self, name: str = "", enabled: bool = True, synchronize: bool = False, device: Optional[torch.device] = None):
        self.name = name
        self.enabled = enabled
        self.synchronize = synchronize
        self.device = device
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0
        
    def start(self):
        """Start the timer."""
        if not self.enabled:
            return
        if self.synchronize and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self.start_time = time.perf_counter()
        
    def stop(self):
        """Stop the timer and return elapsed time."""
        if not self.enabled:
            return 0.0
        if self.start_time is None:
            log.warning(f"Timer '{self.name}' was not started")
            return 0.0
        if self.synchronize and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self.elapsed = time.perf_counter() - self.start_time
        self.start_time = None
        return self.elapsed
    
    def log(self, prefix: str = ""):
        """Log the elapsed time."""
        if self.enabled and self.elapsed > 0:
            name = f"{prefix}{self.name}" if prefix else self.name
            log.info(f"{name}: {self.elapsed:.4f}s ({self.elapsed*1000:.2f}ms)")
