#!/usr/bin/env python3
"""
Memory Monitoring Utility for Insurance QA API
Helps track memory usage during deployment and optimization
"""

import psutil
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Memory monitoring utility for optimization tracking"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.peak_memory = 0
        self.memory_samples = []
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        try:
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # Update peak memory
            if memory_mb > self.peak_memory:
                self.peak_memory = memory_mb
            
            # Store sample
            self.memory_samples.append({
                'timestamp': time.time(),
                'memory_mb': memory_mb,
                'peak_mb': self.peak_memory
            })
            
            return {
                'current_mb': round(memory_mb, 1),
                'peak_mb': round(self.peak_memory, 1),
                'available_mb': round(512 - memory_mb, 1),  # 512MB limit
                'usage_percent': round((memory_mb / 512) * 100, 1),
                'uptime_seconds': round(time.time() - self.start_time, 1)
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {
                'current_mb': 0,
                'peak_mb': 0,
                'available_mb': 512,
                'usage_percent': 0,
                'uptime_seconds': 0
            }
    
    def log_memory_status(self, stage: str = "current"):
        """Log current memory status"""
        stats = self.get_memory_usage()
        logger.info(f"💾 Memory Status ({stage}):")
        logger.info(f"   Current: {stats['current_mb']} MB")
        logger.info(f"   Peak: {stats['peak_mb']} MB")
        logger.info(f"   Available: {stats['available_mb']} MB")
        logger.info(f"   Usage: {stats['usage_percent']}%")
        logger.info(f"   Uptime: {stats['uptime_seconds']}s")
        
        # Warning if approaching limit
        if stats['usage_percent'] > 80:
            logger.warning(f"⚠️ High memory usage: {stats['usage_percent']}%")
        elif stats['usage_percent'] > 90:
            logger.error(f"🚨 Critical memory usage: {stats['usage_percent']}%")
        
        return stats
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary"""
        if not self.memory_samples:
            return {}
        
        memory_values = [s['memory_mb'] for s in self.memory_samples]
        
        return {
            'total_samples': len(self.memory_samples),
            'average_mb': round(sum(memory_values) / len(memory_values), 1),
            'peak_mb': round(self.peak_memory, 1),
            'min_mb': round(min(memory_values), 1),
            'max_mb': round(max(memory_values), 1),
            'uptime_seconds': round(time.time() - self.start_time, 1)
        }
    
    def print_summary(self):
        """Print memory usage summary"""
        summary = self.get_memory_summary()
        if not summary:
            logger.info("No memory data collected")
            return
        
        logger.info("📊 Memory Usage Summary:")
        logger.info(f"   Samples: {summary['total_samples']}")
        logger.info(f"   Average: {summary['average_mb']} MB")
        logger.info(f"   Peak: {summary['peak_mb']} MB")
        logger.info(f"   Range: {summary['min_mb']} - {summary['max_mb']} MB")
        logger.info(f"   Uptime: {summary['uptime_seconds']}s")

# Global memory monitor instance
memory_monitor = MemoryMonitor()

def log_memory_usage(stage: str):
    """Convenience function to log memory usage"""
    return memory_monitor.log_memory_status(stage)

if __name__ == "__main__":
    # Test memory monitoring
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Memory Monitor...")
    memory_monitor.log_memory_status("start")
    
    # Simulate some work
    import numpy as np
    test_array = np.random.rand(1000, 1000)
    memory_monitor.log_memory_status("after_numpy")
    
    del test_array
    import gc
    gc.collect()
    memory_monitor.log_memory_status("after_cleanup")
    
    memory_monitor.print_summary() 