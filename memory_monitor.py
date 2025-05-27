# -*- coding: utf-8 -*-
"""
memory_monitor.py - 記憶體監控模組
監控GPU記憶體使用量，超過限制時自動暫停Code
"""

import torch
import psutil
import threading
import time
import os
import signal
import gc
import warnings

class MemoryMonitor:
    
    def __init__(self, gpu_limit_gb=20, cpu_limit_gb=32, check_interval=5):
        """        
        Args:
            gpu_limit_gb (int): GPU記憶體限制（GB）
            cpu_limit_gb (int): CPU記憶體限制（GB）
            check_interval (int): 檢查間隔（秒）
        """
        self.gpu_limit_gb = gpu_limit_gb
        self.cpu_limit_gb = cpu_limit_gb
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread = None
        self.emergency_cleanup_triggered = False
        
        # 檢查CUDA可用性
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.gpu_count = torch.cuda.device_count()
            print(f"🖥️  檢測到 {self.gpu_count} 個GPU，記憶體限制: {gpu_limit_gb}GB")
        else:
            print("⚠️  未檢測到CUDA，僅監控CPU記憶體")
    
    def get_gpu_memory_usage(self):
        if not self.cuda_available:
            return {}
        
        gpu_memory = {}
        for i in range(self.gpu_count):
            try:
                # 獲取GPU記憶體信息
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                
                gpu_memory[f"GPU_{i}"] = {
                    "allocated": memory_allocated,
                    "reserved": memory_reserved,
                    "total": memory_total,
                    "usage_percent": (memory_reserved / memory_total) * 100
                }
            except Exception as e:
                print(f"⚠️  獲取GPU {i} 記憶體信息失敗: {e}")
        
        return gpu_memory
    
    def get_cpu_memory_usage(self):
        try:
            # 獲取系統記憶體信息
            memory = psutil.virtual_memory()
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info().rss / 1024**3  # GB
            
            return {
                "total": memory.total / 1024**3,
                "available": memory.available / 1024**3,
                "used": memory.used / 1024**3,
                "percent": memory.percent,
                "process_usage": process_memory
            }
        except Exception as e:
            print(f"⚠️  獲取CPU記憶體信息失敗: {e}")
            return {}
    
    def emergency_cleanup(self):
        if self.emergency_cleanup_triggered:
            return
        
        self.emergency_cleanup_triggered = True
        print("🚨 開始緊急記憶體清理...")
        
        try:
            # 清理GPU記憶體
            if self.cuda_available:
                print("🧹 清理GPU記憶體...")
                for i in range(self.gpu_count):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            # 強制垃圾收集
            print("🧹 執行垃圾收集...")
            gc.collect()
            
            # 等待一段時間讓清理生效
            time.sleep(2)
            
            # 再次檢查記憶體使用
            gpu_memory = self.get_gpu_memory_usage()
            for gpu_id, info in gpu_memory.items():
                if info["reserved"] > self.gpu_limit_gb:
                    print(f"⚠️  {gpu_id} 記憶體仍超過限制: {info['reserved']:.2f}GB")
                    return False
            
            print("✅ 緊急清理完成")
            self.emergency_cleanup_triggered = False
            return True
            
        except Exception as e:
            print(f"❌ 緊急清理失敗: {e}")
            return False
    
    def force_kill_program(self, reason):
        print(f"\n{'='*50}")
        print(f"🚨 記憶體使用超過限制！")
        print(f"📊 原因: {reason}")
        print(f"🔄 嘗試緊急清理...")
        
        # 嘗試緊急清理
        cleanup_success = self.emergency_cleanup()
        
        if not cleanup_success:
            print(f"❌ 緊急清理失敗，強制終止程序...")
            print(f"⏰ 程序將在3秒後終止...")
            
            for i in range(3, 0, -1):
                print(f"⏳ {i}...")
                time.sleep(1)
            
            print("💀 程序已終止")
            os.kill(os.getpid(), signal.SIGTERM)
        else:
            print("✅ 緊急清理成功，繼續運行")
    
    def check_memory_usage(self):
        if self.cuda_available:
            gpu_memory = self.get_gpu_memory_usage()
            for gpu_id, info in gpu_memory.items():
                if info["reserved"] > self.gpu_limit_gb:
                    reason = f"{gpu_id} 記憶體使用: {info['reserved']:.2f}GB > {self.gpu_limit_gb}GB"
                    self.force_kill_program(reason)
                    return False
        
        cpu_memory = self.get_cpu_memory_usage()
        if cpu_memory and cpu_memory.get("process_usage", 0) > self.cpu_limit_gb:
            reason = f"CPU記憶體使用: {cpu_memory['process_usage']:.2f}GB > {self.cpu_limit_gb}GB"
            self.force_kill_program(reason)
            return False
        
        return True
    
    def print_memory_status(self):
        print(f"\n📊 ==== 記憶體使用狀況 ====")
        
        if self.cuda_available:
            gpu_memory = self.get_gpu_memory_usage()
            for gpu_id, info in gpu_memory.items():
                status = "🔴" if info["reserved"] > self.gpu_limit_gb * 0.8 else "🟢"
                print(f"{status} {gpu_id}: {info['reserved']:.2f}GB / {info['total']:.2f}GB ({info['usage_percent']:.1f}%)")
        
        cpu_memory = self.get_cpu_memory_usage()
        if cpu_memory:
            status = "🔴" if cpu_memory["process_usage"] > self.cpu_limit_gb * 0.8 else "🟢"
            print(f"{status} CPU進程: {cpu_memory['process_usage']:.2f}GB")
            print(f"🖥️  系統記憶體: {cpu_memory['used']:.2f}GB / {cpu_memory['total']:.2f}GB ({cpu_memory['percent']:.1f}%)")
        
        print(f"{'='*30}")
    
    def monitor_loop(self):
        print(f"🔍 記憶體監控已啟動 (GPU限制: {self.gpu_limit_gb}GB, 檢查間隔: {self.check_interval}秒)")
        
        while self.monitoring:
            try:
                if not self.check_memory_usage():
                    break
                
                if int(time.time()) % 30 == 0:
                    self.print_memory_status()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                print(f"⚠️  監控循環出錯: {e}")
                time.sleep(self.check_interval)
    
    def start_monitoring(self):
        if self.monitoring:
            print("⚠️  監控已經在運行中")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("✅ 記憶體監控已啟動")
    
    def stop_monitoring(self):
        if not self.monitoring:
            print("⚠️  監控未在運行")
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("🛑 記憶體監控已停止")
    
    def get_current_status(self):
        status = {
            "gpu_memory": self.get_gpu_memory_usage(),
            "cpu_memory": self.get_cpu_memory_usage(),
            "limits": {
                "gpu_limit_gb": self.gpu_limit_gb,
                "cpu_limit_gb": self.cpu_limit_gb
            },
            "monitoring": self.monitoring
        }
        return status

memory_monitor = None

def get_memory_monitor(gpu_limit_gb=20, cpu_limit_gb=32, check_interval=5):
    global memory_monitor
    if memory_monitor is None:
        memory_monitor = MemoryMonitor(gpu_limit_gb, cpu_limit_gb, check_interval)
    return memory_monitor

def start_memory_monitoring(gpu_limit_gb=20, cpu_limit_gb=32, check_interval=5):
    monitor = get_memory_monitor(gpu_limit_gb, cpu_limit_gb, check_interval)
    monitor.start_monitoring()
    return monitor

def stop_memory_monitoring():
    global memory_monitor
    if memory_monitor:
        memory_monitor.stop_monitoring()

if __name__ == "__main__":
    print("測試記憶體監控器...")
    monitor = start_memory_monitoring(gpu_limit_gb=20, check_interval=2)
    
    try:
        time.sleep(10)
        
        status = monitor.get_current_status()
        print("當前狀態:", status)
        
    except KeyboardInterrupt:
        print("用戶中斷")
    finally:
        stop_memory_monitoring()