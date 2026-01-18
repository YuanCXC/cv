"""
================================================================================
效率分析模块
================================================================================

本模块用于分析系统的运行效率，包括：
1. 推理耗时统计
2. 显存占用监控
3. 工具调用次数统计
================================================================================
"""

import os
import json
import time
import torch
import numpy as np
from typing import Dict, Any, List
import sys
import os

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config
from core.main_loop import VQAClosedLoopSystem
from core.utils import load_metadata

class EfficiencyAnalyzer:
    def __init__(self):
        pass

    def analyze_results(self, results_path: str):
        """
        分析结果文件中的效率指标
        """
        print(f"分析效率指标: {results_path}")
        if not os.path.exists(results_path):
            print("结果文件不存在")
            return
            
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 兼容列表或字典格式（有些metrics存为list of dict）
        if isinstance(data, list):
            predictions = data
        else:
            print("格式不支持")
            return

        latencies = []
        sam_calls = []
        clip_calls = []
        qwen_calls = []
        
        for p in predictions:
            # 尝试获取stats
            stats = p.get("stats", {})
            if stats:
                sam_calls.append(stats.get("sam_calls", 0))
                clip_calls.append(stats.get("clip_calls", 0))
                qwen_calls.append(stats.get("qwen_calls", 0))
            
            # 尝试获取latency (baseline有，loop可能有如果加了)
            # 这里的Main Loop目前没显式存总latency到外层，只在QwenModel里有
            # 我们假设results里会有 'total_latency' 字段如果我们在main loop里加了的话
            # 暂时跳过Latency如果不存在
            pass

        summary = {
            "avg_sam_calls": np.mean(sam_calls) if sam_calls else 0,
            "avg_clip_calls": np.mean(clip_calls) if clip_calls else 0,
            "avg_qwen_calls": np.mean(qwen_calls) if qwen_calls else 0,
            "total_samples": len(predictions)
        }
        
        print("效率统计:")
        print(json.dumps(summary, indent=2))
        
        output_dir = os.path.join(config.RESULTS_DIR, "efficiency")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "analysis.json"), 'w') as f:
            json.dump(summary, f, indent=2)

    def benchmark_resource_usage(self, limit: int = 5):
        """
        实时基准测试（显存和时间）
        """
        print(f"开始资源基准测试 (Samples: {limit})...")
        system = VQAClosedLoopSystem()
        metadata = load_metadata(config.IMAGES_DIR)[:limit]
        
        latencies = []
        peak_memories = []
        
        # 预热
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        for item in metadata:
            image_path = os.path.join(config.IMAGES_DIR, item["image_file"])
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                
            start_t = time.time()
            _ = system.solve(image_path, item["question"])
            end_t = time.time()
            
            latencies.append(end_t - start_t)
            
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated() / 1024**2 # MB
                peak_memories.append(peak_mem)
            else:
                peak_memories.append(0)
                
        summary = {
            "avg_latency_sec": np.mean(latencies),
            "avg_peak_memory_mb": np.mean(peak_memories),
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        print("基准测试结果:")
        print(json.dumps(summary, indent=2))
        
        output_dir = os.path.join(config.RESULTS_DIR, "efficiency")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "benchmark.json"), 'w') as f:
            json.dump(summary, f, indent=2)

if __name__ == "__main__":
    analyzer = EfficiencyAnalyzer()
    # 分析全闭环实验的效率
    analyzer.analyze_results(os.path.join(config.RESULTS_DIR, "ablation", "full_loop_predictions.json"))
    # 运行基准测试 (使用20个样本以节省时间，或按需调整)
    analyzer.benchmark_resource_usage(limit=20)
