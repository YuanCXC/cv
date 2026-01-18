"""
================================================================================
消融实验模块
================================================================================

本模块用于执行消融实验，对比不同配置下的系统性能。
主要实验设置：
1. Baseline: 无闭环 (One-shot)
2. Loop w/o CLIP: 闭环但无CLIP验证 (Full Trust SAM)
3. Full Loop: 完整闭环系统
================================================================================
"""

import os
import json
import time
from tqdm import tqdm
from typing import List, Dict, Any
import sys
import os

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config
from core.utils import load_metadata, save_results
from core.main_loop import VQAClosedLoopSystem
from evaluation import compute_vqa_accuracy
from models.model_qwen import QwenVLModel

class AblationRunner:
    def __init__(self):
        self.metadata = load_metadata(config.IMAGES_DIR)
        self.system = None # Lazy init
        self.baseline_model = None

    def _init_system(self):
        if self.system is None:
            self.system = VQAClosedLoopSystem()

    def _init_baseline(self):
        if self.baseline_model is None:
            self.baseline_model = QwenVLModel()

    def run_experiment(self, exp_name: str, use_loop: bool, use_clip: bool, limit: int = None):
        """
        运行单组消融实验
        """
        print(f"\n开始消融实验: {exp_name} (Loop={use_loop}, CLIP={use_clip})")
        results = []
        metrics = {"total": 0, "correct": 0.0}
        
        samples = self.metadata
        if limit:
            samples = samples[:limit]
            
        # 根据配置初始化
        if use_loop:
            self._init_system()
        else:
            self._init_baseline()
            
        for item in tqdm(samples, desc=f"Exp: {exp_name}"):
            image_path = os.path.join(config.IMAGES_DIR, item["image_file"])
            question = item["question"]
            
            try:
                if use_loop:
                    # 运行闭环系统
                    sys_result = self.system.solve(image_path, question, use_clip=use_clip)
                    prediction = sys_result["answer"]
                    # 提取额外信息
                    bbox = sys_result.get("bbox")
                    clip_score = sys_result.get("clip_score")
                    stats = sys_result.get("stats")
                else:
                    # 运行Baseline
                    resp = self.baseline_model.chat([image_path], f"Answer the question: {question}\nPlease answer briefly.")
                    prediction = resp["text"]
                    bbox = None
                    clip_score = 0.0
                    stats = {}
                    
                # 计算指标
                acc = compute_vqa_accuracy(prediction, item["answers"])
                metrics["total"] += 1
                metrics["correct"] += acc
                
                # Debug print for first few samples
                if metrics["total"] <= 3:
                     print(f"Sample {item['id']}: Pred='{prediction}', GT={item['answers'][:1]}..., Acc={acc}")
                
                results.append({
                    "id": item["id"],
                    "question": question,
                    "gt_answers": item["answers"],
                    "prediction": prediction,
                    "accuracy": acc,
                    "exp_config": exp_name,
                    "bbox": bbox,
                    "clip_score": clip_score,
                    "stats": stats,
                    "vqa_acc": acc # error_analysis uses this key
                })
                
            except Exception as e:
                print(f"Error in {exp_name} sample {item['id']}: {e}")
                
        # 保存结果
        summary = {
            "experiment": exp_name,
            "use_loop": use_loop,
            "use_clip": use_clip,
            "samples": metrics["total"],
            "accuracy": metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
        }
        
        output_dir = os.path.join(config.RESULTS_DIR, "ablation")
        os.makedirs(output_dir, exist_ok=True)
        save_results(results, os.path.join(output_dir, f"{exp_name}_predictions.json"))
        save_results([summary], os.path.join(output_dir, f"{exp_name}_metrics.json"))
        
        print(f"[{exp_name}] Accuracy: {summary['accuracy']:.2%}")

    def run_all(self, limit: int = None):
        """运行所有消融实验"""
        # 1. Baseline
        self.run_experiment("baseline", use_loop=False, use_clip=False, limit=limit)
        
        # 2. Loop without CLIP
        self.run_experiment("loop_no_clip", use_loop=True, use_clip=False, limit=limit)
        
        # 3. Full Loop
        self.run_experiment("full_loop", use_loop=True, use_clip=True, limit=limit)

if __name__ == "__main__":
    runner = AblationRunner()
    # runner.run_all(limit=5)
    runner.run_all(limit=100)
