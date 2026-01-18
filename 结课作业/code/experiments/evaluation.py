"""
================================================================================
基础评估模块 (Baseline)
================================================================================

本模块实现了基线(One-shot) VQA评估。
不使用闭环反馈，直接使用Qwen-VL进行推理。
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
from core.utils import load_metadata, normalize_answer, save_results
from models.model_qwen import QwenVLModel

def compute_vqa_accuracy(prediction: str, ground_truths: List[str]) -> float:
    """
    计算VQA准确率 (Standard VQA Metric)
    Acc = min(matching_count / 3, 1)
    """
    pred = normalize_answer(prediction)
    match_count = 0
    
    for gt in ground_truths:
        if normalize_answer(gt) == pred:
            match_count += 1
            
    return min(match_count / 3.0, 1.0)

def compute_exact_match(prediction: str, ground_truths: List[str]) -> bool:
    """
    计算精确匹配 (Exact Match)
    只要预测结果在GT列表中出现即为True
    """
    pred = normalize_answer(prediction)
    return any(normalize_answer(gt) == pred for gt in ground_truths)

class BaselineEvaluator:
    def __init__(self):
        self.model = QwenVLModel()
        self.data_dir = config.IMAGES_DIR
        self.metadata = load_metadata(config.IMAGES_DIR)
        
    def run(self, limit: int = None):
        """
        运行评估
        
        Args:
            limit (int): 测试样本数量限制 (用于调试)
        """
        print(f"开始基线评估 (One-shot)...")
        results = []
        metrics = {
            "total": 0,
            "correct_vqa": 0.0,
            "correct_em": 0,
            "total_latency": 0.0
        }
        
        samples = self.metadata
        if limit:
            samples = samples[:limit]
            
        for item in tqdm(samples, desc="Evaluating"):
            image_path = os.path.join(self.data_dir, item["image_file"])
            question = item["question"]
            gt_answers = item["answers"]
            
            # One-shot 推理
            start_time = time.time()
            response = self.model.chat([image_path], f"Answer the question briefly: {question}")
            latency = time.time() - start_time
            
            prediction = response["text"]
            
            # 计算指标
            vqa_acc = compute_vqa_accuracy(prediction, gt_answers)
            em = compute_exact_match(prediction, gt_answers)
            
            metrics["total"] += 1
            metrics["correct_vqa"] += vqa_acc
            metrics["correct_em"] += 1 if em else 0
            metrics["total_latency"] += latency
            
            results.append({
                "id": item["id"],
                "image": item["image_file"],
                "question": question,
                "gt_answers": gt_answers,
                "prediction": prediction,
                "vqa_acc": vqa_acc,
                "em": em,
                "latency": latency
            })
            
        # 汇总结果
        summary = {
            "model": config.DISPLAY_MODEL_NAME,
            "mode": "baseline_oneshot",
            "samples": metrics["total"],
            "accuracy_vqa": metrics["correct_vqa"] / metrics["total"] if metrics["total"] > 0 else 0,
            "accuracy_em": metrics["correct_em"] / metrics["total"] if metrics["total"] > 0 else 0,
            "avg_latency": metrics["total_latency"] / metrics["total"] if metrics["total"] > 0 else 0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print("\n评估完成:")
        print(f"VQA Accuracy: {summary['accuracy_vqa']:.2%}")
        print(f"Exact Match:  {summary['accuracy_em']:.2%}")
        
        # 保存结果
        output_dir = os.path.join(config.RESULTS_DIR, "baseline")
        os.makedirs(output_dir, exist_ok=True)
        
        save_results(results, os.path.join(output_dir, "predictions.json"))
        save_results([summary], os.path.join(output_dir, "metrics.json"))

if __name__ == "__main__":
    evaluator = BaselineEvaluator()
    # 默认跑所有数据，如果想快速测试可以 evaluator.run(limit=5)
    evaluator.run(limit=100)
