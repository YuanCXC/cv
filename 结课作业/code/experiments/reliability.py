"""
================================================================================
可靠性评测模块
================================================================================

本模块负责评估系统的鲁棒性和可靠性。
包括：
1. OOD (Out-of-Distribution) 泛化测试（图像扰动）。
2. 长尾/小目标分析（基于检测到的目标面积）。
================================================================================
"""

import os
import json
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
from typing import List, Dict, Any
import sys
import os

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config
from core.utils import load_metadata, save_results, compute_bbox_area
from experiments.evaluation import compute_vqa_accuracy
from core.main_loop import VQAClosedLoopSystem

def perturb_image(image: Image.Image, perturbation_type: str = "random") -> Image.Image:
    """
    对图像应用扰动
    """
    img = image.copy()
    
    if perturbation_type == "random":
        types = ["brightness", "contrast", "blur", "noise"]
        perturbation_type = random.choice(types)
        
    if perturbation_type == "brightness":
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.5, 1.5))
    elif perturbation_type == "contrast":
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.5, 1.5))
    elif perturbation_type == "blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 3)))
    elif perturbation_type == "noise":
        # 添加高斯噪声
        img_np = np.array(img)
        noise = np.random.normal(0, 25, img_np.shape).astype(np.uint8)
        img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        
    return img

class ReliabilityEvaluator:
    def __init__(self):
        self.system = VQAClosedLoopSystem()
        self.metadata = load_metadata(config.IMAGES_DIR)
        
    def run_ood_test(self, limit: int = None):
        """
        运行OOD泛化测试
        """
        print("开始OOD泛化测试 (使用图像扰动)...")
        results = []
        metrics = {"total": 0, "correct": 0.0}
        
        samples = self.metadata
        if limit:
            samples = samples[:limit]
            
        for item in tqdm(samples, desc="OOD Testing"):
            original_image_path = os.path.join(config.IMAGES_DIR, item["image_file"])
            
            # 加载并扰动
            try:
                img = Image.open(original_image_path).convert("RGB")
                perturbed_img = perturb_image(img)
                
                # 保存临时扰动图
                temp_path = os.path.join(config.RESULTS_DIR, "temp_ood.jpg")
                perturbed_img.save(temp_path)
                
                # 运行闭环系统
                sys_result = self.system.solve(temp_path, item["question"])
                
                # 计算指标
                acc = compute_vqa_accuracy(sys_result["answer"], item["answers"])
                metrics["total"] += 1
                metrics["correct"] += acc
                
                results.append({
                    "id": item["id"],
                    "question": item["question"],
                    "perturbation": "random",
                    "prediction": sys_result["answer"],
                    "accuracy": acc
                })
                
            except Exception as e:
                print(f"OOD Error {item['id']}: {e}")
                
        summary = {
            "mode": "ood_robustness",
            "samples": metrics["total"],
            "accuracy": metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
        }
        
        output_dir = os.path.join(config.RESULTS_DIR, "reliability")
        save_results(results, os.path.join(output_dir, "ood_predictions.json"))
        save_results([summary], os.path.join(output_dir, "ood_metrics.json"))
        
        print(f"OOD Accuracy: {summary['accuracy']:.2%}")

    def analyze_size_robustness(self, predictions_path: str):
        """
        分析目标大小对准确率的影响
        注意：依赖于predictions.json中包含bbox信息
        """
        print("开始分析目标大小与性能关系...")
        if not os.path.exists(predictions_path):
            print(f"文件不存在: {predictions_path}")
            return
            
        with open(predictions_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
            
        buckets = {
            "small": {"total": 0, "correct": 0.0},   # < 32^2
            "medium": {"total": 0, "correct": 0.0},  # 32^2 - 96^2
            "large": {"total": 0, "correct": 0.0},   # > 96^2
            "unknown": {"total": 0, "correct": 0.0}
        }
        
        for p in predictions:
            bbox = p.get("bbox")
            acc = p.get("vqa_acc", 0.0) # 假设prediction中有acc字段
            # 如果没有预先计算acc，这里可能需要重新计算，但通常prediction file里会存
            
            if not bbox:
                buckets["unknown"]["total"] += 1
                buckets["unknown"]["correct"] += acc
                continue
                
            area = compute_bbox_area(bbox)
            root_area = area ** 0.5
            
            if root_area < 32:
                key = "small"
            elif root_area < 96:
                key = "medium"
            else:
                key = "large"
                
            buckets[key]["total"] += 1
            buckets[key]["correct"] += acc
            
        # 计算各桶准确率
        analysis = {}
        for k, v in buckets.items():
            if v["total"] > 0:
                analysis[k] = v["correct"] / v["total"]
            else:
                analysis[k] = 0.0
                
        output_dir = os.path.join(config.RESULTS_DIR, "reliability")
        save_results([analysis], os.path.join(output_dir, "size_analysis.json"))
        print("大小分析结果:", analysis)

if __name__ == "__main__":
    evaluator = ReliabilityEvaluator()
    evaluator.run_ood_test(limit=100)
