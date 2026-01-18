"""
================================================================================
错误分析模块
================================================================================

本模块用于分析实验结果中的失败案例，并进行归因分类。
分类标准：
1. 定位失败 (Location Failure): 未检测到有效边界框。
2. 验证拒绝 (Verification Rejection): 检测到区域但CLIP分数过低。
3. 推理错误 (Reasoning Failure): 证据有效但最终答案错误。
================================================================================
"""

import os
import json
import shutil
from typing import List, Dict, Any
import sys
import os

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config
from core.utils import save_results

class ErrorAnalyzer:
    def __init__(self):
        pass

    def analyze(self, predictions_path: str, output_dir: str = None):
        """
        分析预测结果文件
        """
        print(f"分析错误案例: {predictions_path}")
        if not os.path.exists(predictions_path):
            print("文件不存在")
            return
            
        with open(predictions_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
            
        if not output_dir:
            output_dir = os.path.join(config.RESULTS_DIR, "error_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        categories = {
            "location_failure": [],
            "verification_rejection": [],
            "reasoning_failure": [],
            "correct": []
        }
        
        for p in predictions:
            # 获取必要字段
            bbox = p.get("bbox")
            clip_score = p.get("clip_score", 0.0)
            acc = p.get("vqa_acc", 0.0)
            
            # 判断逻辑
            if acc == 1.0:
                categories["correct"].append(p)
            elif bbox is None:
                categories["location_failure"].append(p)
            elif clip_score < config.CLIP_SIMILARITY_THRESHOLD and clip_score > 0:
                # 假设0分可能是没跑CLIP（例如bbox none时），这里排除bbox none的情况
                categories["verification_rejection"].append(p)
            else:
                # BBox存在，CLIP通过（或没跑但也没被拒绝），但答案错
                categories["reasoning_failure"].append(p)
                
        # 统计
        stats = {k: len(v) for k, v in categories.items()}
        total = sum(stats.values())
        stats_percent = {k: v/total if total > 0 else 0 for k, v in stats.items()}
        
        print("错误归因统计:")
        print(json.dumps(stats_percent, indent=2))
        
        # 保存分类结果
        for k, v in categories.items():
            if v:
                save_results(v, os.path.join(output_dir, f"{k}.json"))
                
        save_results([stats_percent], os.path.join(output_dir, "error_stats.json"))

        # 提取典型失败案例 (每个类别取前3个)
        self._extract_examples(categories, output_dir)

    def _extract_examples(self, categories: Dict[str, List], output_dir: str):
        """提取并保存典型案例的图片（可选，这里只生成JSON描述）"""
        examples = {}
        for k, v in categories.items():
            if k == "correct": continue
            examples[k] = v[:3]
            
        save_results([examples], os.path.join(output_dir, "typical_errors.json"))

if __name__ == "__main__":
    analyzer = ErrorAnalyzer()
    # 分析全闭环实验的错误案例
    target_file = os.path.join(config.RESULTS_DIR, "ablation", "full_loop_predictions.json")
    if os.path.exists(target_file):
        analyzer.analyze(target_file)
    else:
        print(f"未找到结果文件: {target_file}，请先运行 ablation.py")
