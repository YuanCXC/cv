"""
================================================================================
实验报告生成模块
================================================================================

本模块自动汇总所有实验结果，生成 Markdown 格式的实验报告。
================================================================================
"""

import os
import json
import sys

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config

class ReportGenerator:
    def __init__(self):
        self.results_dir = config.RESULTS_DIR
        self.report_path = os.path.join(self.results_dir, "experiment_report.md")

    def load_json(self, path):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data[0] if isinstance(data, list) else data
        return {}

    def generate(self):
        print("正在生成实验报告...")
        
        # 1. 加载数据
        baseline_metrics = self.load_json(os.path.join(self.results_dir, "baseline", "metrics.json"))
        
        ablation_baseline = self.load_json(os.path.join(self.results_dir, "ablation", "baseline_metrics.json"))
        ablation_no_clip = self.load_json(os.path.join(self.results_dir, "ablation", "loop_no_clip_metrics.json"))
        ablation_full = self.load_json(os.path.join(self.results_dir, "ablation", "full_loop_metrics.json"))
        
        reliability_ood = self.load_json(os.path.join(self.results_dir, "reliability", "ood_metrics.json"))
        reliability_size = self.load_json(os.path.join(self.results_dir, "reliability", "size_analysis.json"))
        
        efficiency = self.load_json(os.path.join(self.results_dir, "efficiency", "analysis.json"))
        efficiency_bench = self.load_json(os.path.join(self.results_dir, "efficiency", "benchmark.json"))
        
        error_stats = self.load_json(os.path.join(self.results_dir, "error_analysis", "error_stats.json"))

        # 2. 构建报告内容
        content = []
        content.append("# 闭环式多模态视觉系统实验报告")
        content.append(f"生成时间: {json.dumps(baseline_metrics.get('timestamp', 'Unknown'))}\n")
        
        content.append("## 1. 实验摘要")
        content.append("本项目实现并评估了一个基于 Qwen3-VL、SAM2 和 CLIP 的闭环视觉问答系统。")
        content.append(f"基础模型准确率: {baseline_metrics.get('accuracy_vqa', 0):.2%}")
        content.append(f"完整闭环准确率: {ablation_full.get('accuracy', 0):.2%}\n")
        
        content.append("## 2. 消融实验结果")
        content.append("| 实验配置 | 准确率 | 样本数 |")
        content.append("| :--- | :--- | :--- |")
        content.append(f"| Baseline (One-shot) | {ablation_baseline.get('accuracy', 0):.2%} | {ablation_baseline.get('samples', 0)} |")
        content.append(f"| Loop w/o CLIP | {ablation_no_clip.get('accuracy', 0):.2%} | {ablation_no_clip.get('samples', 0)} |")
        content.append(f"| Full Loop | {ablation_full.get('accuracy', 0):.2%} | {ablation_full.get('samples', 0)} |\n")
        
        content.append("## 3. 可靠性评测")
        content.append("### 3.1 OOD 泛化测试")
        content.append(f"- 图像扰动下准确率: {reliability_ood.get('accuracy', 0):.2%}\n")
        
        content.append("### 3.2 大小分布分析")
        content.append("| 目标大小 | 准确率 |")
        content.append("| :--- | :--- |")
        content.append(f"| Small (<32px) | {reliability_size.get('small', 0):.2%} |")
        content.append(f"| Medium (32-96px) | {reliability_size.get('medium', 0):.2%} |")
        content.append(f"| Large (>96px) | {reliability_size.get('large', 0):.2%} |\n")
        
        content.append("## 4. 效率分析")
        content.append(f"- 平均 SAM 调用: {efficiency.get('avg_sam_calls', 0):.2f}")
        content.append(f"- 平均 CLIP 调用: {efficiency.get('avg_clip_calls', 0):.2f}")
        content.append(f"- 平均 Qwen 调用: {efficiency.get('avg_qwen_calls', 0):.2f}")
        if efficiency_bench:
            content.append(f"- 单样本平均耗时: {efficiency_bench.get('avg_latency_sec', 0):.2f} s")
            content.append(f"- 峰值显存占用: {efficiency_bench.get('avg_peak_memory_mb', 0):.2f} MB")
        
        content.append("\n## 5. 错误归因分析")
        content.append("| 错误类型 | 占比 |")
        content.append("| :--- | :--- |")
        content.append(f"| 定位失败 | {error_stats.get('location_failure', 0):.2%} |")
        content.append(f"| 验证拒绝 | {error_stats.get('verification_rejection', 0):.2%} |")
        content.append(f"| 推理错误 | {error_stats.get('reasoning_failure', 0):.2%} |")
        
        # 3. 保存文件
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(content))
            
        print(f"报告已生成: {self.report_path}")

if __name__ == "__main__":
    ReportGenerator().generate()
