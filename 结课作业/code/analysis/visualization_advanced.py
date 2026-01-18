"""
================================================================================
高级可视化与报告生成模块
================================================================================

本脚本基于现有的实验结果，生成高质量的图表和Excel汇总表。
包括：
1. 消融实验对比图 (Bar Chart)
2. 可靠性 OOD 评测图 (Bar Chart)
3. 目标大小鲁棒性分析图 (Bar Chart)
4. 效率分析图 (Bar Chart)
5. 错误归因分析图 (Pie Chart)
6. 汇总 Excel 表格生成
================================================================================
"""

import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from datetime import datetime
import sys

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config

# 配置 seaborn 样式
sns.set_theme(style="whitegrid")

# 中文字体设置
def find_chinese_font():
    font_dirs = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts"), Path.home() / ".fonts"]
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei']
    
    for font_dir in font_dirs:
        if font_dir.exists():
            for ext in ['.ttf', '.ttc', '.otf']:
                for font_file in font_dir.glob(f'*{ext}'):
                    try:
                        font_name = font_file.stem.lower()
                        for cn_font in chinese_fonts:
                            if cn_font.lower() in font_name:
                                return str(font_file)
                    except:
                        continue
    return None

font_path = find_chinese_font()
if font_path:
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False
    print(f"使用字体: {font_name}")

class AdvancedVisualizer:
    def __init__(self):
        self.results_dir = config.RESULTS_DIR
        self.timestamp = datetime.now().strftime("%Y%m%d")
        
    def load_json(self, path):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data[0] if isinstance(data, list) else data
        return {}

    def save_plot(self, filename):
        path = os.path.join(self.results_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"图片已保存: {path}")

    def plot_ablation_comparison(self):
        """生成消融实验对比图"""
        # 加载数据
        baseline = self.load_json(os.path.join(self.results_dir, "ablation", "baseline_metrics.json"))
        no_clip = self.load_json(os.path.join(self.results_dir, "ablation", "loop_no_clip_metrics.json"))
        full = self.load_json(os.path.join(self.results_dir, "ablation", "full_loop_metrics.json"))
        
        data = pd.DataFrame([
            {"Config": "Baseline (One-shot)", "Accuracy": baseline.get("accuracy_vqa", baseline.get("accuracy", 0))},
            {"Config": "Loop w/o CLIP", "Accuracy": no_clip.get("accuracy", 0)},
            {"Config": "Full Loop", "Accuracy": full.get("accuracy", 0)}
        ])
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x="Config", y="Accuracy", data=data, palette="viridis")
        
        plt.title("消融实验：系统准确率对比", fontsize=14, pad=20)
        plt.xlabel("实验配置", fontsize=12)
        plt.ylabel("准确率", fontsize=12)
        plt.ylim(0, 1.0)
        
        # 添加数值标签
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2%}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points')
            
        self.save_plot(f"analysis_ablation_{self.timestamp}.png")

    def plot_reliability_analysis(self):
        """生成可靠性分析图（OOD和大小分布）"""
        # 1. OOD
        ood = self.load_json(os.path.join(self.results_dir, "reliability", "ood_metrics.json"))
        baseline = self.load_json(os.path.join(self.results_dir, "ablation", "full_loop_metrics.json")) # 用Full Loop作为参照
        
        ood_data = pd.DataFrame([
            {"Condition": "Standard", "Accuracy": baseline.get("accuracy", 0)},
            {"Condition": "OOD (Perturbed)", "Accuracy": ood.get("accuracy", 0)}
        ])
        
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x="Condition", y="Accuracy", data=ood_data, palette="magma")
        plt.title("鲁棒性测试：标准 vs OOD", fontsize=14)
        plt.ylim(0, 1.0)
        
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
            
        self.save_plot(f"analysis_ood_{self.timestamp}.png")
        
        # 2. Size Analysis
        # 注意：这里需要重新计算Size Acc，或者读取已有的
        # 假设 reliability.py 生成了 size_analysis.json，如果没生成，这里尝试读取
        # 由于之前的reliability.py里analyze_size_robustness好像没有被main调用，这里可能缺文件
        # 我们先检查文件是否存在，不存在则跳过或现场计算
        # 为了演示，我们假设文件存在或者从full_loop_predictions.json现场算
        
        pred_path = os.path.join(self.results_dir, "ablation", "full_loop_predictions.json")
        if os.path.exists(pred_path):
            with open(pred_path, 'r', encoding='utf-8') as f:
                preds = json.load(f)
            
            # 简易计算Size Acc
            buckets = {"Small": [], "Medium": [], "Large": []}
            for p in preds:
                bbox = p.get("bbox")
                acc = p.get("accuracy", 0)
                if bbox:
                    area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
                    root_area = area**0.5
                    if root_area < 32: buckets["Small"].append(acc)
                    elif root_area < 96: buckets["Medium"].append(acc)
                    else: buckets["Large"].append(acc)
            
            size_data = []
            for k, v in buckets.items():
                mean_acc = sum(v)/len(v) if v else 0
                size_data.append({"Size": k, "Accuracy": mean_acc, "Count": len(v)})
                
            size_df = pd.DataFrame(size_data)
            
            plt.figure(figsize=(8, 6))
            ax = sns.barplot(x="Size", y="Accuracy", data=size_df, palette="Blues")
            plt.title("不同目标大小下的准确率", fontsize=14)
            plt.ylim(0, 1.0)
            
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
                
            self.save_plot(f"analysis_size_{self.timestamp}.png")

    def plot_efficiency(self):
        """生成效率分析图"""
        eff = self.load_json(os.path.join(self.results_dir, "efficiency", "analysis.json"))
        bench = self.load_json(os.path.join(self.results_dir, "efficiency", "benchmark.json"))
        
        # 调用次数
        calls = pd.DataFrame([
            {"Module": "SAM2", "Calls": eff.get("avg_sam_calls", 0)},
            {"Module": "CLIP", "Calls": eff.get("avg_clip_calls", 0)},
            {"Module": "Qwen-VL", "Calls": eff.get("avg_qwen_calls", 0)}
        ])
        
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(x="Module", y="Calls", data=calls, palette="Set2")
        plt.title("平均每个样本的工具调用次数", fontsize=14)
        plt.ylabel("调用次数")
        
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
            
        self.save_plot(f"analysis_efficiency_calls_{self.timestamp}.png")

    def plot_error_analysis(self):
        """生成错误归因饼图"""
        stats = self.load_json(os.path.join(self.results_dir, "error_analysis", "error_stats.json"))
        
        # 移除 correct
        errors = {k: v for k, v in stats.items() if k != "correct" and v > 0}
        
        if not errors:
            print("没有错误数据，跳过饼图")
            return
            
        labels = list(errors.keys())
        sizes = list(errors.values())
        
        # 翻译标签
        label_map = {
            "location_failure": "定位失败",
            "verification_rejection": "验证拒绝",
            "reasoning_failure": "推理错误"
        }
        labels = [label_map.get(l, l) for l in labels]
        
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
        plt.title("失败案例归因分析", fontsize=14)
        
        self.save_plot(f"analysis_error_pie_{self.timestamp}.png")

    def generate_excel_report(self):
        """生成汇总Excel报告"""
        output_path = os.path.join(self.results_dir, f"summary_report_{self.timestamp}.xlsx")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 1. 概览 Sheet
            baseline = self.load_json(os.path.join(self.results_dir, "ablation", "baseline_metrics.json"))
            full = self.load_json(os.path.join(self.results_dir, "ablation", "full_loop_metrics.json"))
            ood = self.load_json(os.path.join(self.results_dir, "reliability", "ood_metrics.json"))
            
            overview_data = [
                {"Metric": "Baseline Accuracy", "Value": baseline.get("accuracy_vqa", baseline.get("accuracy", 0))},
                {"Metric": "Full Loop Accuracy", "Value": full.get("accuracy", 0)},
                {"Metric": "OOD Accuracy", "Value": ood.get("accuracy", 0)},
                {"Metric": "Sample Count", "Value": full.get("samples", 0)}
            ]
            pd.DataFrame(overview_data).to_excel(writer, sheet_name="Overview", index=False)
            
            # 2. 消融实验 Sheet
            ablation_data = []
            for name in ["baseline", "loop_no_clip", "full_loop"]:
                path = os.path.join(self.results_dir, "ablation", f"{name}_metrics.json")
                if os.path.exists(path):
                    data = self.load_json(path)
                    data["config"] = name
                    ablation_data.append(data)
            pd.DataFrame(ablation_data).to_excel(writer, sheet_name="Ablation", index=False)
            
            # 3. 详细预测结果 Sheet (Full Loop)
            full_preds_path = os.path.join(self.results_dir, "ablation", "full_loop_predictions.json")
            if os.path.exists(full_preds_path):
                with open(full_preds_path, 'r', encoding='utf-8') as f:
                    preds = json.load(f)
                # 简化字段
                simple_preds = []
                for p in preds:
                    simple_preds.append({
                        "ID": p.get("id"),
                        "Question": p.get("question"),
                        "GT Answer": str(p.get("gt_answers", [])),
                        "Prediction": p.get("prediction"),
                        "Accuracy": p.get("accuracy"),
                        "BBox": str(p.get("bbox")),
                        "CLIP Score": p.get("clip_score")
                    })
                pd.DataFrame(simple_preds).to_excel(writer, sheet_name="Detailed_Predictions", index=False)
                
        print(f"Excel报告已生成: {output_path}")

if __name__ == "__main__":
    viz = AdvancedVisualizer()
    print("开始生成高级可视化报表...")
    viz.plot_ablation_comparison()
    viz.plot_reliability_analysis()
    viz.plot_efficiency()
    viz.plot_error_analysis()
    viz.generate_excel_report()
    print("所有任务完成。")
