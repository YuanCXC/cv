"""
================================================================================
项目统一运行入口
================================================================================

本脚本用于一键运行闭环视觉系统的所有实验和分析任务。
支持命令行参数控制运行模块。

Usage:
    python run.py --all
    python run.py --ablation --report
    python run.py --help
================================================================================
"""

import argparse
import sys
import os

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.ablation import AblationRunner
from experiments.reliability import ReliabilityEvaluator
from experiments.efficiency import EfficiencyAnalyzer
from analysis.error_analysis import ErrorAnalyzer
from analysis.report import ReportGenerator
from analysis.visualization_advanced import AdvancedVisualizer
from core import config

def run_ablation(limit=None):
    print("\n>>> 正在运行消融实验 (Ablation Study)...")
    runner = AblationRunner()
    runner.run_all(limit=limit)

def run_reliability(limit=None):
    print("\n>>> 正在运行可靠性评测 (Reliability Test)...")
    evaluator = ReliabilityEvaluator()
    evaluator.run_ood_test(limit=limit)
    # 大小分析依赖于 ablation 的结果，无需单独运行推理，只需分析逻辑（集成在visualization中）

def run_efficiency(limit=None):
    print("\n>>> 正在运行效率分析 (Efficiency Analysis)...")
    analyzer = EfficiencyAnalyzer()
    # 分析全闭环实验的效率
    target_file = os.path.join(config.RESULTS_DIR, "ablation", "full_loop_predictions.json")
    if os.path.exists(target_file):
        analyzer.analyze_results(target_file)
    else:
        print(f"警告：未找到 {target_file}，跳过离线效率分析")
    
    # 运行实时基准测试
    analyzer.benchmark_resource_usage(limit=min(limit, 20) if limit else 5)

def run_analysis():
    print("\n>>> 正在运行结果分析 (Error Analysis & Visualization)...")
    
    # 1. 错误归因
    error_analyzer = ErrorAnalyzer()
    target_file = os.path.join(config.RESULTS_DIR, "ablation", "full_loop_predictions.json")
    if os.path.exists(target_file):
        error_analyzer.analyze(target_file)
    
    # 2. 生成图表
    viz = AdvancedVisualizer()
    viz.plot_ablation_comparison()
    viz.plot_reliability_analysis()
    viz.plot_efficiency()
    viz.plot_error_analysis()
    viz.generate_excel_report()

def run_report():
    print("\n>>> 正在生成实验报告 (Markdown Report)...")
    ReportGenerator().generate()

def main():
    parser = argparse.ArgumentParser(description="闭环多模态视觉系统实验运行脚本")
    parser.add_argument("--all", action="store_true", help="运行所有实验和分析")
    parser.add_argument("--ablation", action="store_true", help="运行消融实验")
    parser.add_argument("--reliability", action="store_true", help="运行可靠性评测")
    parser.add_argument("--efficiency", action="store_true", help="运行效率分析")
    parser.add_argument("--analysis", action="store_true", help="运行结果分析与可视化")
    parser.add_argument("--report", action="store_true", help="生成 Markdown 报告")
    parser.add_argument("--limit", type=int, default=100, help="限制处理样本数量 (默认: 100)")
    
    args = parser.parse_args()
    
    # 如果未指定任何参数，打印帮助
    if not any(vars(args).values()):
        parser.print_help()
        return

    if args.all:
        args.ablation = True
        args.reliability = True
        args.efficiency = True
        args.analysis = True
        args.report = True

    if args.ablation:
        run_ablation(limit=args.limit)
        
    if args.reliability:
        run_reliability(limit=args.limit)
        
    if args.efficiency:
        run_efficiency(limit=args.limit)
        
    if args.analysis:
        run_analysis()
        
    if args.report:
        run_report()
        
    print("\n=== 所有任务执行完毕 ===")

if __name__ == "__main__":
    main()
