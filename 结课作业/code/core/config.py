"""
================================================================================
配置模块
================================================================================

本文件定义了项目的所有配置参数，包括路径、模型设置、API密钥等。
所有路径均基于项目根目录配置。
================================================================================
"""

import os
import sys

# 项目根目录
# 获取当前脚本所在目录的上一级目录作为项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 目录配置
CODE_DIR = os.path.join(PROJECT_ROOT, "code")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


DATA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "data")

IMAGES_DIR = os.path.join(DATA_DIR, "images")

# 确保目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)

# 配置文件路径
# .env 位于结课作业的上级目录
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(PROJECT_ROOT)), ".env")

def load_api_key():
    """从.env文件加载API Key"""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if api_key:
        return api_key
    
    if os.path.exists(ENV_PATH):
        try:
            with open(ENV_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DASHSCOPE_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        os.environ["DASHSCOPE_API_KEY"] = key
                        return key
        except Exception as e:
            print(f"读取.env文件失败: {e}")
    
    return None

# 加载API Key
DASHSCOPE_API_KEY = load_api_key()

# 模型配置
# 注意：DashScope API通常使用 qwen-vl-max, qwen-vl-plus。
# 用户指定 qwen3-vl-8b，若不可用需回退或修改。
# 这里设置为 qwen-vl-max 作为默认API调用名，但在逻辑中可以记录为 qwen3-vl-8b
MODEL_NAME = "qwen-vl-max"  # 实际调用的API模型名
DISPLAY_MODEL_NAME = "qwen3-vl-8b" # 显示/记录用的模型名

# CLIP模型配置
CLIP_MODEL_ID = "openai/clip-vit-base-patch32" # 或者本地路径

# SAM2模型配置
# 假设使用 facebook/sam2-hiera-large 或其他可用版本
SAM2_CHECKPOINT = "facebook/sam2-hiera-large" 

# 实验参数
MAX_ITERATIONS = 3  # 最大迭代轮数
CONFIDENCE_THRESHOLD = 0.8  # 提前停止的置信度阈值
CLIP_SIMILARITY_THRESHOLD = 0.2 # CLIP相关性阈值

# 随机种子
RANDOM_SEED = 42
