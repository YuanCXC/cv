"""
================================================================================
工具函数模块
================================================================================

本文件提供项目所需的通用工具函数。
包括图像处理、数据加载、结果保存等功能。
================================================================================
"""

import os
import json
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np


def image_to_base64(image_path: str, max_size: int = 1024) -> Optional[str]:
    """
    将图像转换为base64编码
    
    Args:
        image_path (str): 图像文件路径
        max_size (int): 最大边长，默认为1024
        
    Returns:
        Optional[str]: base64编码的图像字符串，如果失败则返回None
    """
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
            
    except Exception as e:
        print(f"图片处理错误 {image_path}: {e}")
        return None


def load_metadata(data_dir: str) -> List[Dict[str, Any]]:
    """
    加载数据集元数据
    
    Args:
        data_dir (str): 数据目录路径
        
    Returns:
        List[Dict]: 元数据列表
    """
    metadata_path = os.path.join(data_dir, "metadata.json")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"加载了 {len(metadata)} 条元数据")
    return metadata


def save_results(results: List[Dict[str, Any]], output_path: str):
    """
    保存结果到JSON文件
    
    Args:
        results (List[Dict]): 结果列表
        output_path (str): 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_path}")


def load_image(image_path: str) -> Optional[Image.Image]:
    """
    加载图像
    
    Args:
        image_path (str): 图像文件路径
        
    Returns:
        Optional[Image.Image]: PIL图像对象，如果失败则返回None
    """
    try:
        if os.path.exists(image_path):
            return Image.open(image_path).convert("RGB")
        return None
    except Exception as e:
        print(f"图像加载错误 {image_path}: {e}")
        return None


def crop_image_with_bbox(image: Image.Image, bbox: List[int], 
                    padding: int = 10) -> Image.Image:
    """
    使用边界框裁剪图像
    
    Args:
        image (Image.Image): PIL图像对象
        bbox (List[int]): 边界框[x1, y1, x2, y2]
        padding (int): 裁剪边距，默认为10像素
        
    Returns:
        Image.Image: 裁剪后的图像
    """
    x1, y1, x2, y2 = bbox
    
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.width, x2 + padding)
    y2 = min(image.height, y2 + padding)
    
    return image.crop((x1, y1, x2, y2))


def compute_bbox_area(bbox: List[int]) -> int:
    """
    计算边界框面积
    
    Args:
        bbox (List[int]): 边界框[x1, y1, x2, y2]
        
    Returns:
        int: 边界框面积
    """
    x1, y1, x2, y2 = bbox
    return abs((x2 - x1) * (y2 - y1))


def classify_question(question: str) -> str:
    """
    问题类型分类
    
    Args:
        question (str): 问题文本
        
    Returns:
        str: 问题类型
    """
    q = question.lower()
    
    categories = {
        'counting': ['how many', '多少', 'count', 'number of', '数量'],
        'attribute': ['what color', 'what brand', 'what type', 'what kind', 'what year',
                      '颜色', '品牌', '类型', '年份', '时间'],
        'spatial': ['where', 'what is on the left', 'what is on the right', 'what is in front',
                    '位置', '左边', '右边', '前面', '后面'],
        'reading': ['what does it say', 'what does the sign say', 'what does the text say',
                    'what is written', 'read', '读取', '文字', '写的'],
        'yesno': ['is this', 'are these', 'was the', 'does her shirt say', 
                  '是否', '是不是', 'does it', 'is there', 'are there'],
        'identification': ['who is', 'what is the name', 'what is this', 'who was',
                          '谁', '名称', '是什么', 'what does']
    }
    
    for cat, keywords in categories.items():
        if any(kw in q for kw in keywords):
            return cat
    
    return 'other'


def normalize_answer(answer: str) -> str:
    """
    答案标准化
    
    Args:
        answer (str): 原始答案文本
        
    Returns:
        str: 标准化后的答案文本
    """
    if not answer:
        return ""
    
    import re
    answer = answer.lower().strip()
    answer = re.sub(r'[^\w\s]', ' ', answer)
    answer = re.sub(r'\s+', ' ', answer)
    
    return answer.strip()


def compute_mask_area(mask: np.ndarray) -> int:
    """
    计算mask面积
    
    Args:
        mask (np.ndarray): 二值mask数组
        
    Returns:
        int: mask面积
    """
    return int(mask.sum())


def batch_images(images: List[Image.Image], batch_size: int = 4) -> List[List[Image.Image]]:
    """
    将图像列表分批
    
    Args:
        images (List[Image.Image]): 图像列表
        batch_size (int): 批次大小
        
    Returns:
        List[List[Image.Image]]: 批次列表
    """
    batches = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batches.append(batch)
    
    return batches
