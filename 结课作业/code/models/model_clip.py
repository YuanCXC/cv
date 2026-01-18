"""
================================================================================
CLIP 验证模块
================================================================================

本模块封装了CLIP模型，用于计算图像和文本的相似度。
主要用于闭环系统中验证分割出的区域（Evidence）与问题/答案的相关性。
================================================================================
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import sys
import os

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config
from typing import List, Union, Tuple

class CLIPVerifier:
    def __init__(self, model_id: str = None):
        """
        初始化CLIP验证器
        
        Args:
            model_id (str): 模型ID，默认为None（使用config配置）
        """
        self.model_id = model_id or config.CLIP_MODEL_ID
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"正在加载CLIP模型: {self.model_id} 到 {self.device}...")
        try:
            self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
            print("CLIP模型加载完成")
        except Exception as e:
            print(f"CLIP模型加载失败: {e}")
            raise

    def compute_similarity(self, image: Image.Image, text: str) -> float:
        """
        计算单张图像和单个文本的相似度
        
        Args:
            image (Image.Image): PIL图像
            text (str): 文本描述
            
        Returns:
            float: 相似度分数 (0-1)
        """
        try:
            inputs = self.processor(
                text=[text], 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 图像-文本相似度 (logits_per_image)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1) # 这里的softmax可能不适用单文本，单文本直接看logit或cosine
            
            # 使用 cosine similarity 更直接
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            # 归一化
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            # 计算余弦相似度
            similarity = torch.matmul(image_embeds, text_embeds.t()).item()
            
            # 简单的归一化到 0-1 (CLIP logits通常在logit_scale作用下很大，这里直接用原始余弦值)
            # 余弦相似度范围 -1 到 1，通常在 0.2-0.3 以上算相关
            return max(0.0, similarity)
            
        except Exception as e:
            print(f"CLIP计算出错: {e}")
            return 0.0

    def verify_evidence(self, evidence_image: Image.Image, question: str, threshold: float = None) -> bool:
        """
        验证证据图像是否与问题相关
        
        Args:
            evidence_image (Image.Image): 证据图像
            question (str): 问题
            threshold (float): 阈值
            
        Returns:
            bool: 是否通过验证
        """
        if threshold is None:
            threshold = config.CLIP_SIMILARITY_THRESHOLD
            
        score = self.compute_similarity(evidence_image, question)
        return score >= threshold

    def rank_candidates(self, images: List[Image.Image], text: str) -> List[Tuple[int, float]]:
        """
        对一组候选图像进行排序
        
        Args:
            images (List[Image.Image]): 图像列表
            text (str): 文本
            
        Returns:
            List[Tuple[int, float]]: (索引, 分数) 列表，按分数降序排列
        """
        scores = []
        for idx, img in enumerate(images):
            score = self.compute_similarity(img, text)
            scores.append((idx, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    # 测试
    pass
