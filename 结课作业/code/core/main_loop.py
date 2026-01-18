"""
================================================================================
闭环式多模态视觉系统控制器
================================================================================

本模块实现了 "Qwen3-VL -> SAM2 -> CLIP -> Qwen3-VL" 的闭环推理流程。
================================================================================
"""

import re
import os
import json
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import numpy as np
import sys
import os

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config
from models.model_qwen import QwenVLModel
from models.model_sam2 import SAM2Segmenter
from models.model_clip import CLIPVerifier
from core.utils import load_image, crop_image_with_bbox

class VQAClosedLoopSystem:
    def __init__(self):
        print("初始化闭环系统...")
        self.qwen = QwenVLModel()
        self.sam = SAM2Segmenter()
        self.clip = CLIPVerifier()
        
        # 统计数据
        self.stats = {
            "iterations": [],
            "sam_calls": 0,
            "clip_calls": 0,
            "qwen_calls": 0
        }

    def reset_stats(self):
        self.stats = {
            "iterations": [],
            "sam_calls": 0,
            "clip_calls": 0,
            "qwen_calls": 0
        }

    def parse_bbox(self, text: str, width: int, height: int) -> Optional[List[int]]:
        """
        从文本解析边界框
        假设格式为 [x1, y1, x2, y2] 或 (x1, y1), (x2, y2) 且为归一化坐标(0-1000)
        """
        # 匹配 [100, 200, 300, 400]
        pattern1 = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        match = re.search(pattern1, text)
        
        if match:
            coords = [int(g) for g in match.groups()]
            # 转换为绝对坐标 (Qwen通常输出0-1000)
            x1 = int(coords[0] / 1000 * width)
            y1 = int(coords[1] / 1000 * height)
            x2 = int(coords[2] / 1000 * width)
            y2 = int(coords[3] / 1000 * height)
            return [x1, y1, x2, y2]
            
        return None

    def solve(self, image_path: str, question: str, use_clip: bool = True) -> Dict[str, Any]:
        """
        解决单个VQA问题
        Args:
            image_path (str): 图像路径
            question (str): 问题
            use_clip (bool): 是否使用CLIP验证
        """
        self.reset_stats()
        
        image = load_image(image_path)
        if image is None:
            return {"answer": "Error: Image load failed", "error": "Image load failed"}
            
        width, height = image.size
        
        # ------------------------------------------------------------------
        # 步骤 1: 初始分析与证据定位
        # ------------------------------------------------------------------
        print(f"\n[Qwen] 分析问题: {question}")
        evidence_prompt = self.qwen.generate_evidence_prompt(question)
        
        # 调用Qwen获取关注目标
        # 注意：这里我们直接问Qwen要检测框
        detect_prompt = f"请找出图中与问题'{question}'最相关的物体，并以[x1, y1, x2, y2]格式（0-1000坐标系）返回其边界框。只返回坐标，不要废话。"
        
        self.stats["qwen_calls"] += 1
        detect_response = self.qwen.chat([image_path], detect_prompt)
        detect_text = detect_response["text"]
        print(f"[Qwen] 建议关注区域: {detect_text}")
        
        bbox = self.parse_bbox(detect_text, width, height)
        
        evidence_image = None
        evidence_score = 0.0
        
        # ------------------------------------------------------------------
        # 步骤 2 & 3: 分割与裁剪 (SAM2)
        # ------------------------------------------------------------------
        if bbox:
            print(f"[SAM2] 对区域 {bbox} 进行分割...")
            self.stats["sam_calls"] += 1
            mask, sam_score = self.sam.predict_with_box(image, bbox)
            
            if mask is not None:
                evidence_image = self.sam.crop_with_mask(image, mask)
                # 保存临时证据图用于调试（可选）
                # evidence_image.save("temp_evidence.jpg")
        else:
            print("[System] 未能解析出有效边界框，跳过分割步骤。")

        # ------------------------------------------------------------------
        # 步骤 4: 验证 (CLIP)
        # ------------------------------------------------------------------
        passed_verification = False
        if evidence_image:
            if use_clip:
                print("[CLIP] 验证证据相关性...")
                self.stats["clip_calls"] += 1
                evidence_score = self.clip.compute_similarity(evidence_image, question)
                print(f"[CLIP] 相关性得分: {evidence_score:.4f}")
                
                if evidence_score >= config.CLIP_SIMILARITY_THRESHOLD:
                    passed_verification = True
                    print("[System] 证据有效，将回馈给VLM。")
                else:
                    print("[System] 证据相关性低，丢弃。")
                    evidence_image = None
            else:
                # 不使用CLIP，默认全部接受
                print("[System] CLIP验证已禁用，默认接受证据。")
                passed_verification = True
                evidence_score = 1.0 # 模拟满分

        # ------------------------------------------------------------------
        # 步骤 5: 最终推理 (Qwen)
        # ------------------------------------------------------------------
        final_prompt = f"请回答问题: {question}\n请注意：必须简短回答，不要废话。只输出答案本身。"
        image_inputs = [image_path]
        
        if passed_verification and evidence_image:
            # 将证据图保存为临时文件传入 (DashScope需要路径)
            temp_evidence_path = os.path.join(config.RESULTS_DIR, "temp_evidence.jpg")
            evidence_image.save(temp_evidence_path)
            image_inputs.append(temp_evidence_path)
            final_prompt += "\n（已附上该区域的详细特写图作为图2，请结合图2回答）"
        
        print(f"[Qwen] 最终推理...")
        self.stats["qwen_calls"] += 1
        final_response = self.qwen.chat(image_inputs, final_prompt)
        final_answer = final_response["text"]
        
        print(f"--> 最终答案: {final_answer}")
        
        result = {
            "question": question,
            "answer": final_answer,
            "bbox": bbox,
            "clip_score": evidence_score,
            "stats": self.stats
        }
        
        return result

if __name__ == "__main__":
    # 简单测试
    pass
