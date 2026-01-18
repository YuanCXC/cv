"""
================================================================================
Qwen-VL 模型包装器
================================================================================

本模块封装了DashScope API的Qwen-VL模型调用。
提供多模态对话、证据定位等功能。
================================================================================
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Union
import dashscope
from http import HTTPStatus
import sys
import os

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config

class QwenVLModel:
    def __init__(self):
        """初始化Qwen-VL模型客户端"""
        self.api_key = config.DASHSCOPE_API_KEY
        if not self.api_key:
            raise ValueError("未找到DASHSCOPE_API_KEY，请检查.env文件或环境变量")
        
        dashscope.api_key = self.api_key
        self.model_name = config.MODEL_NAME
        self.display_name = config.DISPLAY_MODEL_NAME
        
    def chat(self, 
             image_paths: List[str], 
             prompt: str, 
             history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        与Qwen-VL进行对话
        
        Args:
            image_paths (List[str]): 图片路径列表
            prompt (str): 文本提示
            history (List[Dict]): 对话历史
            
        Returns:
            Dict: 包含 'text' (回答文本) 和 'usage' (Token消耗)
        """
        messages = []
        if history:
            messages.extend(history)
            
        # 构建当前轮次的消息
        content = []
        for img_path in image_paths:
            # DashScope支持本地文件路径(file://)或URL
            # 这里处理本地路径，确保是绝对路径
            abs_path = os.path.abspath(img_path)
            content.append({"image": f"file://{abs_path}"})
            
        content.append({"text": prompt})
        messages.append({"role": "user", "content": content})
        
        try:
            start_time = time.time()
            response = dashscope.MultiModalConversation.call(
                model=self.model_name,
                messages=messages
            )
            end_time = time.time()
            
            if response.status_code == HTTPStatus.OK:
                output_text = response.output.choices[0].message.content[0]['text']
                usage = response.usage
                
                return {
                    "text": output_text,
                    "usage": usage,
                    "latency": end_time - start_time,
                    "status": "success"
                }
            else:
                return {
                    "text": f"Error: {response.code} - {response.message}",
                    "usage": None,
                    "latency": end_time - start_time,
                    "status": "error"
                }
                
        except Exception as e:
            return {
                "text": f"Exception: {str(e)}",
                "usage": None,
                "latency": 0,
                "status": "error"
            }

    def generate_evidence_prompt(self, question: str) -> str:
        """
        生成用于定位证据的提示词
        
        Args:
            question (str): 原始问题
            
        Returns:
            str: 提示词，引导模型输出需要关注的物体或区域
        """
        prompt = (
            f"为了回答问题: '{question}'，我们需要观察图片中的哪些具体物体或区域？\n"
            "请列出最关键的1-3个物体名称，用逗号分隔。\n"
            "例如，如果问题是'那个穿红衣服的人在干什么？'，请回答'穿红衣服的人'。\n"
            "请直接输出物体名称，不要包含其他废话。"
        )
        return prompt

    def refine_answer_with_evidence(self, question: str, evidence_desc: str) -> str:
        """
        结合证据生成最终答案的提示词
        
        Args:
            question (str): 原始问题
            evidence_desc: 证据描述（通常是图片内容的进一步分析）
            
        Returns:
            str: 提示词
        """
        prompt = (
            f"基于重点观察区域（{evidence_desc}），请再次回答问题: '{question}'\n"
            "请给出简短、准确的答案。"
        )
        return prompt

if __name__ == "__main__":
    # 简单测试
    model = QwenVLModel()
    print(f"Model initialized: {model.display_name} (API: {model.model_name})")
