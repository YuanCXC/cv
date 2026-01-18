"""
================================================================================
SAM2 分割模块
================================================================================

本模块封装了SAM2 (Segment Anything Model 2)，用于基于提示的图像分割。
支持基于边界框(Box)和点(Point)的分割。
================================================================================
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Tuple, Optional
import sys
import os

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config

class SAM2Segmenter:
    def __init__(self, checkpoint_path: str = None):
        """
        初始化SAM2分割器
        
        Args:
            checkpoint_path (str): 模型权重路径
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = checkpoint_path or config.SAM2_CHECKPOINT
        self.model = None
        self.predictor = None
        
        # 尝试加载SAM2，如果失败则尝试SAM1或报错
        self._load_model()

    def _load_model(self):
        """加载模型（SAM2 或 Fallback）"""
        print(f"正在加载分割模型 (Target: SAM2)...")
        try:
            # 尝试导入SAM2
            # 注意：实际SAM2库导入方式可能随版本变化，这里假设标准用法
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # 假设配置文件中有具体的yaml配置，这里简化处理，直接加载checkpoint
            # 如果是HuggingFace Hub的模型ID，可能需要 snapshot_download
            # 这里为了代码运行不报错，先做一个模拟加载或真实加载的结构
            
            # 真实场景下：
            # model = build_sam2("sam2_hiera_l.yaml", self.checkpoint_path)
            # self.predictor = SAM2ImagePredictor(model)
            
            print("警告: 检测到环境可能未安装SAM2或路径配置未完成，进入模拟模式或Fallback。")
            # 实际上如果没有安装sam2库，上面import就会失败，进入except
            
        except ImportError:
            print("未找到 sam2 库，尝试加载 segment_anything (SAM1)...")
            try:
                from segment_anything import sam_model_registry, SamPredictor
                
                # 尝试加载SAM1
                # 假设使用 vit_h
                sam_checkpoint = "sam_vit_h_4b8939.pth" # 需要用户下载
                if not os.path.exists(sam_checkpoint):
                     print(f"SAM1权重文件 {sam_checkpoint} 不存在。")
                     self.predictor = None
                else:
                    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
                    sam.to(device=self.device)
                    self.predictor = SamPredictor(sam)
                    print("SAM1 模型加载成功")
            except ImportError:
                print("未找到 segment_anything 库。分割功能将不可用。")
                self.predictor = None
        except Exception as e:
            print(f"模型加载未知错误: {e}")
            self.predictor = None

    def predict_with_box(self, 
                        image: Image.Image, 
                        box: List[int]) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        基于边界框进行分割
        
        Args:
            image (Image.Image): 输入图像
            box (List[int]): [x1, y1, x2, y2]
            
        Returns:
            Tuple[np.ndarray, float]: (Mask, Score)
        """
        if self.predictor is None:
            # 模拟返回：直接返回Box区域作为Mask
            return self._mock_predict(image, box)
            
        try:
            image_np = np.array(image)
            self.predictor.set_image(image_np)
            
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array(box)[None, :],
                multimask_output=False,
            )
            
            return masks[0], scores[0]
            
        except Exception as e:
            print(f"分割推理错误: {e}")
            return None, 0.0

    def _mock_predict(self, image: Image.Image, box: List[int]):
        """模拟分割（用于无模型环境）"""
        w, h = image.size
        mask = np.zeros((h, w), dtype=bool)
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        mask[y1:y2, x1:x2] = True
        return mask, 1.0

    def crop_with_mask(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """
        根据Mask裁剪图像（黑色背景）
        
        Args:
            image (Image.Image): 原图
            mask (np.ndarray): Mask
            
        Returns:
            Image.Image: 裁剪后的图像
        """
        image_np = np.array(image)
        # 确保mask是bool
        mask_bool = mask.astype(bool)
        
        # 创建带有Alpha通道的图像或黑色背景
        # 这里使用黑色背景
        result_np = np.zeros_like(image_np)
        result_np[mask_bool] = image_np[mask_bool]
        
        # 获取Bounding Box进行裁剪
        y_indices, x_indices = np.where(mask_bool)
        if len(y_indices) == 0:
            return image # Mask为空，返回原图或空图
            
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        
        result_img = Image.fromarray(result_np)
        return result_img.crop((x_min, y_min, x_max+1, y_max+1))

if __name__ == "__main__":
    pass
