"""
信号预处理模块
=============

包含:
- 小波去噪 (Wavelet Denoising)
- 基线漂移校正 (Baseline Wander Removal)
- R峰检测 (R-peak Detection)
- 心拍分割 (Beat Segmentation)
"""

from .wavelet_denoising import WaveletDenoiser
from .baseline_correction import BaselineCorrector
from .rpeak_detection import RPeakDetector
from .beat_segmentation import BeatSegmenter
from .signal_pipeline import SignalPreprocessor

__all__ = [
    'WaveletDenoiser',
    'BaselineCorrector', 
    'RPeakDetector',
    'BeatSegmenter',
    'SignalPreprocessor'
]
