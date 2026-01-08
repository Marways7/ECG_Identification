"""
信号预处理管道
==============

整合所有预处理步骤的统一接口
实现端到端的ECG信号处理流程
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass, field
from loguru import logger

from .wavelet_denoising import WaveletDenoiser, AdaptiveWaveletDenoiser
from .baseline_correction import BaselineCorrector, MultiStageBaselineCorrector
from .rpeak_detection import RPeakDetector
from .beat_segmentation import BeatSegmenter


@dataclass
class PreprocessingConfig:
    """预处理配置"""
    
    # 采样率 (ADS1292R默认250Hz)
    sampling_rate: float = 250.0
    
    # 小波去噪参数
    wavelet: str = 'db4'
    wavelet_level: Optional[int] = None
    wavelet_mode: str = 'soft'
    
    # 基线校正参数
    baseline_method: str = 'morphological'
    
    # R峰检测参数
    rpeak_method: str = 'pan_tompkins'
    
    # 心拍分割参数
    beat_pre_r: float = 0.25  # R峰前 (秒)
    beat_post_r: float = 0.45  # R峰后 (秒)
    beat_target_length: int = 140  # 归一化长度
    
    # 质量控制参数
    min_beats: int = 10
    max_abnormal_ratio: float = 0.3
    
    # 数据增强
    enable_augmentation: bool = False


@dataclass
class PreprocessingResult:
    """预处理结果"""
    
    # 原始信号
    raw_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # 处理后信号
    denoised_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    baseline_corrected: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # R峰检测结果
    r_peaks: np.ndarray = field(default_factory=lambda: np.array([]))
    rr_intervals: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # 心拍数据
    beats: np.ndarray = field(default_factory=lambda: np.array([]))
    beat_info: List[dict] = field(default_factory=list)
    beat_templates: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # 质量指标
    signal_quality: float = 0.0
    snr_improvement: float = 0.0
    heart_rate: float = 0.0
    
    # 中间结果
    baseline: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def is_valid(self) -> bool:
        """检查结果是否有效"""
        return len(self.beats) > 0 and len(self.r_peaks) > 1


class SignalPreprocessor:
    """
    ECG信号预处理器
    
    集成小波去噪、基线校正、R峰检测和心拍分割
    提供统一的处理接口
    
    Usage:
        preprocessor = SignalPreprocessor(config)
        result = preprocessor.process(ecg_signal)
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        初始化预处理器
        
        Args:
            config: 预处理配置 (默认使用默认配置)
        """
        self.config = config if config else PreprocessingConfig()
        
        # 初始化各处理模块
        self.denoiser = AdaptiveWaveletDenoiser(
            wavelet=self.config.wavelet,
            level=self.config.wavelet_level,
            threshold_mode=self.config.wavelet_mode
        )
        
        self.baseline_corrector = MultiStageBaselineCorrector(
            sampling_rate=self.config.sampling_rate
        )
        
        self.rpeak_detector = RPeakDetector(
            sampling_rate=self.config.sampling_rate,
            method=self.config.rpeak_method
        )
        
        self.beat_segmenter = BeatSegmenter(
            sampling_rate=self.config.sampling_rate,
            pre_r=self.config.beat_pre_r,
            post_r=self.config.beat_post_r,
            target_length=self.config.beat_target_length
        )
        
        logger.info("信号预处理器初始化完成")
    
    def process(
        self,
        ecg_signal: np.ndarray,
        return_all: bool = True
    ) -> PreprocessingResult:
        """
        执行完整的预处理流程
        
        处理步骤:
        1. 信号标准化
        2. 小波去噪
        3. 基线校正
        4. R峰检测
        5. 心拍分割
        6. 异常心拍过滤
        7. 模板提取
        
        Args:
            ecg_signal: 输入ECG信号
            return_all: 是否返回所有中间结果
            
        Returns:
            PreprocessingResult对象
        """
        result = PreprocessingResult()
        result.raw_signal = np.asarray(ecg_signal, dtype=np.float64)
        
        try:
            # Step 1: 信号标准化
            logger.info("Step 1: 信号标准化")
            normalized = self._normalize_signal(result.raw_signal)
            
            # Step 2: 小波去噪
            logger.info("Step 2: 小波去噪")
            denoised, denoise_details = self.denoiser.denoise(
                normalized, 
                return_details=True
            )
            result.denoised_signal = denoised
            if denoise_details:
                result.snr_improvement = denoise_details.get('snr_improvement', 0)
            
            # Step 3: 基线校正
            logger.info("Step 3: 基线校正")
            baseline_corrected, baseline = self.baseline_corrector.correct(denoised)
            result.baseline_corrected = baseline_corrected
            result.baseline = baseline
            
            # Step 4: R峰检测
            logger.info("Step 4: R峰检测")
            r_peaks, _ = self.rpeak_detector.detect(baseline_corrected)
            result.r_peaks = r_peaks
            
            if len(r_peaks) < 2:
                logger.warning("R峰数量不足，使用集成检测")
                r_peaks, _ = self.rpeak_detector.ensemble_detect(baseline_corrected)
                result.r_peaks = r_peaks
            
            # 计算RR间期
            if len(r_peaks) > 1:
                result.rr_intervals = np.diff(r_peaks) / self.config.sampling_rate * 1000  # ms
                result.heart_rate = 60000 / np.mean(result.rr_intervals)
            
            # Step 5: 心拍分割
            logger.info("Step 5: 心拍分割")
            beats, beat_info = self.beat_segmenter.segment(
                baseline_corrected,
                r_peaks,
                normalize=True,
                align=True
            )
            result.beats = beats
            result.beat_info = beat_info
            
            # Step 6: 异常心拍过滤
            logger.info("Step 6: 异常心拍过滤")
            if len(beats) > self.config.min_beats:
                filtered_beats, filtered_info = self.beat_segmenter.filter_abnormal_beats(
                    beats, beat_info
                )
                result.beats = filtered_beats
                result.beat_info = filtered_info
            
            # Step 7: 模板提取
            logger.info("Step 7: 模板提取")
            if len(result.beats) >= 3:
                templates, _ = self.beat_segmenter.get_beat_templates(
                    result.beats,
                    n_clusters=min(3, len(result.beats) // 10 + 1)
                )
                result.beat_templates = templates
            
            # 计算信号质量
            result.signal_quality = self._assess_signal_quality(result)
            
            logger.info(f"预处理完成: {len(result.beats)} 个有效心拍, "
                       f"心率 {result.heart_rate:.1f} BPM, "
                       f"质量评分 {result.signal_quality:.2f}")
            
        except Exception as e:
            logger.error(f"预处理失败: {e}")
            raise
        
        return result
    
    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        信号标准化
        
        Z-score标准化: x' = (x - μ) / σ
        
        Args:
            signal: 输入信号
            
        Returns:
            标准化后的信号
        """
        mean = np.mean(signal)
        std = np.std(signal)
        
        if std < 1e-10:
            return signal - mean
        
        return (signal - mean) / std
    
    def _assess_signal_quality(self, result: PreprocessingResult) -> float:
        """
        评估信号质量
        
        基于多个指标综合评估:
        1. R峰检测置信度
        2. RR间期规律性
        3. 心拍形态一致性
        4. 信噪比
        
        Args:
            result: 预处理结果
            
        Returns:
            质量评分 (0-1)
        """
        scores = []
        
        # 1. R峰数量评分
        expected_beats = len(result.raw_signal) / self.config.sampling_rate / 0.8  # 假设75bpm
        beat_ratio = min(len(result.r_peaks) / max(expected_beats, 1), 1.5)
        beat_score = 1.0 if 0.7 < beat_ratio < 1.3 else max(0, 1 - abs(beat_ratio - 1))
        scores.append(beat_score)
        
        # 2. RR间期规律性 (变异系数)
        if len(result.rr_intervals) > 1:
            cv = np.std(result.rr_intervals) / np.mean(result.rr_intervals)
            rr_score = max(0, 1 - cv * 2)  # CV < 0.5 视为良好
            scores.append(rr_score)
        
        # 3. 心拍形态一致性
        if len(result.beats) > 10:
            mean_beat = np.mean(result.beats, axis=0)
            correlations = [np.corrcoef(beat, mean_beat)[0, 1] for beat in result.beats]
            morph_score = np.mean(correlations)
            scores.append(max(0, morph_score))
        
        # 4. SNR改善评分
        snr_score = min(result.snr_improvement / 20, 1) if result.snr_improvement > 0 else 0.5
        scores.append(snr_score)
        
        # 综合评分
        quality = np.mean(scores) if scores else 0.0
        
        return quality
    
    def process_batch(
        self,
        signals: List[np.ndarray],
        labels: Optional[List[str]] = None
    ) -> Tuple[List[PreprocessingResult], Dict]:
        """
        批量处理多个信号
        
        Args:
            signals: 信号列表
            labels: 对应的标签列表
            
        Returns:
            results: 处理结果列表
            summary: 汇总统计
        """
        results = []
        success_count = 0
        total_beats = 0
        
        for i, signal in enumerate(signals):
            label = labels[i] if labels else f"Signal_{i}"
            
            try:
                result = self.process(signal)
                results.append(result)
                
                if result.is_valid():
                    success_count += 1
                    total_beats += len(result.beats)
                    
            except Exception as e:
                logger.error(f"处理信号 {label} 失败: {e}")
                results.append(PreprocessingResult())
        
        summary = {
            'total': len(signals),
            'success': success_count,
            'failed': len(signals) - success_count,
            'total_beats': total_beats,
            'avg_beats_per_signal': total_beats / max(success_count, 1)
        }
        
        logger.info(f"批量处理完成: {success_count}/{len(signals)} 成功, "
                   f"共 {total_beats} 个心拍")
        
        return results, summary
    
    def augment_beats(
        self,
        beats: np.ndarray,
        n_augmented: int = 2
    ) -> np.ndarray:
        """
        数据增强
        
        生成增强后的心拍数据:
        1. 时间平移
        2. 幅度缩放
        3. 添加微小噪声
        4. 时间拉伸/压缩
        
        Args:
            beats: 原始心拍 (n_beats, beat_length)
            n_augmented: 每个心拍生成的增强样本数
            
        Returns:
            增强后的心拍 (n_beats * (1 + n_augmented), beat_length)
        """
        if not self.config.enable_augmentation:
            return beats
        
        augmented_list = [beats]
        
        for _ in range(n_augmented):
            aug_beats = []
            
            for beat in beats:
                aug_beat = beat.copy()
                
                # 随机选择增强方式
                aug_type = np.random.choice(['shift', 'scale', 'noise', 'stretch'])
                
                if aug_type == 'shift':
                    # 时间平移 (±5个采样点)
                    shift = np.random.randint(-5, 6)
                    aug_beat = np.roll(aug_beat, shift)
                    
                elif aug_type == 'scale':
                    # 幅度缩放 (0.9-1.1倍)
                    scale = np.random.uniform(0.9, 1.1)
                    aug_beat = aug_beat * scale
                    
                elif aug_type == 'noise':
                    # 添加高斯噪声
                    noise = np.random.normal(0, 0.02, len(aug_beat))
                    aug_beat = aug_beat + noise
                    
                elif aug_type == 'stretch':
                    # 时间拉伸/压缩 (0.95-1.05倍)
                    from scipy.interpolate import interp1d
                    stretch = np.random.uniform(0.95, 1.05)
                    x_orig = np.linspace(0, 1, len(aug_beat))
                    x_new = np.linspace(0, 1, int(len(aug_beat) * stretch))
                    f = interp1d(x_orig, aug_beat, kind='cubic', fill_value='extrapolate')
                    stretched = f(x_new)
                    # 重采样回原长度
                    f2 = interp1d(np.linspace(0, 1, len(stretched)), stretched, 
                                  kind='cubic', fill_value='extrapolate')
                    aug_beat = f2(x_orig)
                
                aug_beats.append(aug_beat)
            
            augmented_list.append(np.array(aug_beats))
        
        return np.vstack(augmented_list)


class EDRExtractor:
    """
    心电导出呼吸 (ECG-Derived Respiration) 提取器
    
    从ECG信号中提取呼吸信号，用于CRC计算
    
    方法:
    1. R峰幅度法 (R-amplitude)
    2. R-R间期法 (RR-interval)
    3. QRS面积法 (QRS area)
    """
    
    def __init__(self, sampling_rate: float = 200.0):
        self.sampling_rate = sampling_rate
    
    def extract(
        self,
        ecg_signal: np.ndarray,
        r_peaks: np.ndarray,
        method: str = 'amplitude'
    ) -> np.ndarray:
        """
        提取EDR信号
        
        Args:
            ecg_signal: ECG信号
            r_peaks: R峰位置
            method: 提取方法
            
        Returns:
            EDR信号 (与R峰数量相同长度)
        """
        if method == 'amplitude':
            return self._amplitude_edr(ecg_signal, r_peaks)
        elif method == 'rri':
            return self._rri_edr(r_peaks)
        elif method == 'area':
            return self._area_edr(ecg_signal, r_peaks)
        else:
            raise ValueError(f"不支持的方法: {method}")
    
    def _amplitude_edr(
        self, 
        ecg_signal: np.ndarray, 
        r_peaks: np.ndarray
    ) -> np.ndarray:
        """R峰幅度法"""
        return ecg_signal[r_peaks]
    
    def _rri_edr(self, r_peaks: np.ndarray) -> np.ndarray:
        """RR间期法"""
        rr_intervals = np.diff(r_peaks)
        # 插值到与R峰数量相同
        return np.concatenate([[rr_intervals[0]], rr_intervals])
    
    def _area_edr(
        self, 
        ecg_signal: np.ndarray, 
        r_peaks: np.ndarray,
        window: int = 10
    ) -> np.ndarray:
        """QRS面积法"""
        areas = []
        for peak in r_peaks:
            start = max(0, peak - window)
            end = min(len(ecg_signal), peak + window)
            area = np.trapz(np.abs(ecg_signal[start:end]))
            areas.append(area)
        return np.array(areas)
    
    def interpolate_edr(
        self,
        edr: np.ndarray,
        r_peaks: np.ndarray,
        target_fs: float = 4.0,
        signal_length: int = None
    ) -> np.ndarray:
        """
        将EDR信号插值到均匀采样
        
        Args:
            edr: EDR值 (每个R峰一个值)
            r_peaks: R峰位置
            target_fs: 目标采样率 (Hz)
            signal_length: 原始信号长度
            
        Returns:
            均匀采样的EDR信号
        """
        from scipy.interpolate import interp1d
        
        # 时间点
        times = r_peaks / self.sampling_rate
        
        # 创建插值函数
        f = interp1d(times, edr, kind='cubic', fill_value='extrapolate')
        
        # 生成均匀时间点
        if signal_length:
            duration = signal_length / self.sampling_rate
        else:
            duration = times[-1]
        
        uniform_times = np.arange(0, duration, 1/target_fs)
        
        # 插值
        edr_uniform = f(uniform_times)
        
        return edr_uniform
