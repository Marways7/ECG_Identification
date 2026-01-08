"""
小波变换去噪模块
================

实现基于离散小波变换(DWT)的ECG信号去噪算法
采用自适应阈值选择策略，有效去除高频噪声和基线漂移

数学原理:
---------
1. 信号分解: x(t) = Σ c_j,k · ψ_j,k(t) + Σ d_j,k · φ_j,k(t)
   其中 ψ 为小波函数, φ 为尺度函数

2. 阈值处理: 对细节系数 d_j,k 应用软/硬阈值
   - 软阈值: d'_j,k = sign(d_j,k) · max(|d_j,k| - λ, 0)
   - 硬阈值: d'_j,k = d_j,k · I(|d_j,k| > λ)

3. 自适应阈值: λ = σ · √(2·log(N)) / √N  (Universal Threshold)
   其中 σ = MAD(d_1) / 0.6745 (使用MAD估计噪声标准差)
"""

import numpy as np
import pywt
from typing import Tuple, Optional, List
from loguru import logger


class WaveletDenoiser:
    """
    基于小波变换的ECG信号去噪器
    
    采用多分辨率分析(MRA)进行信号分解，
    通过自适应阈值处理去除噪声成分
    
    Attributes:
        wavelet: 小波基函数名称 (推荐: 'db4', 'sym8', 'coif5')
        level: 分解层数 (None表示自动计算最大层数)
        threshold_mode: 阈值模式 ('soft' 或 'hard')
        threshold_type: 阈值选择策略 ('universal', 'sure', 'minimax')
    """
    
    # ECG去噪常用小波基及其特性
    WAVELET_RECOMMENDATIONS = {
        'db4': '德布西小波4阶，与QRS波群形态匹配度高',
        'db6': '德布西小波6阶，更平滑的近似',
        'sym8': '对称小波8阶，减少相位失真',
        'coif5': 'Coiflet小波5阶，具有近似对称性',
        'bior3.5': '双正交小波，ECG基线漂移去除效果好'
    }
    
    def __init__(
        self,
        wavelet: str = 'db4',
        level: Optional[int] = None,
        threshold_mode: str = 'soft',
        threshold_type: str = 'universal'
    ):
        """
        初始化小波去噪器
        
        Args:
            wavelet: 小波基函数 (默认'db4'，对ECG的QRS波群有良好匹配)
            level: 分解层数 (默认None，自动计算)
            threshold_mode: 'soft'(软阈值) 或 'hard'(硬阈值)
            threshold_type: 阈值选择策略
        """
        self.wavelet = wavelet
        self.level = level
        self.threshold_mode = threshold_mode
        self.threshold_type = threshold_type
        
        # 验证参数
        if wavelet not in pywt.wavelist():
            raise ValueError(f"不支持的小波基: {wavelet}")
        if threshold_mode not in ['soft', 'hard']:
            raise ValueError("threshold_mode 必须是 'soft' 或 'hard'")
        if threshold_type not in ['universal', 'sure', 'minimax', 'sqtwolog']:
            raise ValueError("不支持的阈值类型")
            
        logger.info(f"初始化小波去噪器: wavelet={wavelet}, mode={threshold_mode}")
    
    def _calculate_optimal_level(self, signal_length: int) -> int:
        """
        计算最优分解层数
        
        基于信号长度和小波滤波器长度自动计算
        确保每层分解后有足够的系数进行阈值处理
        
        Args:
            signal_length: 信号长度
            
        Returns:
            最优分解层数
        """
        wavelet_obj = pywt.Wavelet(self.wavelet)
        max_level = pywt.dwt_max_level(signal_length, wavelet_obj.dec_len)
        
        # ECG信号通常分解4-8层效果最佳
        # 过多层数会丢失有用信息，过少则去噪不彻底
        optimal_level = min(max_level, 8)
        optimal_level = max(optimal_level, 4)
        
        return optimal_level
    
    def _estimate_noise_sigma(self, detail_coeffs: np.ndarray) -> float:
        """
        使用中位数绝对偏差(MAD)估计噪声标准差
        
        MAD估计器对异常值具有鲁棒性，适合ECG信号
        σ = MAD(d) / 0.6745
        
        其中0.6745是正态分布MAD与标准差的换算系数
        
        Args:
            detail_coeffs: 最高频细节系数 (通常为第一层)
            
        Returns:
            估计的噪声标准差
        """
        mad = np.median(np.abs(detail_coeffs - np.median(detail_coeffs)))
        sigma = mad / 0.6745
        return max(sigma, 1e-10)  # 防止除零
    
    def _calculate_threshold(
        self, 
        coeffs: np.ndarray, 
        sigma: float,
        n_samples: int
    ) -> float:
        """
        计算自适应阈值
        
        实现多种阈值选择策略:
        1. Universal (VisuShrink): λ = σ√(2·log(N))
        2. SURE (SureShrink): 最小化Stein无偏风险估计
        3. Minimax: 最小化最大均方误差
        
        Args:
            coeffs: 小波系数
            sigma: 噪声标准差估计
            n_samples: 原始信号样本数
            
        Returns:
            计算得到的阈值
        """
        if self.threshold_type == 'universal':
            # Universal threshold (Donoho & Johnstone, 1994)
            threshold = sigma * np.sqrt(2 * np.log(n_samples))
            
        elif self.threshold_type == 'sqtwolog':
            # sqrt(2*log(N)) 变体
            threshold = sigma * np.sqrt(2 * np.log(len(coeffs)))
            
        elif self.threshold_type == 'minimax':
            # Minimax threshold
            if n_samples <= 32:
                threshold = 0
            else:
                threshold = sigma * (0.3936 + 0.1829 * np.log(n_samples) / np.log(2))
                
        elif self.threshold_type == 'sure':
            # SURE (Stein's Unbiased Risk Estimate) threshold
            threshold = self._sure_threshold(coeffs, sigma)
        else:
            threshold = sigma * np.sqrt(2 * np.log(n_samples))
            
        return threshold
    
    def _sure_threshold(self, coeffs: np.ndarray, sigma: float) -> float:
        """
        计算SURE最优阈值
        
        SURE方法通过最小化Stein无偏风险估计来选择阈值
        适合信号稀疏度未知的情况
        
        Args:
            coeffs: 小波系数
            sigma: 噪声标准差
            
        Returns:
            SURE最优阈值
        """
        n = len(coeffs)
        coeffs_normalized = coeffs / sigma
        coeffs_sorted = np.sort(np.abs(coeffs_normalized)) ** 2
        
        # 计算累积和
        cumsum = np.cumsum(coeffs_sorted)
        
        # SURE风险估计
        risks = (n - 2 * np.arange(1, n + 1) + 
                 cumsum + 
                 np.arange(n, 0, -1) * coeffs_sorted)
        
        # 找到最小风险对应的阈值
        best_idx = np.argmin(risks)
        threshold = sigma * np.sqrt(coeffs_sorted[best_idx])
        
        # 与universal阈值比较，取较小值
        universal = sigma * np.sqrt(2 * np.log(n))
        
        return min(threshold, universal)
    
    def _apply_threshold(
        self, 
        coeffs: np.ndarray, 
        threshold: float
    ) -> np.ndarray:
        """
        应用阈值处理
        
        软阈值(Soft): 缩减系数幅值，保持信号平滑
        硬阈值(Hard): 直接置零，保持系数幅值
        
        Args:
            coeffs: 小波系数
            threshold: 阈值
            
        Returns:
            处理后的系数
        """
        if self.threshold_mode == 'soft':
            # 软阈值: sign(x) * max(|x| - λ, 0)
            return pywt.threshold(coeffs, threshold, mode='soft')
        else:
            # 硬阈值: x * I(|x| > λ)
            return pywt.threshold(coeffs, threshold, mode='hard')
    
    def denoise(
        self, 
        signal: np.ndarray,
        return_details: bool = False
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        执行小波去噪
        
        完整流程:
        1. DWT分解信号为近似系数和细节系数
        2. 估计噪声水平
        3. 对每层细节系数应用自适应阈值
        4. IDWT重构去噪信号
        
        Args:
            signal: 输入ECG信号 (1D numpy数组)
            return_details: 是否返回分解细节
            
        Returns:
            denoised_signal: 去噪后的信号
            details: 分解细节字典 (可选)
        """
        signal = np.asarray(signal, dtype=np.float64)
        n_samples = len(signal)
        
        # 确定分解层数
        level = self.level if self.level else self._calculate_optimal_level(n_samples)
        
        logger.debug(f"小波分解: {level}层, 信号长度={n_samples}")
        
        # 执行多层小波分解
        # coeffs = [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        coeffs = pywt.wavedec(signal, self.wavelet, level=level)
        
        # 使用最高频细节系数估计噪声
        sigma = self._estimate_noise_sigma(coeffs[-1])
        logger.debug(f"估计噪声标准差: σ = {sigma:.4f}")
        
        # 对每层细节系数应用阈值
        denoised_coeffs = [coeffs[0]]  # 保留近似系数
        thresholds = []
        
        for i, detail in enumerate(coeffs[1:], 1):
            # 每层使用不同的阈值 (层级越低，阈值越大)
            threshold = self._calculate_threshold(detail, sigma, n_samples)
            thresholds.append(threshold)
            
            # 应用阈值处理
            denoised_detail = self._apply_threshold(detail, threshold)
            denoised_coeffs.append(denoised_detail)
            
            logger.debug(f"层{i}: 阈值={threshold:.4f}, "
                        f"系数数={len(detail)}, "
                        f"置零率={np.mean(np.abs(denoised_detail) < 1e-10)*100:.1f}%")
        
        # 重构信号
        denoised_signal = pywt.waverec(denoised_coeffs, self.wavelet)
        
        # 确保长度一致 (小波重构可能产生微小长度差异)
        denoised_signal = denoised_signal[:n_samples]
        
        if return_details:
            details = {
                'level': level,
                'wavelet': self.wavelet,
                'sigma': sigma,
                'thresholds': thresholds,
                'original_coeffs': coeffs,
                'denoised_coeffs': denoised_coeffs,
                'snr_improvement': self._estimate_snr_improvement(signal, denoised_signal)
            }
            return denoised_signal, details
        
        return denoised_signal, None
    
    def _estimate_snr_improvement(
        self, 
        original: np.ndarray, 
        denoised: np.ndarray
    ) -> float:
        """
        估计信噪比改善量
        
        通过比较去噪前后的信号方差变化来估计SNR改善
        
        Args:
            original: 原始信号
            denoised: 去噪信号
            
        Returns:
            SNR改善量 (dB)
        """
        noise_estimate = original - denoised
        signal_power = np.var(denoised)
        noise_power = np.var(noise_estimate)
        
        if noise_power > 0:
            snr_improvement = 10 * np.log10(signal_power / noise_power)
        else:
            snr_improvement = float('inf')
            
        return snr_improvement
    
    def multi_scale_denoise(
        self, 
        signal: np.ndarray,
        wavelets: List[str] = ['db4', 'sym8', 'coif5']
    ) -> np.ndarray:
        """
        多尺度融合去噪
        
        使用多种小波基分别去噪，然后融合结果
        可以获得更鲁棒的去噪效果
        
        Args:
            signal: 输入信号
            wavelets: 小波基列表
            
        Returns:
            融合后的去噪信号
        """
        denoised_signals = []
        
        original_wavelet = self.wavelet
        
        for wavelet in wavelets:
            self.wavelet = wavelet
            denoised, _ = self.denoise(signal)
            denoised_signals.append(denoised)
        
        self.wavelet = original_wavelet
        
        # 中值融合 (对异常值鲁棒)
        fused_signal = np.median(denoised_signals, axis=0)
        
        return fused_signal


class AdaptiveWaveletDenoiser(WaveletDenoiser):
    """
    自适应小波去噪器
    
    根据信号特性自动选择最优小波基和参数
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def auto_select_wavelet(self, signal: np.ndarray) -> str:
        """
        自动选择最优小波基
        
        通过计算不同小波基的重构误差和能量集中度
        选择最适合当前信号的小波基
        
        Args:
            signal: 输入信号
            
        Returns:
            最优小波基名称
        """
        candidates = ['db4', 'db6', 'sym8', 'coif5']
        best_wavelet = 'db4'
        best_score = float('inf')
        
        for wavelet in candidates:
            try:
                # 计算重构质量得分
                coeffs = pywt.wavedec(signal, wavelet, level=5)
                reconstructed = pywt.waverec(coeffs, wavelet)[:len(signal)]
                
                # 重构误差
                recon_error = np.mean((signal - reconstructed) ** 2)
                
                # 能量集中度 (越集中越好)
                energy_concentration = np.sum(coeffs[0] ** 2) / np.sum(signal ** 2)
                
                # 综合得分
                score = recon_error / (energy_concentration + 1e-10)
                
                if score < best_score:
                    best_score = score
                    best_wavelet = wavelet
                    
            except Exception as e:
                logger.warning(f"小波{wavelet}评估失败: {e}")
                continue
        
        logger.info(f"自动选择小波基: {best_wavelet}")
        return best_wavelet
    
    def denoise(
        self, 
        signal: np.ndarray,
        auto_wavelet: bool = True,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        执行自适应去噪
        
        Args:
            signal: 输入信号
            auto_wavelet: 是否自动选择小波基
            **kwargs: 传递给父类denoise的参数
            
        Returns:
            去噪信号和详细信息
        """
        if auto_wavelet:
            self.wavelet = self.auto_select_wavelet(signal)
            
        return super().denoise(signal, **kwargs)
