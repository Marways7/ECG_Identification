"""
基线漂移校正模块
================

实现多种基线漂移去除算法:
1. 形态学滤波 (Morphological Filtering)
2. 多项式拟合 (Polynomial Fitting)
3. 中值滤波 (Median Filtering)
4. 小波分解法 (Wavelet-based)

数学原理:
---------
基线漂移主要由呼吸运动、电极运动等低频成分引起
频率通常在0.05-0.5Hz范围内

形态学方法:
baseline = Opening(Closing(x, se), se)
其中 se 为结构元素，长度约为200ms (覆盖QRS波群)
"""

import numpy as np
from scipy import signal, ndimage
from scipy.interpolate import UnivariateSpline
from typing import Tuple, Optional, Literal
import pywt
from loguru import logger


class BaselineCorrector:
    """
    ECG基线漂移校正器
    
    支持多种校正方法，可根据信号特性选择最优方法
    
    Attributes:
        method: 校正方法
        sampling_rate: 采样率 (Hz)
    """
    
    METHODS = ['morphological', 'polynomial', 'median', 'wavelet', 'highpass', 'spline']
    
    def __init__(
        self,
        method: str = 'morphological',
        sampling_rate: float = 200.0
    ):
        """
        初始化基线校正器
        
        Args:
            method: 校正方法 ('morphological', 'polynomial', 'median', 
                    'wavelet', 'highpass', 'spline')
            sampling_rate: 采样率
        """
        if method not in self.METHODS:
            raise ValueError(f"不支持的方法: {method}. 可选: {self.METHODS}")
            
        self.method = method
        self.sampling_rate = sampling_rate
        
        logger.info(f"初始化基线校正器: method={method}, fs={sampling_rate}Hz")
    
    def correct(
        self, 
        signal_data: np.ndarray,
        return_baseline: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        执行基线校正
        
        Args:
            signal_data: 输入ECG信号
            return_baseline: 是否返回估计的基线
            
        Returns:
            corrected_signal: 校正后的信号
            baseline: 估计的基线 (可选)
        """
        signal_data = np.asarray(signal_data, dtype=np.float64)
        
        method_map = {
            'morphological': self._morphological_correction,
            'polynomial': self._polynomial_correction,
            'median': self._median_correction,
            'wavelet': self._wavelet_correction,
            'highpass': self._highpass_correction,
            'spline': self._spline_correction
        }
        
        corrector = method_map[self.method]
        corrected, baseline = corrector(signal_data)
        
        if return_baseline:
            return corrected, baseline
        return corrected, None
    
    def _morphological_correction(
        self, 
        signal_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        形态学滤波基线校正
        
        使用开运算和闭运算的组合来估计基线
        结构元素长度应覆盖QRS波群但小于T波周期
        
        算法:
        1. 闭运算填充R峰上方区域
        2. 开运算去除S波下方区域
        3. 结果即为基线估计
        
        Args:
            signal_data: 输入信号
            
        Returns:
            校正后信号和基线
        """
        # 结构元素长度: 约200ms (覆盖QRS波群)
        se_length = int(0.2 * self.sampling_rate)
        if se_length % 2 == 0:
            se_length += 1  # 确保为奇数
        
        # 创建平坦结构元素
        structure = np.ones(se_length)
        
        # 闭运算 -> 开运算
        closed = ndimage.grey_closing(signal_data, size=se_length)
        baseline = ndimage.grey_opening(closed, size=se_length)
        
        # 可选: 对基线进行平滑
        baseline = ndimage.uniform_filter1d(baseline, size=se_length // 2)
        
        corrected = signal_data - baseline
        
        logger.debug(f"形态学校正: se_length={se_length}, "
                    f"baseline_range=[{baseline.min():.2f}, {baseline.max():.2f}]")
        
        return corrected, baseline
    
    def _polynomial_correction(
        self, 
        signal_data: np.ndarray,
        order: int = 6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        多项式拟合基线校正
        
        使用低阶多项式拟合信号的低频趋势
        
        Args:
            signal_data: 输入信号
            order: 多项式阶数
            
        Returns:
            校正后信号和基线
        """
        n = len(signal_data)
        x = np.arange(n)
        
        # 多项式拟合
        coeffs = np.polyfit(x, signal_data, order)
        baseline = np.polyval(coeffs, x)
        
        corrected = signal_data - baseline
        
        return corrected, baseline
    
    def _median_correction(
        self, 
        signal_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        中值滤波基线校正
        
        两阶段中值滤波:
        1. 短窗口去除QRS波群: 200ms
        2. 长窗口平滑基线: 600ms
        
        Args:
            signal_data: 输入信号
            
        Returns:
            校正后信号和基线
        """
        # 第一阶段: 去除QRS (200ms窗口)
        window1 = int(0.2 * self.sampling_rate)
        if window1 % 2 == 0:
            window1 += 1
        median1 = signal.medfilt(signal_data, kernel_size=window1)
        
        # 第二阶段: 平滑 (600ms窗口)
        window2 = int(0.6 * self.sampling_rate)
        if window2 % 2 == 0:
            window2 += 1
        baseline = signal.medfilt(median1, kernel_size=window2)
        
        corrected = signal_data - baseline
        
        return corrected, baseline
    
    def _wavelet_correction(
        self, 
        signal_data: np.ndarray,
        wavelet: str = 'db4',
        level: int = 9
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        小波分解基线校正
        
        将信号分解到高层近似系数，该系数代表低频基线成分
        
        基线漂移频率: 0.05-0.5Hz
        在fs=200Hz时，9层分解的最低频带约为: fs/(2^9) ≈ 0.39Hz
        
        Args:
            signal_data: 输入信号
            wavelet: 小波基
            level: 分解层数
            
        Returns:
            校正后信号和基线
        """
        # 计算最大分解层数
        max_level = pywt.dwt_max_level(len(signal_data), pywt.Wavelet(wavelet).dec_len)
        level = min(level, max_level)
        
        # 小波分解
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        
        # 将高层近似系数置零以去除基线
        # 保留所有细节系数，重构得到去基线信号
        baseline_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        detail_coeffs = [np.zeros_like(coeffs[0])] + coeffs[1:]
        
        # 重构基线
        baseline = pywt.waverec(baseline_coeffs, wavelet)[:len(signal_data)]
        
        # 重构去基线信号
        corrected = signal_data - baseline
        
        return corrected, baseline
    
    def _highpass_correction(
        self, 
        signal_data: np.ndarray,
        cutoff: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        高通滤波基线校正
        
        使用Butterworth高通滤波器去除低于cutoff频率的成分
        
        Args:
            signal_data: 输入信号
            cutoff: 截止频率 (Hz)
            
        Returns:
            校正后信号和基线
        """
        # 设计高通滤波器
        nyquist = self.sampling_rate / 2
        normalized_cutoff = cutoff / nyquist
        
        # 确保截止频率有效
        if normalized_cutoff >= 1:
            normalized_cutoff = 0.99
        if normalized_cutoff <= 0:
            normalized_cutoff = 0.01
            
        b, a = signal.butter(4, normalized_cutoff, btype='highpass')
        
        # 零相位滤波 (避免相位失真)
        corrected = signal.filtfilt(b, a, signal_data)
        
        # 基线 = 原信号 - 高通信号
        baseline = signal_data - corrected
        
        return corrected, baseline
    
    def _spline_correction(
        self, 
        signal_data: np.ndarray,
        knot_spacing: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        三次样条基线校正
        
        在检测到的R峰之间使用样条插值估计基线
        
        Args:
            signal_data: 输入信号
            knot_spacing: 节点间距 (秒)
            
        Returns:
            校正后信号和基线
        """
        n = len(signal_data)
        x = np.arange(n)
        
        # 选择均匀分布的节点
        knot_interval = int(knot_spacing * self.sampling_rate)
        knot_indices = np.arange(0, n, knot_interval)
        
        # 在每个节点使用局部中值作为基线值
        window = int(0.1 * self.sampling_rate)
        knot_values = []
        for idx in knot_indices:
            start = max(0, idx - window)
            end = min(n, idx + window)
            knot_values.append(np.median(signal_data[start:end]))
        
        knot_values = np.array(knot_values)
        
        # 样条插值
        try:
            spline = UnivariateSpline(knot_indices, knot_values, s=len(knot_indices))
            baseline = spline(x)
        except Exception as e:
            logger.warning(f"样条插值失败: {e}，使用线性插值")
            baseline = np.interp(x, knot_indices, knot_values)
        
        corrected = signal_data - baseline
        
        return corrected, baseline
    
    def adaptive_correct(
        self, 
        signal_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        自适应基线校正
        
        根据信号特性自动选择最佳校正方法
        
        Args:
            signal_data: 输入信号
            
        Returns:
            校正后信号和基线
        """
        # 评估信号特性
        # 计算低频能量占比
        freqs = np.fft.rfftfreq(len(signal_data), 1/self.sampling_rate)
        fft_mag = np.abs(np.fft.rfft(signal_data))
        
        low_freq_mask = freqs < 0.5
        low_freq_energy = np.sum(fft_mag[low_freq_mask] ** 2)
        total_energy = np.sum(fft_mag ** 2)
        low_freq_ratio = low_freq_energy / (total_energy + 1e-10)
        
        logger.debug(f"低频能量占比: {low_freq_ratio:.2%}")
        
        # 根据低频能量选择方法
        if low_freq_ratio > 0.5:
            # 严重基线漂移，使用形态学方法
            self.method = 'morphological'
        elif low_freq_ratio > 0.2:
            # 中等基线漂移，使用小波方法
            self.method = 'wavelet'
        else:
            # 轻微基线漂移，使用高通滤波
            self.method = 'highpass'
        
        logger.info(f"自适应选择方法: {self.method}")
        
        return self.correct(signal_data, return_baseline=True)


class MultiStageBaselineCorrector:
    """
    多阶段基线校正器
    
    组合多种方法进行递进式基线校正
    """
    
    def __init__(self, sampling_rate: float = 200.0):
        self.sampling_rate = sampling_rate
    
    def correct(
        self, 
        signal_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        多阶段基线校正
        
        阶段1: 高通滤波去除极低频漂移 (<0.1Hz)
        阶段2: 形态学滤波去除呼吸相关漂移
        
        Args:
            signal_data: 输入信号
            
        Returns:
            校正后信号和总基线
        """
        total_baseline = np.zeros_like(signal_data)
        current_signal = signal_data.copy()
        
        # 阶段1: 高通滤波 (0.1Hz)
        corrector1 = BaselineCorrector('highpass', self.sampling_rate)
        corrected1, baseline1 = corrector1._highpass_correction(current_signal, cutoff=0.1)
        total_baseline += baseline1
        current_signal = corrected1
        
        # 阶段2: 形态学滤波
        corrector2 = BaselineCorrector('morphological', self.sampling_rate)
        corrected2, baseline2 = corrector2._morphological_correction(current_signal)
        total_baseline += baseline2
        
        return corrected2, total_baseline
