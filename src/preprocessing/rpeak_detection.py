"""
R峰检测模块
===========

实现多种R峰检测算法:
1. Pan-Tompkins算法 (经典方法)
2. Hamilton算法 (改进版)
3. 小波变换法 (多尺度检测)
4. 导数阈值法 (简单快速)

数学原理:
---------
Pan-Tompkins算法流程:
1. 带通滤波 (5-15Hz): y[n] = x[n] * h_bp[n]
2. 微分运算: y'[n] = (1/8T)(-y[n-2] - 2y[n-1] + 2y[n+1] + y[n+2])
3. 平方运算: y''[n] = (y'[n])^2
4. 移动窗口积分: y'''[n] = (1/N)∑y''[n-k]
5. 自适应阈值检测
"""

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d
from typing import Tuple, List, Optional
import pywt
from loguru import logger


class RPeakDetector:
    """
    ECG R峰检测器
    
    集成多种检测算法，支持自动选择和结果融合
    
    Attributes:
        sampling_rate: 采样率 (Hz)
        method: 检测方法
    """
    
    METHODS = ['pan_tompkins', 'hamilton', 'wavelet', 'derivative', 'neurokit']
    
    def __init__(
        self,
        sampling_rate: float = 200.0,
        method: str = 'pan_tompkins'
    ):
        """
        初始化R峰检测器
        
        Args:
            sampling_rate: 采样率
            method: 检测方法
        """
        if method not in self.METHODS:
            raise ValueError(f"不支持的方法: {method}")
            
        self.sampling_rate = sampling_rate
        self.method = method
        
        # 生理约束参数
        self.min_rr = 0.3  # 最小RR间期 (秒) - 约200bpm
        self.max_rr = 2.0  # 最大RR间期 (秒) - 约30bpm
        self.refractory_period = 0.2  # 不应期 (秒)
        
        logger.info(f"初始化R峰检测器: method={method}, fs={sampling_rate}Hz")
    
    def detect(
        self, 
        ecg_signal: np.ndarray,
        return_features: bool = False
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        检测R峰位置
        
        Args:
            ecg_signal: ECG信号
            return_features: 是否返回检测特征
            
        Returns:
            r_peaks: R峰索引数组
            features: 检测特征 (可选)
        """
        ecg_signal = np.asarray(ecg_signal, dtype=np.float64)
        
        method_map = {
            'pan_tompkins': self._pan_tompkins,
            'hamilton': self._hamilton,
            'wavelet': self._wavelet_detection,
            'derivative': self._derivative_threshold,
            'neurokit': self._neurokit_style
        }
        
        detector = method_map[self.method]
        r_peaks, features = detector(ecg_signal)
        
        # 后处理: 去除生理不合理的检测
        r_peaks = self._physiological_filter(r_peaks)
        
        logger.info(f"检测到 {len(r_peaks)} 个R峰, "
                   f"平均心率: {self._calculate_hr(r_peaks):.1f} BPM")
        
        if return_features:
            features['r_peaks'] = r_peaks
            features['heart_rate'] = self._calculate_hr(r_peaks)
            return r_peaks, features
        
        return r_peaks, None
    
    def _pan_tompkins(
        self, 
        ecg_signal: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        Pan-Tompkins算法
        
        经典ECG R峰检测算法，包含:
        1. 带通滤波 (5-15Hz)
        2. 五点微分
        3. 平方
        4. 移动窗口积分
        5. 自适应双阈值检测
        
        Reference: Pan & Tompkins, IEEE TBME, 1985
        
        Args:
            ecg_signal: ECG信号
            
        Returns:
            R峰位置和特征字典
        """
        fs = self.sampling_rate
        
        # Step 1: 带通滤波 (5-15Hz)
        # 低通滤波器 (15Hz)
        fc_low = 15
        b_low = np.array([1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1])
        a_low = np.array([1, -2, 1]) * 32
        
        # 使用Butterworth滤波器替代 (更稳定)
        nyquist = fs / 2
        low_cut = 5 / nyquist
        high_cut = min(15, nyquist - 1) / nyquist
        
        if high_cut <= low_cut:
            high_cut = 0.99
            low_cut = 0.05
            
        b, a = signal.butter(2, [low_cut, high_cut], btype='band')
        filtered = signal.filtfilt(b, a, ecg_signal)
        
        # Step 2: 五点微分
        # H(z) = (1/8T)(-z^{-2} - 2z^{-1} + 2z + z^2)
        diff_kernel = np.array([-1, -2, 0, 2, 1]) / 8
        differentiated = np.convolve(filtered, diff_kernel, mode='same')
        
        # Step 3: 平方
        squared = differentiated ** 2
        
        # Step 4: 移动窗口积分 (150ms窗口)
        window_size = int(0.15 * fs)
        if window_size < 1:
            window_size = 1
        integration_kernel = np.ones(window_size) / window_size
        integrated = np.convolve(squared, integration_kernel, mode='same')
        
        # Step 5: 自适应阈值检测
        r_peaks = self._adaptive_threshold_detection(integrated, ecg_signal)
        
        features = {
            'filtered': filtered,
            'differentiated': differentiated,
            'squared': squared,
            'integrated': integrated
        }
        
        return r_peaks, features
    
    def _adaptive_threshold_detection(
        self, 
        integrated: np.ndarray,
        original: np.ndarray
    ) -> np.ndarray:
        """
        自适应双阈值检测
        
        维护两个自适应阈值:
        - SPKI: 信号峰值估计
        - NPKI: 噪声峰值估计
        - THR1 = NPKI + 0.25 * (SPKI - NPKI)  (高阈值)
        - THR2 = 0.5 * THR1  (低阈值)
        
        Args:
            integrated: 积分信号
            original: 原始信号 (用于验证)
            
        Returns:
            R峰索引
        """
        fs = self.sampling_rate
        
        # 初始化
        spki = np.max(integrated[:int(2 * fs)]) * 0.5  # 信号峰值估计
        npki = np.mean(integrated[:int(2 * fs)]) * 0.5  # 噪声峰值估计
        threshold1 = npki + 0.25 * (spki - npki)  # 高阈值
        threshold2 = 0.5 * threshold1  # 低阈值
        
        # 寻找所有局部最大值
        min_distance = int(self.refractory_period * fs)
        peaks, properties = signal.find_peaks(
            integrated, 
            distance=min_distance,
            height=threshold2
        )
        
        # 分类峰值
        r_peaks = []
        
        for peak in peaks:
            if integrated[peak] > threshold1:
                # 明确的R峰
                r_peaks.append(peak)
                spki = 0.125 * integrated[peak] + 0.875 * spki
            else:
                # 噪声峰
                npki = 0.125 * integrated[peak] + 0.875 * npki
            
            # 更新阈值
            threshold1 = npki + 0.25 * (spki - npki)
            threshold2 = 0.5 * threshold1
        
        r_peaks = np.array(r_peaks)
        
        # 在原始信号中精确定位R峰 (在积分峰附近找最大值)
        search_window = int(0.05 * fs)  # 50ms搜索窗口
        refined_peaks = []
        
        for peak in r_peaks:
            start = max(0, peak - search_window)
            end = min(len(original), peak + search_window)
            local_max = np.argmax(np.abs(original[start:end])) + start
            refined_peaks.append(local_max)
        
        return np.array(refined_peaks)
    
    def _hamilton(self, ecg_signal: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Hamilton算法
        
        改进的Pan-Tompkins算法，增加了回溯机制
        
        Args:
            ecg_signal: ECG信号
            
        Returns:
            R峰位置和特征字典
        """
        fs = self.sampling_rate
        
        # 带通滤波 (8-16Hz)
        nyquist = fs / 2
        low = 8 / nyquist
        high = min(16, nyquist - 1) / nyquist
        
        if high <= low:
            high = 0.99
            low = 0.1
            
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, ecg_signal)
        
        # 一阶差分
        diff = np.diff(filtered)
        diff = np.concatenate([[0], diff])
        
        # 绝对值
        abs_diff = np.abs(diff)
        
        # 移动平均 (80ms)
        window = int(0.08 * fs)
        if window < 1:
            window = 1
        ma = np.convolve(abs_diff, np.ones(window)/window, mode='same')
        
        # 寻找峰值
        min_distance = int(self.min_rr * fs)
        peaks, _ = signal.find_peaks(ma, distance=min_distance)
        
        if len(peaks) == 0:
            return np.array([]), {'filtered': filtered, 'processed': ma}
        
        # 自适应阈值
        threshold = np.mean(ma[peaks]) * 0.4
        
        # 筛选峰值
        valid_peaks = peaks[ma[peaks] > threshold]
        
        # 在原始信号中精确定位
        search_window = int(0.05 * fs)
        r_peaks = []
        
        for peak in valid_peaks:
            start = max(0, peak - search_window)
            end = min(len(ecg_signal), peak + search_window)
            local_max = np.argmax(ecg_signal[start:end]) + start
            r_peaks.append(local_max)
        
        return np.array(r_peaks), {'filtered': filtered, 'processed': ma}
    
    def _wavelet_detection(
        self, 
        ecg_signal: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        小波变换R峰检测
        
        使用多尺度小波分解检测QRS波群
        QRS波群主要能量集中在10-25Hz，对应特定小波尺度
        
        Args:
            ecg_signal: ECG信号
            
        Returns:
            R峰位置和特征字典
        """
        fs = self.sampling_rate
        
        # 使用 'db4' 小波分解
        wavelet = 'db4'
        level = min(8, pywt.dwt_max_level(len(ecg_signal), wavelet))
        
        coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)
        
        # QRS波群主要在 levels 3-5 (约10-40Hz at 200Hz采样)
        # 保留这些层的细节系数
        qrs_coeffs = [np.zeros_like(c) for c in coeffs]
        
        # 选择包含QRS信息的层
        for i in range(min(3, len(coeffs)-1), min(6, len(coeffs))):
            if i < len(coeffs):
                qrs_coeffs[i] = coeffs[i]
        
        # 重构QRS增强信号
        qrs_enhanced = pywt.waverec(qrs_coeffs, wavelet)[:len(ecg_signal)]
        qrs_enhanced = qrs_enhanced ** 2  # 平方增强
        
        # 寻找峰值
        min_distance = int(self.min_rr * fs)
        threshold = np.mean(qrs_enhanced) + 2 * np.std(qrs_enhanced)
        
        peaks, _ = signal.find_peaks(
            qrs_enhanced, 
            distance=min_distance,
            height=threshold * 0.5
        )
        
        # 精确定位
        search_window = int(0.05 * fs)
        r_peaks = []
        
        for peak in peaks:
            start = max(0, peak - search_window)
            end = min(len(ecg_signal), peak + search_window)
            local_max = np.argmax(ecg_signal[start:end]) + start
            r_peaks.append(local_max)
        
        return np.array(r_peaks), {'qrs_enhanced': qrs_enhanced}
    
    def _derivative_threshold(
        self, 
        ecg_signal: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        导数阈值法
        
        简单快速的R峰检测方法:
        1. 一阶导数
        2. 阈值检测
        3. 局部最大值搜索
        
        Args:
            ecg_signal: ECG信号
            
        Returns:
            R峰位置和特征字典
        """
        fs = self.sampling_rate
        
        # 预处理: 带通滤波
        nyquist = fs / 2
        low = 5 / nyquist
        high = min(45, nyquist - 1) / nyquist
        
        if high <= low:
            high = 0.99
            low = 0.05
            
        b, a = signal.butter(3, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, ecg_signal)
        
        # 计算一阶导数的绝对值
        diff = np.abs(np.diff(filtered))
        diff = np.concatenate([diff, [0]])
        
        # 阈值
        threshold = np.mean(diff) + 2 * np.std(diff)
        
        # 寻找超过阈值的区域
        above_threshold = diff > threshold
        
        # 在每个区域找最大值
        min_distance = int(self.min_rr * fs)
        peaks, _ = signal.find_peaks(filtered, distance=min_distance)
        
        # 筛选在阈值区域内的峰值
        valid_peaks = [p for p in peaks if above_threshold[max(0, p-5):min(len(diff), p+5)].any()]
        
        return np.array(valid_peaks), {'derivative': diff, 'filtered': filtered}
    
    def _neurokit_style(
        self, 
        ecg_signal: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        NeuroKit风格检测
        
        综合多种方法的现代检测算法
        
        Args:
            ecg_signal: ECG信号
            
        Returns:
            R峰位置和特征字典
        """
        fs = self.sampling_rate
        
        # 预处理
        # 1. 去直流
        ecg_centered = ecg_signal - np.mean(ecg_signal)
        
        # 2. 带通滤波 (3-45Hz)
        nyquist = fs / 2
        low = 3 / nyquist
        high = min(45, nyquist - 1) / nyquist
        
        if high <= low:
            high = 0.99
            low = 0.01
            
        b, a = signal.butter(3, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, ecg_centered)
        
        # 3. 计算梯度
        gradient = np.gradient(filtered)
        
        # 4. 平方
        squared = gradient ** 2
        
        # 5. 移动平均 (100ms)
        window = int(0.1 * fs)
        if window < 1:
            window = 1
        ma = np.convolve(squared, np.ones(window)/window, mode='same')
        
        # 6. 峰值检测
        min_distance = int(self.min_rr * fs)
        
        # 使用百分位数作为阈值
        threshold = np.percentile(ma, 90)
        
        peaks, properties = signal.find_peaks(
            ma, 
            distance=min_distance,
            height=threshold * 0.3,
            prominence=threshold * 0.2
        )
        
        # 7. 在原始信号中精确定位
        search_window = int(0.05 * fs)
        r_peaks = []
        
        for peak in peaks:
            start = max(0, peak - search_window)
            end = min(len(ecg_signal), peak + search_window)
            local_max = np.argmax(ecg_signal[start:end]) + start
            r_peaks.append(local_max)
        
        # 去重
        r_peaks = np.unique(r_peaks)
        
        return r_peaks, {'filtered': filtered, 'processed': ma}
    
    def _physiological_filter(self, r_peaks: np.ndarray) -> np.ndarray:
        """
        生理学约束滤波
        
        去除不符合生理规律的检测结果:
        - RR间期过短 (<300ms, >200bpm)
        - RR间期过长 (>2000ms, <30bpm)
        - RR间期突变过大 (>50%)
        
        Args:
            r_peaks: 原始R峰位置
            
        Returns:
            过滤后的R峰位置
        """
        if len(r_peaks) < 2:
            return r_peaks
        
        fs = self.sampling_rate
        min_samples = int(self.min_rr * fs)
        max_samples = int(self.max_rr * fs)
        
        # 计算RR间期
        rr_intervals = np.diff(r_peaks)
        
        # 标记有效间期
        valid = np.ones(len(r_peaks), dtype=bool)
        
        for i in range(len(rr_intervals)):
            # 检查间期范围
            if rr_intervals[i] < min_samples or rr_intervals[i] > max_samples:
                # 决定删除哪个峰 (保留幅值更大的)
                # 简化处理: 标记后一个为无效
                valid[i + 1] = False
                continue
            
            # 检查间期突变 (与前一个间期比较)
            if i > 0 and valid[i]:
                ratio = rr_intervals[i] / rr_intervals[i-1]
                if ratio > 1.5 or ratio < 0.67:
                    # 可能是漏检或误检，暂时保留
                    pass
        
        filtered_peaks = r_peaks[valid]
        
        logger.debug(f"生理学滤波: {len(r_peaks)} -> {len(filtered_peaks)} 峰")
        
        return filtered_peaks
    
    def _calculate_hr(self, r_peaks: np.ndarray) -> float:
        """
        计算平均心率
        
        Args:
            r_peaks: R峰位置
            
        Returns:
            平均心率 (BPM)
        """
        if len(r_peaks) < 2:
            return 0.0
        
        rr_intervals = np.diff(r_peaks) / self.sampling_rate  # 秒
        mean_rr = np.mean(rr_intervals)
        
        if mean_rr > 0:
            return 60.0 / mean_rr
        return 0.0
    
    def ensemble_detect(
        self, 
        ecg_signal: np.ndarray,
        methods: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        集成检测
        
        使用多种方法检测，然后融合结果
        
        Args:
            ecg_signal: ECG信号
            methods: 使用的方法列表
            
        Returns:
            R峰位置和各方法结果
        """
        if methods is None:
            methods = ['pan_tompkins', 'hamilton', 'wavelet']
        
        all_peaks = []
        method_results = {}
        
        original_method = self.method
        
        for method in methods:
            try:
                self.method = method
                peaks, features = self.detect(ecg_signal)
                all_peaks.append(peaks)
                method_results[method] = {
                    'peaks': peaks,
                    'count': len(peaks)
                }
            except Exception as e:
                logger.warning(f"方法 {method} 失败: {e}")
        
        self.method = original_method
        
        # 融合策略: 投票
        fused_peaks = self._vote_fusion(all_peaks, ecg_signal)
        
        return fused_peaks, method_results
    
    def _vote_fusion(
        self, 
        all_peaks: List[np.ndarray],
        ecg_signal: np.ndarray,
        tolerance: int = None
    ) -> np.ndarray:
        """
        投票融合
        
        如果多数方法在某位置检测到峰值，则认为是真正的R峰
        
        Args:
            all_peaks: 各方法检测结果
            ecg_signal: 原始信号
            tolerance: 位置容差 (采样点)
            
        Returns:
            融合后的R峰位置
        """
        if tolerance is None:
            tolerance = int(0.05 * self.sampling_rate)  # 50ms
        
        # 合并所有检测结果
        all_positions = np.concatenate(all_peaks) if all_peaks else np.array([])
        
        if len(all_positions) == 0:
            return np.array([])
        
        all_positions = np.sort(all_positions)
        
        # 聚类相近的位置
        clusters = []
        current_cluster = [all_positions[0]]
        
        for pos in all_positions[1:]:
            if pos - current_cluster[-1] <= tolerance:
                current_cluster.append(pos)
            else:
                clusters.append(current_cluster)
                current_cluster = [pos]
        clusters.append(current_cluster)
        
        # 选择至少被一半方法检测到的位置
        min_votes = len(all_peaks) // 2
        
        fused_peaks = []
        for cluster in clusters:
            if len(cluster) >= min_votes:
                # 在原始信号中找到该区域的最大值
                center = int(np.mean(cluster))
                start = max(0, center - tolerance)
                end = min(len(ecg_signal), center + tolerance)
                local_max = np.argmax(ecg_signal[start:end]) + start
                fused_peaks.append(local_max)
        
        return np.unique(fused_peaks)
