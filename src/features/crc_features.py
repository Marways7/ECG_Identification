"""
CRC (心肺耦合) 特征提取模块
==========================

心肺耦合 (Cardiorespiratory Coupling) 描述心脏和呼吸系统之间的
动态交互关系，是评估自主神经功能的重要指标。

计算方法:
---------
1. 相位同步指数 (PSI, Phase Synchronization Index)
2. 互信息 (MI, Mutual Information)
3. 交叉谱相干性 (Coherence)
4. 传递熵 (Transfer Entropy)

数学原理:
---------
相位同步:
从ECG导出呼吸信号(EDR)和RR间期序列
使用Hilbert变换提取瞬时相位
计算相位差的一致性

PSI = |⟨e^{i·Δφ(t)}⟩| ∈ [0, 1]
其中 Δφ(t) = φ_resp(t) - n·φ_cardiac(t)
n:m 相位同步对应特定的心肺耦合模式

参考文献:
- Bartsch et al., PNAS, 2012
- Schulz et al., Philos Trans A, 2013
"""

import numpy as np
from scipy import signal, interpolate, stats
from typing import Dict, Tuple, Optional, List
from loguru import logger


class CRCFeatureExtractor:
    """
    心肺耦合特征提取器
    
    从ECG和呼吸信号中提取心肺耦合指标
    
    Attributes:
        sampling_rate: 原始采样率 (Hz)
        resp_rate_range: 呼吸频率范围 (Hz)
    """
    
    def __init__(
        self,
        sampling_rate: float = 200.0,
        resp_rate_range: Tuple[float, float] = (0.1, 0.5)
    ):
        """
        初始化CRC特征提取器
        
        Args:
            sampling_rate: 采样率
            resp_rate_range: 正常呼吸频率范围 (6-30次/分 = 0.1-0.5Hz)
        """
        self.sampling_rate = sampling_rate
        self.resp_rate_range = resp_rate_range
        
        logger.info(f"初始化CRC特征提取器: fs={sampling_rate}Hz")
    
    def extract_all(
        self,
        ecg_signal: np.ndarray,
        r_peaks: np.ndarray,
        resp_signal: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        提取所有CRC特征
        
        Args:
            ecg_signal: ECG信号
            r_peaks: R峰位置
            resp_signal: 呼吸信号 (可选，如果没有则从ECG导出)
            
        Returns:
            CRC特征字典
        """
        features = {}
        
        # 如果没有呼吸信号，从ECG导出
        if resp_signal is None:
            resp_signal = self.extract_edr(ecg_signal, r_peaks)
        
        # 计算RR间期序列
        rr_intervals = np.diff(r_peaks) / self.sampling_rate * 1000  # ms
        rr_times = np.cumsum(rr_intervals) / 1000  # 秒
        
        if len(rr_intervals) < 30 or len(resp_signal) < 100:
            logger.warning("数据不足，无法计算CRC特征")
            return self._empty_features()
        
        try:
            # 1. 相位同步指数
            psi_features = self.compute_phase_synchronization(
                rr_intervals, rr_times, resp_signal
            )
            features.update(psi_features)
            
            # 2. 交叉谱相干性
            coherence_features = self.compute_coherence(
                rr_intervals, rr_times, resp_signal
            )
            features.update(coherence_features)
            
            # 3. 呼吸性窦性心律不齐 (RSA)
            rsa_features = self.compute_rsa(rr_intervals, rr_times, resp_signal)
            features.update(rsa_features)
            
            # 4. 互信息
            mi_features = self.compute_mutual_information(
                rr_intervals, resp_signal, r_peaks
            )
            features.update(mi_features)
            
            # 5. 传递熵
            te_features = self.compute_transfer_entropy(
                rr_intervals, resp_signal, r_peaks
            )
            features.update(te_features)
            
        except Exception as e:
            logger.error(f"CRC特征提取失败: {e}")
            return self._empty_features()
        
        return features
    
    def _empty_features(self) -> Dict[str, float]:
        """返回空特征字典"""
        return {
            'psi': 0, 'psi_strength': 0, 'phase_diff_std': 0,
            'coherence_mean': 0, 'coherence_max': 0, 'coherence_freq': 0,
            'rsa_amplitude': 0, 'rsa_power': 0,
            'mi_cardiac_resp': 0, 'mi_resp_cardiac': 0,
            'te_cardiac_to_resp': 0, 'te_resp_to_cardiac': 0
        }
    
    def extract_edr(
        self,
        ecg_signal: np.ndarray,
        r_peaks: np.ndarray,
        method: str = 'amplitude'
    ) -> np.ndarray:
        """
        从ECG提取呼吸信号 (ECG-Derived Respiration)
        
        方法:
        1. R峰幅度法: 呼吸调制R波幅度
        2. RR间期法: 呼吸性窦性心律不齐
        3. QRS面积法: 呼吸调制QRS波群面积
        
        Args:
            ecg_signal: ECG信号
            r_peaks: R峰位置
            method: 提取方法
            
        Returns:
            EDR信号
        """
        if method == 'amplitude':
            # R峰幅度
            edr_values = ecg_signal[r_peaks]
            
        elif method == 'rri':
            # RR间期变化
            rr_intervals = np.diff(r_peaks)
            edr_values = np.concatenate([[rr_intervals[0]], rr_intervals])
            
        elif method == 'area':
            # QRS面积
            window = int(0.04 * self.sampling_rate)  # 40ms窗口
            edr_values = []
            for peak in r_peaks:
                start = max(0, peak - window)
                end = min(len(ecg_signal), peak + window)
                area = np.trapz(np.abs(ecg_signal[start:end]))
                edr_values.append(area)
            edr_values = np.array(edr_values)
        
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        # 带通滤波 (0.1-0.5Hz，呼吸频率范围)
        edr_times = r_peaks / self.sampling_rate
        
        # 插值到均匀采样 (4Hz)
        target_fs = 4.0
        t_uniform = np.arange(edr_times[0], edr_times[-1], 1/target_fs)
        
        f_interp = interpolate.interp1d(
            edr_times, edr_values,
            kind='cubic',
            fill_value='extrapolate'
        )
        edr_uniform = f_interp(t_uniform)
        
        # 带通滤波
        nyquist = target_fs / 2
        low = self.resp_rate_range[0] / nyquist
        high = min(self.resp_rate_range[1], nyquist - 0.01) / nyquist
        
        if high > low:
            b, a = signal.butter(3, [low, high], btype='band')
            edr_filtered = signal.filtfilt(b, a, edr_uniform)
        else:
            edr_filtered = edr_uniform - np.mean(edr_uniform)
        
        return edr_filtered
    
    def compute_phase_synchronization(
        self,
        rr_intervals: np.ndarray,
        rr_times: np.ndarray,
        resp_signal: np.ndarray
    ) -> Dict[str, float]:
        """
        计算相位同步指数 (Phase Synchronization Index)
        
        步骤:
        1. Hilbert变换提取瞬时相位
        2. 计算心脏-呼吸相位差
        3. 计算相位同步指数 PSI = |⟨e^{iΔφ}⟩|
        
        PSI = 1: 完全同步
        PSI = 0: 完全不同步
        
        Args:
            rr_intervals: RR间期序列 (ms)
            rr_times: RR间期时间点 (s)
            resp_signal: 呼吸信号
            
        Returns:
            相位同步特征
        """
        features = {}
        
        # 将RR序列插值到与呼吸信号相同的采样率
        target_fs = 4.0  # Hz
        n_samples = len(resp_signal)
        
        if len(rr_times) < 2:
            return {'psi': 0, 'psi_strength': 0, 'phase_diff_std': 0}
        
        t_uniform = np.linspace(rr_times[0], rr_times[-1], n_samples)
        
        f_interp = interpolate.interp1d(
            rr_times, rr_intervals[1:] if len(rr_intervals) > len(rr_times) else rr_intervals[:len(rr_times)],
            kind='cubic',
            fill_value='extrapolate'
        )
        rr_uniform = f_interp(t_uniform[:len(t_uniform)])
        
        # 去均值
        rr_centered = rr_uniform - np.mean(rr_uniform)
        resp_centered = resp_signal[:len(rr_centered)] - np.mean(resp_signal[:len(rr_centered)])
        
        # Hilbert变换提取相位
        rr_analytic = signal.hilbert(rr_centered)
        resp_analytic = signal.hilbert(resp_centered)
        
        rr_phase = np.angle(rr_analytic)
        resp_phase = np.angle(resp_analytic)
        
        # 计算相位差 (1:1同步)
        phase_diff = resp_phase - rr_phase
        
        # 相位同步指数 (平均相位向量长度)
        psi = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        # 相位差的标准差
        phase_diff_std = np.std(np.mod(phase_diff + np.pi, 2*np.pi) - np.pi)
        
        # 同步强度 (基于Shannon熵)
        n_bins = 18  # 每20度一个bin
        hist, _ = np.histogram(np.mod(phase_diff, 2*np.pi), bins=n_bins, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist)) / np.log(n_bins)
        psi_strength = 1 - entropy
        
        features['psi'] = psi
        features['psi_strength'] = psi_strength
        features['phase_diff_std'] = phase_diff_std
        
        return features
    
    def compute_coherence(
        self,
        rr_intervals: np.ndarray,
        rr_times: np.ndarray,
        resp_signal: np.ndarray
    ) -> Dict[str, float]:
        """
        计算交叉谱相干性 (Cross-Spectral Coherence)
        
        相干性衡量两个信号在特定频率上的线性相关程度
        
        Cxy(f) = |Pxy(f)|² / (Pxx(f) · Pyy(f))
        
        Cxy ∈ [0, 1]
        - 1: 完全线性相关
        - 0: 无相关
        
        Args:
            rr_intervals: RR间期序列
            rr_times: 时间点
            resp_signal: 呼吸信号
            
        Returns:
            相干性特征
        """
        features = {}
        
        target_fs = 4.0
        n_samples = len(resp_signal)
        
        if len(rr_times) < 2:
            return {'coherence_mean': 0, 'coherence_max': 0, 'coherence_freq': 0}
        
        # 插值
        t_uniform = np.linspace(rr_times[0], rr_times[-1], n_samples)
        f_interp = interpolate.interp1d(
            rr_times, rr_intervals[1:] if len(rr_intervals) > len(rr_times) else rr_intervals[:len(rr_times)],
            kind='cubic',
            fill_value='extrapolate'
        )
        rr_uniform = f_interp(t_uniform)
        
        # 计算相干性
        nperseg = min(256, len(rr_uniform) // 4)
        if nperseg < 32:
            nperseg = min(32, len(rr_uniform))
        
        freqs, coherence = signal.coherence(
            rr_uniform,
            resp_signal[:len(rr_uniform)],
            fs=target_fs,
            nperseg=nperseg
        )
        
        # 呼吸频率范围内的相干性
        resp_mask = (freqs >= self.resp_rate_range[0]) & (freqs <= self.resp_rate_range[1])
        
        if np.any(resp_mask):
            features['coherence_mean'] = np.mean(coherence[resp_mask])
            features['coherence_max'] = np.max(coherence[resp_mask])
            features['coherence_freq'] = freqs[resp_mask][np.argmax(coherence[resp_mask])]
        else:
            features['coherence_mean'] = 0
            features['coherence_max'] = 0
            features['coherence_freq'] = 0
        
        return features
    
    def compute_rsa(
        self,
        rr_intervals: np.ndarray,
        rr_times: np.ndarray,
        resp_signal: np.ndarray
    ) -> Dict[str, float]:
        """
        计算呼吸性窦性心律不齐 (Respiratory Sinus Arrhythmia)
        
        RSA是心率随呼吸周期变化的现象:
        - 吸气时心率加快
        - 呼气时心率减慢
        
        RSA反映迷走神经对心脏的调节
        
        计算方法:
        1. 检测呼吸周期
        2. 计算每个周期内RR间期的最大-最小差
        3. 或使用HF频带功率作为RSA指标
        
        Args:
            rr_intervals: RR间期序列
            rr_times: 时间点
            resp_signal: 呼吸信号
            
        Returns:
            RSA特征
        """
        features = {}
        
        # 方法1: 峰-谷法 (peak-to-trough)
        # 找呼吸信号的峰和谷
        resp_peaks, _ = signal.find_peaks(resp_signal, distance=int(2.0 * 4))  # 假设4Hz采样
        resp_valleys, _ = signal.find_peaks(-resp_signal, distance=int(2.0 * 4))
        
        target_fs = 4.0
        
        if len(resp_peaks) > 2 and len(rr_times) > 2:
            # 插值RR到呼吸采样点
            n_samples = len(resp_signal)
            t_uniform = np.linspace(rr_times[0], rr_times[-1], n_samples)
            
            f_interp = interpolate.interp1d(
                rr_times, 
                rr_intervals[1:] if len(rr_intervals) > len(rr_times) else rr_intervals[:len(rr_times)],
                kind='cubic',
                fill_value='extrapolate'
            )
            rr_uniform = f_interp(t_uniform)
            
            # 计算每个呼吸周期内的RSA
            rsa_values = []
            
            for i in range(len(resp_peaks) - 1):
                start = resp_peaks[i]
                end = resp_peaks[i + 1]
                
                if end < len(rr_uniform):
                    cycle_rr = rr_uniform[start:end]
                    if len(cycle_rr) > 0:
                        rsa_values.append(np.max(cycle_rr) - np.min(cycle_rr))
            
            if rsa_values:
                features['rsa_amplitude'] = np.mean(rsa_values)
            else:
                features['rsa_amplitude'] = 0
        else:
            features['rsa_amplitude'] = 0
        
        # 方法2: HF频带功率
        if len(rr_intervals) > 30:
            # 使用Welch方法估计PSD
            nperseg = min(64, len(rr_intervals) // 2)
            
            # 先插值到均匀采样
            rr_interp_fs = 4.0
            t_uniform = np.linspace(rr_times[0], rr_times[-1], 
                                    int((rr_times[-1] - rr_times[0]) * rr_interp_fs))
            f_interp = interpolate.interp1d(
                rr_times,
                rr_intervals[1:] if len(rr_intervals) > len(rr_times) else rr_intervals[:len(rr_times)],
                kind='cubic',
                fill_value='extrapolate'
            )
            rr_uniform = f_interp(t_uniform)
            
            freqs, psd = signal.welch(
                rr_uniform - np.mean(rr_uniform),
                fs=rr_interp_fs,
                nperseg=min(nperseg, len(rr_uniform) // 2)
            )
            
            # HF频带 (0.15-0.4 Hz) 作为RSA功率
            hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
            if np.any(hf_mask):
                features['rsa_power'] = np.trapz(psd[hf_mask], freqs[hf_mask])
            else:
                features['rsa_power'] = 0
        else:
            features['rsa_power'] = 0
        
        return features
    
    def compute_mutual_information(
        self,
        rr_intervals: np.ndarray,
        resp_signal: np.ndarray,
        r_peaks: np.ndarray
    ) -> Dict[str, float]:
        """
        计算互信息 (Mutual Information)
        
        MI量化两个变量之间的相互依赖程度
        
        MI(X,Y) = H(X) + H(Y) - H(X,Y)
        
        其中H是Shannon熵
        
        Args:
            rr_intervals: RR间期序列
            resp_signal: 呼吸信号
            r_peaks: R峰位置
            
        Returns:
            互信息特征
        """
        features = {}
        
        # 将两个序列对齐
        rr_times = np.cumsum(rr_intervals) / 1000
        target_fs = 4.0
        
        if len(rr_times) < 2:
            return {'mi_cardiac_resp': 0, 'mi_resp_cardiac': 0}
        
        n_samples = min(len(resp_signal), int((rr_times[-1] - rr_times[0]) * target_fs))
        t_uniform = np.linspace(rr_times[0], rr_times[-1], n_samples)
        
        f_interp = interpolate.interp1d(
            rr_times,
            rr_intervals[1:] if len(rr_intervals) > len(rr_times) else rr_intervals[:len(rr_times)],
            kind='cubic',
            fill_value='extrapolate'
        )
        rr_uniform = f_interp(t_uniform)
        
        resp_aligned = resp_signal[:len(rr_uniform)]
        
        # 离散化
        n_bins = 10
        rr_binned = np.digitize(rr_uniform, np.linspace(rr_uniform.min(), rr_uniform.max(), n_bins))
        resp_binned = np.digitize(resp_aligned, np.linspace(resp_aligned.min(), resp_aligned.max(), n_bins))
        
        # 计算互信息
        mi = self._mutual_information(rr_binned, resp_binned)
        
        features['mi_cardiac_resp'] = mi
        features['mi_resp_cardiac'] = mi  # MI是对称的
        
        return features
    
    def _mutual_information(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        计算互信息
        
        Args:
            x, y: 离散化后的序列
            
        Returns:
            互信息值
        """
        # 联合概率分布
        joint_hist = np.histogram2d(x, y, bins=10)[0]
        joint_prob = joint_hist / joint_hist.sum()
        
        # 边缘概率
        px = joint_prob.sum(axis=1)
        py = joint_prob.sum(axis=0)
        
        # 计算MI
        mi = 0
        for i in range(len(px)):
            for j in range(len(py)):
                if joint_prob[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (px[i] * py[j]))
        
        return max(0, mi)
    
    def compute_transfer_entropy(
        self,
        rr_intervals: np.ndarray,
        resp_signal: np.ndarray,
        r_peaks: np.ndarray,
        lag: int = 1
    ) -> Dict[str, float]:
        """
        计算传递熵 (Transfer Entropy)
        
        传递熵量化有向信息流
        
        TE(X→Y) = H(Y_{t+1}|Y_t) - H(Y_{t+1}|Y_t, X_t)
        
        正值表示X→Y存在信息传递
        
        Args:
            rr_intervals: RR间期序列
            resp_signal: 呼吸信号
            r_peaks: R峰位置
            lag: 时间滞后
            
        Returns:
            传递熵特征
        """
        features = {}
        
        # 对齐序列
        rr_times = np.cumsum(rr_intervals) / 1000
        target_fs = 4.0
        
        if len(rr_times) < lag + 10:
            return {'te_cardiac_to_resp': 0, 'te_resp_to_cardiac': 0}
        
        n_samples = min(len(resp_signal), int((rr_times[-1] - rr_times[0]) * target_fs))
        t_uniform = np.linspace(rr_times[0], rr_times[-1], n_samples)
        
        f_interp = interpolate.interp1d(
            rr_times,
            rr_intervals[1:] if len(rr_intervals) > len(rr_times) else rr_intervals[:len(rr_times)],
            kind='cubic',
            fill_value='extrapolate'
        )
        rr_uniform = f_interp(t_uniform)
        
        resp_aligned = resp_signal[:len(rr_uniform)]
        
        # 离散化
        n_bins = 5
        rr_binned = np.digitize(rr_uniform, np.linspace(rr_uniform.min(), rr_uniform.max(), n_bins))
        resp_binned = np.digitize(resp_aligned, np.linspace(resp_aligned.min(), resp_aligned.max(), n_bins))
        
        # 计算传递熵
        # RR → Resp
        te_rr_to_resp = self._transfer_entropy(rr_binned, resp_binned, lag)
        
        # Resp → RR
        te_resp_to_rr = self._transfer_entropy(resp_binned, rr_binned, lag)
        
        features['te_cardiac_to_resp'] = te_rr_to_resp
        features['te_resp_to_cardiac'] = te_resp_to_rr
        
        return features
    
    def _transfer_entropy(
        self,
        source: np.ndarray,
        target: np.ndarray,
        lag: int = 1
    ) -> float:
        """
        计算单向传递熵
        
        TE(source → target) = 
            H(target_{t+1}|target_t) - H(target_{t+1}|target_t, source_t)
        
        Args:
            source: 源序列
            target: 目标序列
            lag: 时间滞后
            
        Returns:
            传递熵值
        """
        n = len(source) - lag
        
        if n < 10:
            return 0
        
        # 构建时间嵌入
        target_t = target[:-lag]
        target_t1 = target[lag:]
        source_t = source[:-lag]
        
        # 条件熵 H(target_{t+1}|target_t)
        h_t1_given_t = self._conditional_entropy(target_t1, target_t)
        
        # 条件熵 H(target_{t+1}|target_t, source_t)
        # 使用联合条件
        joint_condition = target_t * 100 + source_t  # 简单编码
        h_t1_given_ts = self._conditional_entropy(target_t1, joint_condition)
        
        te = h_t1_given_t - h_t1_given_ts
        
        return max(0, te)
    
    def _conditional_entropy(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        计算条件熵 H(X|Y)
        
        H(X|Y) = H(X,Y) - H(Y)
        
        Args:
            x, y: 离散序列
            
        Returns:
            条件熵
        """
        # 联合熵
        joint_hist = np.histogram2d(x, y, bins=10)[0]
        joint_prob = joint_hist / joint_hist.sum()
        joint_prob = joint_prob[joint_prob > 0]
        h_xy = -np.sum(joint_prob * np.log2(joint_prob))
        
        # Y的熵
        hist_y = np.histogram(y, bins=10)[0]
        prob_y = hist_y / hist_y.sum()
        prob_y = prob_y[prob_y > 0]
        h_y = -np.sum(prob_y * np.log2(prob_y))
        
        return h_xy - h_y


class CRCAnalyzer:
    """
    综合心肺耦合分析器
    
    提供可视化和报告功能
    """
    
    def __init__(self, sampling_rate: float = 200.0):
        self.crc_extractor = CRCFeatureExtractor(sampling_rate)
        self.sampling_rate = sampling_rate
    
    def analyze(
        self,
        ecg_signal: np.ndarray,
        r_peaks: np.ndarray,
        resp_signal: Optional[np.ndarray] = None
    ) -> Dict:
        """
        完整的CRC分析
        
        Args:
            ecg_signal: ECG信号
            r_peaks: R峰位置
            resp_signal: 呼吸信号
            
        Returns:
            分析结果字典
        """
        results = {}
        
        # 提取特征
        features = self.crc_extractor.extract_all(ecg_signal, r_peaks, resp_signal)
        results['features'] = features
        
        # 计算耦合强度等级
        psi = features.get('psi', 0)
        coherence = features.get('coherence_max', 0)
        
        # 综合耦合指数
        coupling_index = (psi + coherence) / 2
        
        if coupling_index > 0.7:
            coupling_level = '强耦合'
        elif coupling_index > 0.4:
            coupling_level = '中等耦合'
        else:
            coupling_level = '弱耦合'
        
        results['coupling_index'] = coupling_index
        results['coupling_level'] = coupling_level
        
        # 方向性
        te_cr = features.get('te_cardiac_to_resp', 0)
        te_rc = features.get('te_resp_to_cardiac', 0)
        
        if te_cr > te_rc * 1.2:
            direction = '心脏→呼吸 主导'
        elif te_rc > te_cr * 1.2:
            direction = '呼吸→心脏 主导'
        else:
            direction = '双向耦合'
        
        results['coupling_direction'] = direction
        
        return results
