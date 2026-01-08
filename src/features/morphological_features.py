"""
ECG形态学特征提取模块
====================

从单个心拍和整体ECG波形中提取形态学特征
用于身份识别的关键特征

特征类别:
---------
1. 心拍形态特征 (Beat Morphology)
2. 波形统计特征 (Waveform Statistics)
3. 时频特征 (Time-Frequency)
4. 小波特征 (Wavelet Features)
"""

import numpy as np
from scipy import signal, stats
from scipy.fft import fft
import pywt
from typing import Dict, List, Tuple, Optional
from loguru import logger


class MorphologicalFeatureExtractor:
    """
    ECG形态学特征提取器
    
    提取个人特异性的ECG形态学特征
    用于身份识别
    """
    
    def __init__(self, sampling_rate: float = 200.0):
        """
        初始化特征提取器
        
        Args:
            sampling_rate: 采样率
        """
        self.sampling_rate = sampling_rate
        
        logger.info(f"初始化形态学特征提取器: fs={sampling_rate}Hz")
    
    def extract_all(
        self,
        beats: np.ndarray,
        ecg_signal: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        提取所有形态学特征
        
        Args:
            beats: 心拍数组 (n_beats, beat_length)
            ecg_signal: 完整ECG信号 (可选)
            
        Returns:
            特征字典
        """
        features = {}
        
        if len(beats) == 0:
            return features
        
        # 1. 单心拍形态特征 (使用平均心拍)
        mean_beat = np.mean(beats, axis=0)
        beat_features = self.extract_beat_features(mean_beat)
        features.update(beat_features)
        
        # 2. 心拍间变异性特征
        variability_features = self.extract_beat_variability(beats)
        features.update(variability_features)
        
        # 3. 小波特征
        wavelet_features = self.extract_wavelet_features(mean_beat)
        features.update(wavelet_features)
        
        # 4. 频域特征
        freq_features = self.extract_frequency_features(mean_beat)
        features.update(freq_features)
        
        # 5. 统计矩特征
        moment_features = self.extract_statistical_moments(mean_beat)
        features.update(moment_features)
        
        # 6. 形状描述符
        shape_features = self.extract_shape_descriptors(mean_beat)
        features.update(shape_features)
        
        return features
    
    def extract_beat_features(self, beat: np.ndarray) -> Dict[str, float]:
        """
        提取单心拍形态特征
        
        特征包括:
        - P波、QRS波群、T波的幅度和时间参数
        - 波形斜率和曲率
        - 面积和能量
        
        Args:
            beat: 单个心拍波形
            
        Returns:
            特征字典
        """
        features = {}
        
        n = len(beat)
        r_pos = n // 2  # 假设R峰在中心
        
        # ========== 基本统计 ==========
        features['beat_amplitude'] = np.max(beat) - np.min(beat)
        features['beat_mean'] = np.mean(beat)
        features['beat_std'] = np.std(beat)
        features['beat_energy'] = np.sum(beat ** 2)
        
        # R峰特征
        features['r_amplitude'] = beat[r_pos]
        
        # ========== QRS波群特征 ==========
        qrs_half = int(0.04 * self.sampling_rate)  # ±40ms
        qrs_start = max(0, r_pos - qrs_half)
        qrs_end = min(n, r_pos + qrs_half)
        qrs = beat[qrs_start:qrs_end]
        
        features['qrs_amplitude'] = np.max(qrs) - np.min(qrs)
        features['qrs_area'] = np.trapz(np.abs(qrs))
        features['qrs_energy'] = np.sum(qrs ** 2)
        
        # Q波 (R峰前的负向波)
        pre_r = beat[qrs_start:r_pos]
        if len(pre_r) > 0:
            q_idx = np.argmin(pre_r)
            features['q_amplitude'] = pre_r[q_idx]
            features['q_duration'] = q_idx / self.sampling_rate * 1000  # ms
        else:
            features['q_amplitude'] = 0
            features['q_duration'] = 0
        
        # S波 (R峰后的负向波)
        post_r = beat[r_pos:qrs_end]
        if len(post_r) > 0:
            s_idx = np.argmin(post_r)
            features['s_amplitude'] = post_r[s_idx]
            features['s_duration'] = s_idx / self.sampling_rate * 1000  # ms
        else:
            features['s_amplitude'] = 0
            features['s_duration'] = 0
        
        # ========== P波特征 ==========
        p_start = int(0.15 * self.sampling_rate)  # R峰前150ms开始
        p_end = int(0.05 * self.sampling_rate)    # R峰前50ms结束
        
        if r_pos - p_start >= 0:
            p_region = beat[r_pos - p_start:r_pos - p_end]
            if len(p_region) > 0:
                features['p_amplitude'] = np.max(p_region) - np.min(p_region)
                features['p_area'] = np.trapz(p_region)
                features['p_peak_pos'] = np.argmax(p_region) / self.sampling_rate * 1000
            else:
                features['p_amplitude'] = 0
                features['p_area'] = 0
                features['p_peak_pos'] = 0
        else:
            features['p_amplitude'] = 0
            features['p_area'] = 0
            features['p_peak_pos'] = 0
        
        # ========== T波特征 ==========
        t_start = int(0.1 * self.sampling_rate)   # R峰后100ms开始
        t_end = int(0.35 * self.sampling_rate)    # R峰后350ms结束
        
        if r_pos + t_end < n:
            t_region = beat[r_pos + t_start:r_pos + t_end]
            if len(t_region) > 0:
                features['t_amplitude'] = np.max(t_region) - np.min(t_region)
                features['t_area'] = np.trapz(t_region)
                features['t_peak_pos'] = np.argmax(t_region) / self.sampling_rate * 1000
                features['t_symmetry'] = self._compute_symmetry(t_region)
            else:
                features['t_amplitude'] = 0
                features['t_area'] = 0
                features['t_peak_pos'] = 0
                features['t_symmetry'] = 0
        else:
            features['t_amplitude'] = 0
            features['t_area'] = 0
            features['t_peak_pos'] = 0
            features['t_symmetry'] = 0
        
        # ========== ST段特征 ==========
        st_start = r_pos + int(0.04 * self.sampling_rate)
        st_end = r_pos + int(0.1 * self.sampling_rate)
        
        if st_end < n:
            st_segment = beat[st_start:st_end]
            features['st_level'] = np.mean(st_segment) - beat[r_pos - qrs_half]
            features['st_slope'] = np.polyfit(np.arange(len(st_segment)), st_segment, 1)[0]
        else:
            features['st_level'] = 0
            features['st_slope'] = 0
        
        # ========== 波形斜率特征 ==========
        # R波上升斜率
        if r_pos - 5 >= 0:
            r_up = beat[r_pos - 5:r_pos]
            features['r_upslope'] = np.max(np.diff(r_up)) * self.sampling_rate
        else:
            features['r_upslope'] = 0
        
        # R波下降斜率
        if r_pos + 5 < n:
            r_down = beat[r_pos:r_pos + 5]
            features['r_downslope'] = np.min(np.diff(r_down)) * self.sampling_rate
        else:
            features['r_downslope'] = 0
        
        return features
    
    def _compute_symmetry(self, wave: np.ndarray) -> float:
        """计算波形对称性"""
        if len(wave) < 3:
            return 0
        
        peak_idx = np.argmax(wave)
        left = wave[:peak_idx]
        right = wave[peak_idx:]
        
        # 将两边调整为相同长度
        min_len = min(len(left), len(right))
        if min_len == 0:
            return 0
        
        left_norm = left[-min_len:]
        right_norm = right[:min_len][::-1]  # 翻转右侧
        
        # 计算相关系数
        if np.std(left_norm) > 0 and np.std(right_norm) > 0:
            symmetry = np.corrcoef(left_norm, right_norm)[0, 1]
            return max(0, symmetry)
        
        return 0
    
    def extract_beat_variability(self, beats: np.ndarray) -> Dict[str, float]:
        """
        提取心拍间变异性特征
        
        捕捉心拍形态的一致性和变异性
        
        Args:
            beats: 心拍数组
            
        Returns:
            变异性特征
        """
        features = {}
        
        if len(beats) < 2:
            return {
                'beat_consistency': 0,
                'amplitude_cv': 0,
                'shape_variance': 0
            }
        
        # 心拍一致性 (平均相关系数)
        mean_beat = np.mean(beats, axis=0)
        correlations = []
        
        for beat in beats:
            if np.std(beat) > 0:
                corr = np.corrcoef(beat, mean_beat)[0, 1]
                correlations.append(corr)
        
        features['beat_consistency'] = np.mean(correlations) if correlations else 0
        
        # 幅度变异系数
        amplitudes = np.max(beats, axis=1) - np.min(beats, axis=1)
        features['amplitude_cv'] = np.std(amplitudes) / np.mean(amplitudes) * 100 if np.mean(amplitudes) > 0 else 0
        
        # 形状方差 (主成分分析)
        if len(beats) >= 3:
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(3, len(beats) - 1))
                pca.fit(beats)
                features['shape_variance'] = 1 - np.sum(pca.explained_variance_ratio_)
            except:
                features['shape_variance'] = np.var(beats)
        else:
            features['shape_variance'] = np.var(beats)
        
        # 特定点的变异性
        n = beats.shape[1]
        r_pos = n // 2
        
        # R峰幅度变异
        r_amplitudes = beats[:, r_pos]
        features['r_amplitude_std'] = np.std(r_amplitudes)
        
        return features
    
    def extract_wavelet_features(
        self,
        beat: np.ndarray,
        wavelet: str = 'db4',
        level: int = 4
    ) -> Dict[str, float]:
        """
        提取小波特征
        
        使用多尺度小波分解捕捉不同频率成分
        
        Args:
            beat: 心拍波形
            wavelet: 小波基
            level: 分解层数
            
        Returns:
            小波特征
        """
        features = {}
        
        # 小波分解
        coeffs = pywt.wavedec(beat, wavelet, level=level)
        
        # 每层系数的统计特征
        for i, coeff in enumerate(coeffs):
            prefix = f'dwt_l{i}_'
            
            features[prefix + 'energy'] = np.sum(coeff ** 2)
            features[prefix + 'mean'] = np.mean(coeff)
            features[prefix + 'std'] = np.std(coeff)
            features[prefix + 'max'] = np.max(np.abs(coeff))
            
            # 熵
            prob = np.abs(coeff) / (np.sum(np.abs(coeff)) + 1e-10)
            prob = prob[prob > 0]
            features[prefix + 'entropy'] = -np.sum(prob * np.log2(prob + 1e-10))
        
        # 能量分布
        total_energy = sum(np.sum(c ** 2) for c in coeffs)
        for i, coeff in enumerate(coeffs):
            features[f'dwt_l{i}_energy_ratio'] = np.sum(coeff ** 2) / (total_energy + 1e-10)
        
        return features
    
    def extract_frequency_features(self, beat: np.ndarray) -> Dict[str, float]:
        """
        提取频域特征
        
        Args:
            beat: 心拍波形
            
        Returns:
            频域特征
        """
        features = {}
        
        n = len(beat)
        
        # FFT
        fft_coeffs = fft(beat)
        fft_mag = np.abs(fft_coeffs[:n // 2])
        freqs = np.fft.fftfreq(n, 1 / self.sampling_rate)[:n // 2]
        
        # 总功率
        total_power = np.sum(fft_mag ** 2)
        features['total_spectral_power'] = total_power
        
        # 频带功率
        bands = {
            'low': (0, 10),
            'mid': (10, 40),
            'high': (40, 100)
        }
        
        for band_name, (f_low, f_high) in bands.items():
            mask = (freqs >= f_low) & (freqs < f_high)
            band_power = np.sum(fft_mag[mask] ** 2)
            features[f'{band_name}_freq_power'] = band_power
            features[f'{band_name}_freq_ratio'] = band_power / (total_power + 1e-10)
        
        # 主频
        features['dominant_freq'] = freqs[np.argmax(fft_mag)]
        
        # 频谱质心
        features['spectral_centroid'] = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-10)
        
        # 频谱展宽
        spectral_centroid = features['spectral_centroid']
        features['spectral_spread'] = np.sqrt(
            np.sum((freqs - spectral_centroid) ** 2 * fft_mag) / (np.sum(fft_mag) + 1e-10)
        )
        
        # 频谱熵
        fft_prob = fft_mag / (np.sum(fft_mag) + 1e-10)
        fft_prob = fft_prob[fft_prob > 0]
        features['spectral_entropy'] = -np.sum(fft_prob * np.log2(fft_prob + 1e-10))
        
        return features
    
    def extract_statistical_moments(self, beat: np.ndarray) -> Dict[str, float]:
        """
        提取统计矩特征
        
        Args:
            beat: 心拍波形
            
        Returns:
            统计矩特征
        """
        features = {}
        
        # 中心矩
        features['moment_1'] = np.mean(beat)  # 均值
        features['moment_2'] = np.var(beat)   # 方差
        features['moment_3'] = stats.skew(beat)  # 偏度
        features['moment_4'] = stats.kurtosis(beat)  # 峰度
        
        # 分位数
        features['q25'] = np.percentile(beat, 25)
        features['q50'] = np.percentile(beat, 50)
        features['q75'] = np.percentile(beat, 75)
        features['iqr'] = features['q75'] - features['q25']
        
        # 信号范围
        features['signal_range'] = np.max(beat) - np.min(beat)
        
        # 过零率
        zero_crossings = np.sum(np.diff(np.sign(beat - np.mean(beat))) != 0)
        features['zero_crossing_rate'] = zero_crossings / len(beat)
        
        return features
    
    def extract_shape_descriptors(self, beat: np.ndarray) -> Dict[str, float]:
        """
        提取形状描述符
        
        使用几何和拓扑特征描述心拍形状
        
        Args:
            beat: 心拍波形
            
        Returns:
            形状特征
        """
        features = {}
        
        n = len(beat)
        
        # 曲线长度
        diff = np.diff(beat)
        curve_length = np.sum(np.sqrt(1 + diff ** 2))
        features['curve_length'] = curve_length
        
        # 曲线弯曲度 (总曲率)
        if n > 2:
            second_diff = np.diff(diff)
            curvature = np.sum(np.abs(second_diff)) / n
            features['curvature'] = curvature
        else:
            features['curvature'] = 0
        
        # 波形复杂度
        features['waveform_complexity'] = curve_length / (np.max(beat) - np.min(beat) + 1e-10)
        
        # 面积 (相对于基线)
        baseline = np.min(beat)
        features['area_above_baseline'] = np.trapz(beat - baseline)
        
        # 正/负面积比
        positive_area = np.trapz(np.maximum(beat - np.mean(beat), 0))
        negative_area = np.trapz(np.maximum(np.mean(beat) - beat, 0))
        features['pos_neg_area_ratio'] = positive_area / (negative_area + 1e-10)
        
        # 峰值数量
        peaks, _ = signal.find_peaks(beat, prominence=0.1 * (np.max(beat) - np.min(beat)))
        valleys, _ = signal.find_peaks(-beat, prominence=0.1 * (np.max(beat) - np.min(beat)))
        
        features['n_peaks'] = len(peaks)
        features['n_valleys'] = len(valleys)
        
        # Hjorth参数
        features.update(self._hjorth_parameters(beat))
        
        return features
    
    def _hjorth_parameters(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        计算Hjorth参数
        
        Hjorth参数描述信号的活动性、移动性和复杂性
        
        - Activity: 信号方差
        - Mobility: 一阶导数方差/信号方差
        - Complexity: 二阶导数移动性/一阶导数移动性
        
        Args:
            signal_data: 输入信号
            
        Returns:
            Hjorth参数
        """
        # Activity
        activity = np.var(signal_data)
        
        # 一阶导数
        diff1 = np.diff(signal_data)
        
        # Mobility
        mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
        
        # 二阶导数
        diff2 = np.diff(diff1)
        mobility_d1 = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10))
        
        # Complexity
        complexity = mobility_d1 / (mobility + 1e-10)
        
        return {
            'hjorth_activity': activity,
            'hjorth_mobility': mobility,
            'hjorth_complexity': complexity
        }
    
    def extract_template_features(
        self,
        beats: np.ndarray,
        n_templates: int = 3
    ) -> np.ndarray:
        """
        提取模板特征
        
        使用聚类获取代表性模板，然后计算每个心拍与模板的相似度
        
        Args:
            beats: 心拍数组
            n_templates: 模板数量
            
        Returns:
            模板相似度特征矩阵
        """
        if len(beats) < n_templates:
            return np.zeros((len(beats), n_templates))
        
        from sklearn.cluster import KMeans
        
        # 聚类获取模板
        kmeans = KMeans(n_clusters=n_templates, random_state=42, n_init=10)
        kmeans.fit(beats)
        templates = kmeans.cluster_centers_
        
        # 计算每个心拍与各模板的相似度
        similarities = np.zeros((len(beats), n_templates))
        
        for i, beat in enumerate(beats):
            for j, template in enumerate(templates):
                # 使用相关系数作为相似度
                if np.std(beat) > 0 and np.std(template) > 0:
                    similarities[i, j] = np.corrcoef(beat, template)[0, 1]
        
        return similarities
