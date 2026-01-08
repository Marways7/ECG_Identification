"""
HRV (心率变异性) 特征提取模块
============================

实现全面的HRV分析:
1. 时域指标 (Time-domain)
2. 频域指标 (Frequency-domain)  
3. 非线性指标 (Nonlinear)
4. Poincaré图分析

数学原理:
---------
HRV基于RR间期 (R-R Interval, RRI) 序列分析
RRI = {RR_1, RR_2, ..., RR_N}, 单位为毫秒(ms)

时域指标:
- SDNN: √(1/(N-1) · Σ(RRi - mean(RR))²)
- RMSSD: √(1/(N-1) · Σ(RRi+1 - RRi)²)
- pNN50: #{|ΔRRi| > 50ms} / N · 100%

频域指标 (功率谱分析):
- VLF: 0.003-0.04 Hz (极低频)
- LF: 0.04-0.15 Hz (低频，交感+副交感)
- HF: 0.15-0.4 Hz (高频，副交感)
- LF/HF: 交感/副交感平衡

非线性指标:
- SD1, SD2: Poincaré图椭圆轴
- SampEn: 样本熵
- ApEn: 近似熵
- DFA: 去趋势波动分析

参考文献:
- Task Force of ESC and NASPE, 1996
"""

import numpy as np
from scipy import signal, interpolate, stats
from typing import Dict, Tuple, Optional
from loguru import logger


class HRVFeatureExtractor:
    """
    HRV特征提取器
    
    从RR间期序列中提取时域、频域和非线性特征
    
    Attributes:
        sampling_rate: 原始ECG采样率 (Hz)
        interpolation_rate: RRI序列重采样率 (Hz)
    """
    
    # 频域分析频带定义 (Hz)
    FREQ_BANDS = {
        'VLF': (0.003, 0.04),
        'LF': (0.04, 0.15),
        'HF': (0.15, 0.4)
    }
    
    def __init__(
        self,
        sampling_rate: float = 200.0,
        interpolation_rate: float = 4.0
    ):
        """
        初始化HRV特征提取器
        
        Args:
            sampling_rate: ECG采样率
            interpolation_rate: RRI序列重采样率 (推荐4Hz)
        """
        self.sampling_rate = sampling_rate
        self.interpolation_rate = interpolation_rate
        
        logger.info(f"初始化HRV特征提取器: fs={sampling_rate}Hz, "
                   f"interpolation={interpolation_rate}Hz")
    
    def extract_all(
        self,
        r_peaks: np.ndarray,
        include_nonlinear: bool = True
    ) -> Dict[str, float]:
        """
        提取所有HRV特征
        
        Args:
            r_peaks: R峰位置索引数组
            include_nonlinear: 是否包含非线性特征
            
        Returns:
            特征字典
        """
        # 计算RR间期 (毫秒)
        rr_intervals = np.diff(r_peaks) / self.sampling_rate * 1000
        
        if len(rr_intervals) < 10:
            logger.warning("RR间期数量不足，无法计算HRV特征")
            return {}
        
        # 去除异常RR间期 (生理上不可能的值)
        rr_intervals = self._filter_rr_intervals(rr_intervals)
        
        if len(rr_intervals) < 10:
            return {}
        
        features = {}
        
        # 时域特征
        time_features = self.extract_time_domain(rr_intervals)
        features.update(time_features)
        
        # 频域特征
        freq_features = self.extract_frequency_domain(rr_intervals, r_peaks)
        features.update(freq_features)
        
        # 非线性特征
        if include_nonlinear:
            nonlinear_features = self.extract_nonlinear(rr_intervals)
            features.update(nonlinear_features)
        
        return features
    
    def _filter_rr_intervals(
        self, 
        rr_intervals: np.ndarray,
        min_rr: float = 300,
        max_rr: float = 2000
    ) -> np.ndarray:
        """
        过滤异常RR间期
        
        去除生理上不可能的RR间期:
        - 小于300ms (~200bpm)
        - 大于2000ms (~30bpm)
        - 与前后差异过大 (>20%)
        
        Args:
            rr_intervals: RR间期序列
            min_rr: 最小RR间期 (ms)
            max_rr: 最大RR间期 (ms)
            
        Returns:
            过滤后的RR间期
        """
        # 基本范围过滤
        mask = (rr_intervals >= min_rr) & (rr_intervals <= max_rr)
        
        # 异常值过滤 (基于中位数绝对偏差)
        median = np.median(rr_intervals)
        mad = np.median(np.abs(rr_intervals - median))
        threshold = 3 * mad / 0.6745  # 转换为标准差等效
        
        mask &= np.abs(rr_intervals - median) < threshold
        
        return rr_intervals[mask]
    
    def extract_time_domain(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        提取时域HRV特征
        
        时域特征反映RR间期的统计特性:
        
        1. 统计指标:
           - Mean RR: 平均RR间期
           - SDNN: RR间期标准差 (整体HRV)
           - Mean HR: 平均心率
           - STD HR: 心率标准差
        
        2. 差分指标:
           - RMSSD: 相邻RR差值的均方根 (短期HRV)
           - SDSD: 相邻RR差值的标准差
           - pNN50: 相邻RR差值>50ms的比例 (%)
           - pNN20: 相邻RR差值>20ms的比例 (%)
        
        3. 几何指标:
           - HRV三角指数: 总RR间期数 / 最大柱状图高度
        
        Args:
            rr_intervals: RR间期序列 (ms)
            
        Returns:
            时域特征字典
        """
        features = {}
        
        n = len(rr_intervals)
        
        # 基本统计
        features['rr_mean'] = np.mean(rr_intervals)
        features['rr_std'] = np.std(rr_intervals, ddof=1) if n > 1 else 0
        features['rr_min'] = np.min(rr_intervals)
        features['rr_max'] = np.max(rr_intervals)
        features['rr_range'] = features['rr_max'] - features['rr_min']
        
        # 心率统计
        heart_rates = 60000 / rr_intervals  # BPM
        features['hr_mean'] = np.mean(heart_rates)
        features['hr_std'] = np.std(heart_rates, ddof=1) if n > 1 else 0
        features['hr_min'] = np.min(heart_rates)
        features['hr_max'] = np.max(heart_rates)
        
        # SDNN (Standard Deviation of NN intervals)
        # 反映整体HRV，主要受交感神经调节
        features['sdnn'] = features['rr_std']
        
        # 差分指标
        rr_diff = np.diff(rr_intervals)
        
        # RMSSD (Root Mean Square of Successive Differences)
        # 反映短期HRV，主要受副交感神经调节
        features['rmssd'] = np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) > 0 else 0
        
        # SDSD (Standard Deviation of Successive Differences)
        features['sdsd'] = np.std(rr_diff, ddof=1) if len(rr_diff) > 1 else 0
        
        # pNN50: 相邻RR差值>50ms的比例
        if len(rr_diff) > 0:
            features['pnn50'] = np.sum(np.abs(rr_diff) > 50) / len(rr_diff) * 100
            features['pnn20'] = np.sum(np.abs(rr_diff) > 20) / len(rr_diff) * 100
            features['nn50'] = np.sum(np.abs(rr_diff) > 50)  # 绝对数量
        else:
            features['pnn50'] = 0
            features['pnn20'] = 0
            features['nn50'] = 0
        
        # CV (Coefficient of Variation)
        features['cv_rr'] = features['rr_std'] / features['rr_mean'] * 100 if features['rr_mean'] > 0 else 0
        features['cv_hr'] = features['hr_std'] / features['hr_mean'] * 100 if features['hr_mean'] > 0 else 0
        
        # 几何指标: HRV三角指数
        try:
            hist, bin_edges = np.histogram(rr_intervals, bins='auto')
            if np.max(hist) > 0:
                features['hrv_triangular_index'] = n / np.max(hist)
            else:
                features['hrv_triangular_index'] = 0
        except:
            features['hrv_triangular_index'] = 0
        
        return features
    
    def extract_frequency_domain(
        self,
        rr_intervals: np.ndarray,
        r_peaks: np.ndarray
    ) -> Dict[str, float]:
        """
        提取频域HRV特征
        
        使用Welch方法估计功率谱密度(PSD):
        
        频带划分 (根据Task Force标准):
        - VLF (Very Low Frequency): 0.003-0.04 Hz
          温度调节、RAAS系统、可能的交感活动
        - LF (Low Frequency): 0.04-0.15 Hz
          混合交感和副交感活动，以交感为主
        - HF (High Frequency): 0.15-0.4 Hz
          副交感/迷走神经活动，呼吸相关
        
        计算方法:
        1. 将不等间隔RRI序列插值为等间隔序列
        2. 去趋势 (减去均值)
        3. Welch PSD估计
        4. 计算各频带功率
        
        Args:
            rr_intervals: RR间期序列 (ms)
            r_peaks: R峰位置
            
        Returns:
            频域特征字典
        """
        features = {}
        
        try:
            # 计算时间点 (秒)
            rr_times = np.cumsum(rr_intervals) / 1000
            rr_times = np.insert(rr_times, 0, 0)[:-1]
            
            # 插值到均匀采样
            if len(rr_intervals) < 10 or rr_times[-1] - rr_times[0] < 10:
                logger.warning("RR序列太短，无法进行频域分析")
                return self._empty_freq_features()
            
            # 三次样条插值
            f_interp = interpolate.interp1d(
                rr_times, 
                rr_intervals, 
                kind='cubic',
                fill_value='extrapolate'
            )
            
            # 均匀采样
            t_uniform = np.arange(
                rr_times[0],
                rr_times[-1],
                1 / self.interpolation_rate
            )
            rr_uniform = f_interp(t_uniform)
            
            # 去趋势
            rr_detrend = signal.detrend(rr_uniform - np.mean(rr_uniform))
            
            # Welch PSD估计
            # 窗口长度: 256点或信号长度的1/4，取较小值
            nperseg = min(256, len(rr_detrend) // 4)
            if nperseg < 32:
                nperseg = min(32, len(rr_detrend))
            
            freqs, psd = signal.welch(
                rr_detrend,
                fs=self.interpolation_rate,
                nperseg=nperseg,
                noverlap=nperseg // 2,
                nfft=max(512, nperseg * 2)
            )
            
            # 计算各频带功率
            total_power = 0
            for band_name, (f_low, f_high) in self.FREQ_BANDS.items():
                band_mask = (freqs >= f_low) & (freqs < f_high)
                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                features[f'{band_name.lower()}_power'] = band_power
                total_power += band_power
            
            features['total_power'] = total_power
            
            # 归一化功率 (nu = normalized units)
            lf_hf_total = features['lf_power'] + features['hf_power']
            if lf_hf_total > 0:
                features['lf_nu'] = features['lf_power'] / lf_hf_total * 100
                features['hf_nu'] = features['hf_power'] / lf_hf_total * 100
            else:
                features['lf_nu'] = 0
                features['hf_nu'] = 0
            
            # LF/HF比值 (交感/副交感平衡)
            if features['hf_power'] > 0:
                features['lf_hf_ratio'] = features['lf_power'] / features['hf_power']
            else:
                features['lf_hf_ratio'] = 0
            
            # 峰值频率
            for band_name, (f_low, f_high) in self.FREQ_BANDS.items():
                band_mask = (freqs >= f_low) & (freqs < f_high)
                if np.any(band_mask) and np.any(psd[band_mask]):
                    peak_idx = np.argmax(psd[band_mask])
                    features[f'{band_name.lower()}_peak_freq'] = freqs[band_mask][peak_idx]
                else:
                    features[f'{band_name.lower()}_peak_freq'] = 0
            
        except Exception as e:
            logger.error(f"频域分析失败: {e}")
            return self._empty_freq_features()
        
        return features
    
    def _empty_freq_features(self) -> Dict[str, float]:
        """返回空的频域特征"""
        return {
            'vlf_power': 0, 'lf_power': 0, 'hf_power': 0,
            'total_power': 0, 'lf_nu': 0, 'hf_nu': 0,
            'lf_hf_ratio': 0,
            'vlf_peak_freq': 0, 'lf_peak_freq': 0, 'hf_peak_freq': 0
        }
    
    def extract_nonlinear(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        提取非线性HRV特征
        
        非线性特征捕捉RR间期序列的复杂动态特性:
        
        1. Poincaré图分析:
           - SD1: 短期变异性 (垂直于对角线)
           - SD2: 长期变异性 (沿对角线)
           - SD1/SD2比值: 反映随机性与确定性成分比例
        
        2. 熵指标:
           - 近似熵 (ApEn): 序列规律性
           - 样本熵 (SampEn): 改进的近似熵
        
        3. 去趋势波动分析 (DFA):
           - α1: 短期波动指数 (4-16拍)
           - α2: 长期波动指数 (16-64拍)
        
        Args:
            rr_intervals: RR间期序列 (ms)
            
        Returns:
            非线性特征字典
        """
        features = {}
        
        # Poincaré图分析
        poincare = self._poincare_analysis(rr_intervals)
        features.update(poincare)
        
        # 样本熵
        features['sample_entropy'] = self._sample_entropy(rr_intervals)
        
        # 近似熵
        features['approx_entropy'] = self._approximate_entropy(rr_intervals)
        
        # DFA
        dfa = self._dfa_analysis(rr_intervals)
        features.update(dfa)
        
        return features
    
    def _poincare_analysis(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Poincaré图分析
        
        将RR(n)与RR(n+1)绘制为散点图
        通过拟合椭圆分析短期和长期变异性
        
        SD1 = √(0.5 · Var(RR(n+1) - RR(n)))
        SD2 = √(2 · SDNN² - 0.5 · SDSD²)
        
        Args:
            rr_intervals: RR间期序列
            
        Returns:
            Poincaré特征字典
        """
        if len(rr_intervals) < 3:
            return {'sd1': 0, 'sd2': 0, 'sd1_sd2_ratio': 0, 'ellipse_area': 0}
        
        # RR(n) vs RR(n+1)
        rr_n = rr_intervals[:-1]
        rr_n1 = rr_intervals[1:]
        
        # 差分
        diff = rr_n1 - rr_n
        
        # SD1: 短期变异性 (Poincaré图垂直于对角线的标准差)
        sd1 = np.sqrt(np.var(diff) / 2)
        
        # SD2: 长期变异性 (Poincaré图沿对角线的标准差)
        # SD2² = 2·SDNN² - SD1²
        sdnn = np.std(rr_intervals, ddof=1)
        sd2 = np.sqrt(2 * sdnn**2 - sd1**2) if 2 * sdnn**2 > sd1**2 else 0
        
        # SD1/SD2比值
        sd1_sd2 = sd1 / sd2 if sd2 > 0 else 0
        
        # 椭圆面积
        ellipse_area = np.pi * sd1 * sd2
        
        return {
            'sd1': sd1,
            'sd2': sd2,
            'sd1_sd2_ratio': sd1_sd2,
            'ellipse_area': ellipse_area
        }
    
    def _sample_entropy(
        self,
        rr_intervals: np.ndarray,
        m: int = 2,
        r: float = None
    ) -> float:
        """
        计算样本熵 (Sample Entropy)
        
        SampEn衡量序列的不规则性/复杂性
        值越低表示越规则，值越高表示越随机
        
        算法:
        1. 形成m维向量: X(i) = [RR(i), RR(i+1), ..., RR(i+m-1)]
        2. 计算相似向量对数: C_m(r)
        3. SampEn = -ln(C_{m+1}(r) / C_m(r))
        
        Args:
            rr_intervals: RR间期序列
            m: 嵌入维度 (默认2)
            r: 相似性阈值 (默认0.2*std)
            
        Returns:
            样本熵值
        """
        n = len(rr_intervals)
        
        if n < m + 2:
            return 0.0
        
        if r is None:
            r = 0.2 * np.std(rr_intervals)
        
        def _count_matches(m_dim):
            """计算m维向量的相似对数"""
            count = 0
            templates = []
            
            for i in range(n - m_dim):
                templates.append(rr_intervals[i:i + m_dim])
            
            for i in range(len(templates)):
                for j in range(i + 1, len(templates)):
                    dist = np.max(np.abs(templates[i] - templates[j]))
                    if dist < r:
                        count += 2  # 对称计数
            
            return count / (2 * len(templates) * (len(templates) - 1)) if len(templates) > 1 else 0
        
        phi_m = _count_matches(m)
        phi_m1 = _count_matches(m + 1)
        
        if phi_m > 0 and phi_m1 > 0:
            return -np.log(phi_m1 / phi_m)
        else:
            return 0.0
    
    def _approximate_entropy(
        self,
        rr_intervals: np.ndarray,
        m: int = 2,
        r: float = None
    ) -> float:
        """
        计算近似熵 (Approximate Entropy)
        
        与样本熵类似，但包含自匹配
        计算更稳定，但有偏差
        
        Args:
            rr_intervals: RR间期序列
            m: 嵌入维度
            r: 相似性阈值
            
        Returns:
            近似熵值
        """
        n = len(rr_intervals)
        
        if n < m + 2:
            return 0.0
        
        if r is None:
            r = 0.2 * np.std(rr_intervals)
        
        def _phi(m_dim):
            """计算phi值"""
            templates = []
            for i in range(n - m_dim + 1):
                templates.append(rr_intervals[i:i + m_dim])
            
            c_values = []
            for i in range(len(templates)):
                count = 0
                for j in range(len(templates)):
                    dist = np.max(np.abs(templates[i] - templates[j]))
                    if dist < r:
                        count += 1
                c_values.append(count / len(templates))
            
            return np.mean(np.log(np.array(c_values) + 1e-10))
        
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        
        return phi_m - phi_m1
    
    def _dfa_analysis(
        self,
        rr_intervals: np.ndarray,
        short_range: Tuple[int, int] = (4, 16),
        long_range: Tuple[int, int] = (16, 64)
    ) -> Dict[str, float]:
        """
        去趋势波动分析 (Detrended Fluctuation Analysis)
        
        DFA量化RR间期序列的分形特性
        
        算法:
        1. 累积和: y(k) = Σ(RR(i) - mean(RR))
        2. 分段线性拟合，计算波动函数F(n)
        3. 对数回归: log(F(n)) ~ α·log(n)
        
        α1 (短期, 4-16拍): 反映短期相关性
        α2 (长期, 16-64拍): 反映长期相关性
        
        解释:
        - α ≈ 0.5: 白噪声 (无相关)
        - α ≈ 1.0: 1/f噪声 (长程相关)
        - α ≈ 1.5: 布朗运动
        
        Args:
            rr_intervals: RR间期序列
            short_range: 短期分析范围
            long_range: 长期分析范围
            
        Returns:
            DFA特征字典
        """
        n = len(rr_intervals)
        
        if n < long_range[1]:
            return {'dfa_alpha1': 0, 'dfa_alpha2': 0}
        
        # 累积和 (积分)
        y = np.cumsum(rr_intervals - np.mean(rr_intervals))
        
        # 分析的窗口大小
        scales = np.arange(short_range[0], min(long_range[1], n // 4) + 1)
        
        fluctuations = []
        
        for scale in scales:
            # 分段数
            n_segments = n // scale
            
            if n_segments < 1:
                continue
            
            f_n = []
            
            for i in range(n_segments):
                start = i * scale
                end = start + scale
                segment = y[start:end]
                
                # 线性拟合
                x = np.arange(scale)
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                
                # 残差方差
                residuals = segment - trend
                f_n.append(np.var(residuals))
            
            # 平均波动
            fluctuations.append(np.sqrt(np.mean(f_n)))
        
        scales = scales[:len(fluctuations)]
        fluctuations = np.array(fluctuations)
        
        # 对数回归
        log_scales = np.log(scales)
        log_fluct = np.log(fluctuations + 1e-10)
        
        # 短期 α1
        short_mask = (scales >= short_range[0]) & (scales <= short_range[1])
        if np.sum(short_mask) >= 2:
            alpha1, _ = np.polyfit(log_scales[short_mask], log_fluct[short_mask], 1)
        else:
            alpha1 = 0
        
        # 长期 α2
        long_mask = (scales >= long_range[0]) & (scales <= long_range[1])
        if np.sum(long_mask) >= 2:
            alpha2, _ = np.polyfit(log_scales[long_mask], log_fluct[long_mask], 1)
        else:
            alpha2 = 0
        
        return {
            'dfa_alpha1': alpha1,
            'dfa_alpha2': alpha2
        }


class HRVSegmentAnalyzer:
    """
    分段HRV分析器
    
    将长记录分割为短段进行分析
    用于评估HRV的时间变化
    """
    
    def __init__(
        self,
        segment_length: float = 300.0,  # 5分钟
        overlap: float = 0.5,
        sampling_rate: float = 200.0
    ):
        self.segment_length = segment_length
        self.overlap = overlap
        self.hrv_extractor = HRVFeatureExtractor(sampling_rate)
    
    def analyze_segments(
        self,
        r_peaks: np.ndarray,
        sampling_rate: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        分段分析
        
        Args:
            r_peaks: R峰位置
            sampling_rate: 采样率
            
        Returns:
            segment_times: 每段的中心时间
            segment_features: 每段的特征矩阵
        """
        # 计算时间
        times = r_peaks / sampling_rate
        total_time = times[-1]
        
        # 分段
        step = self.segment_length * (1 - self.overlap)
        segment_starts = np.arange(0, total_time - self.segment_length, step)
        
        all_features = []
        segment_times = []
        
        for start in segment_starts:
            end = start + self.segment_length
            mask = (times >= start) & (times < end)
            segment_peaks = r_peaks[mask]
            
            if len(segment_peaks) > 20:  # 至少20个心拍
                features = self.hrv_extractor.extract_all(segment_peaks, include_nonlinear=False)
                if features:
                    all_features.append(list(features.values()))
                    segment_times.append((start + end) / 2)
        
        return np.array(segment_times), np.array(all_features)
