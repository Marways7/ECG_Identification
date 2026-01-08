"""
心拍分割模块
============

基于R峰位置进行心拍分割
提取单个心拍周期用于形态学特征分析和模板匹配

心拍结构:
---------
P波 -> QRS波群 -> T波
- P波: R峰前约200-250ms
- QRS: R峰±50ms
- T波: R峰后约200-400ms

典型心拍窗口: R峰前200ms到R峰后400ms
"""

import numpy as np
from scipy import signal, interpolate
from typing import Tuple, List, Optional
from loguru import logger


class BeatSegmenter:
    """
    ECG心拍分割器
    
    将连续ECG信号分割为单个心拍
    支持固定窗口和自适应窗口两种模式
    
    Attributes:
        sampling_rate: 采样率 (Hz)
        pre_r: R峰前时间窗口 (秒)
        post_r: R峰后时间窗口 (秒)
        target_length: 重采样目标长度
    """
    
    def __init__(
        self,
        sampling_rate: float = 200.0,
        pre_r: float = 0.25,
        post_r: float = 0.45,
        target_length: Optional[int] = None
    ):
        """
        初始化心拍分割器
        
        Args:
            sampling_rate: 采样率
            pre_r: R峰前时间窗口 (秒)
            post_r: R峰后时间窗口 (秒)
            target_length: 归一化后的心拍长度 (采样点)
        """
        self.sampling_rate = sampling_rate
        self.pre_r = pre_r
        self.post_r = post_r
        
        # 计算默认窗口大小
        self.pre_samples = int(pre_r * sampling_rate)
        self.post_samples = int(post_r * sampling_rate)
        self.beat_length = self.pre_samples + self.post_samples
        
        # 归一化长度
        self.target_length = target_length if target_length else self.beat_length
        
        logger.info(f"初始化心拍分割器: pre={pre_r}s, post={post_r}s, "
                   f"beat_length={self.beat_length} samples")
    
    def segment(
        self,
        ecg_signal: np.ndarray,
        r_peaks: np.ndarray,
        normalize: bool = True,
        align: bool = True
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        分割心拍
        
        Args:
            ecg_signal: ECG信号
            r_peaks: R峰位置索引
            normalize: 是否归一化心拍长度
            align: 是否对齐R峰位置
            
        Returns:
            beats: 心拍数组 (n_beats, beat_length)
            beat_info: 每个心拍的信息列表
        """
        ecg_signal = np.asarray(ecg_signal, dtype=np.float64)
        r_peaks = np.asarray(r_peaks, dtype=np.int64)
        
        beats = []
        beat_info = []
        
        for i, r_peak in enumerate(r_peaks):
            # 计算窗口边界
            start = r_peak - self.pre_samples
            end = r_peak + self.post_samples
            
            # 检查边界
            if start < 0 or end > len(ecg_signal):
                logger.debug(f"心拍 {i} 超出边界，跳过")
                continue
            
            # 提取心拍
            beat = ecg_signal[start:end].copy()
            
            # 对齐R峰 (确保R峰在固定位置)
            if align:
                beat = self._align_rpeak(beat, self.pre_samples)
            
            # 归一化长度
            if normalize and len(beat) != self.target_length:
                beat = self._resample_beat(beat, self.target_length)
            
            beats.append(beat)
            
            # 记录信息
            info = {
                'index': i,
                'r_peak': r_peak,
                'start': start,
                'end': end,
                'rr_prev': r_peaks[i] - r_peaks[i-1] if i > 0 else None,
                'rr_next': r_peaks[i+1] - r_peaks[i] if i < len(r_peaks)-1 else None
            }
            beat_info.append(info)
        
        beats_array = np.array(beats) if beats else np.array([]).reshape(0, self.target_length)
        
        logger.info(f"分割得到 {len(beats)} 个心拍")
        
        return beats_array, beat_info
    
    def _align_rpeak(
        self, 
        beat: np.ndarray, 
        expected_r_pos: int
    ) -> np.ndarray:
        """
        对齐R峰位置
        
        确保心拍中R峰位于固定位置
        
        Args:
            beat: 心拍波形
            expected_r_pos: 期望的R峰位置
            
        Returns:
            对齐后的心拍
        """
        # 在期望位置附近搜索实际R峰
        search_range = int(0.05 * self.sampling_rate)  # ±50ms
        start = max(0, expected_r_pos - search_range)
        end = min(len(beat), expected_r_pos + search_range)
        
        actual_r_pos = np.argmax(beat[start:end]) + start
        
        # 计算偏移
        offset = expected_r_pos - actual_r_pos
        
        if offset != 0:
            # 平移对齐
            aligned = np.zeros_like(beat)
            if offset > 0:
                aligned[offset:] = beat[:-offset]
            else:
                aligned[:offset] = beat[-offset:]
            return aligned
        
        return beat
    
    def _resample_beat(
        self, 
        beat: np.ndarray, 
        target_length: int
    ) -> np.ndarray:
        """
        重采样心拍到目标长度
        
        使用三次样条插值
        
        Args:
            beat: 原始心拍
            target_length: 目标长度
            
        Returns:
            重采样后的心拍
        """
        original_length = len(beat)
        x_original = np.linspace(0, 1, original_length)
        x_target = np.linspace(0, 1, target_length)
        
        # 三次样条插值
        f = interpolate.interp1d(x_original, beat, kind='cubic')
        resampled = f(x_target)
        
        return resampled
    
    def segment_adaptive(
        self,
        ecg_signal: np.ndarray,
        r_peaks: np.ndarray
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        自适应窗口分割
        
        根据实际RR间期调整心拍窗口大小
        
        Args:
            ecg_signal: ECG信号
            r_peaks: R峰位置
            
        Returns:
            归一化的心拍数组和信息
        """
        ecg_signal = np.asarray(ecg_signal, dtype=np.float64)
        r_peaks = np.asarray(r_peaks, dtype=np.int64)
        
        beats = []
        beat_info = []
        
        for i in range(1, len(r_peaks) - 1):
            # 使用相邻RR间期确定窗口
            rr_prev = r_peaks[i] - r_peaks[i-1]
            rr_next = r_peaks[i+1] - r_peaks[i]
            
            # 动态窗口: 前30%到后70%的RR间期
            pre_samples = int(0.35 * rr_prev)
            post_samples = int(0.65 * rr_next)
            
            start = r_peaks[i] - pre_samples
            end = r_peaks[i] + post_samples
            
            if start < 0 or end > len(ecg_signal):
                continue
            
            beat = ecg_signal[start:end].copy()
            
            # 重采样到统一长度
            beat = self._resample_beat(beat, self.target_length)
            beats.append(beat)
            
            info = {
                'index': i,
                'r_peak': r_peaks[i],
                'start': start,
                'end': end,
                'rr_prev': rr_prev,
                'rr_next': rr_next,
                'original_length': end - start
            }
            beat_info.append(info)
        
        beats_array = np.array(beats) if beats else np.array([]).reshape(0, self.target_length)
        
        return beats_array, beat_info
    
    def get_beat_templates(
        self,
        beats: np.ndarray,
        n_clusters: int = 3,
        method: str = 'kmeans'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取心拍模板
        
        通过聚类获取代表性心拍模板
        用于模板匹配和异常检测
        
        Args:
            beats: 心拍数组
            n_clusters: 聚类数量
            method: 聚类方法 ('kmeans', 'hierarchical')
            
        Returns:
            templates: 模板数组
            labels: 每个心拍的类别标签
        """
        if len(beats) < n_clusters:
            return beats, np.arange(len(beats))
        
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(beats)
            templates = kmeans.cluster_centers_
            
        elif method == 'hierarchical':
            from sklearn.cluster import AgglomerativeClustering
            
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clustering.fit_predict(beats)
            
            # 计算每个簇的均值作为模板
            templates = []
            for i in range(n_clusters):
                cluster_beats = beats[labels == i]
                if len(cluster_beats) > 0:
                    templates.append(np.mean(cluster_beats, axis=0))
            templates = np.array(templates)
        
        else:
            raise ValueError(f"不支持的聚类方法: {method}")
        
        logger.info(f"提取 {len(templates)} 个心拍模板")
        
        return templates, labels
    
    def filter_abnormal_beats(
        self,
        beats: np.ndarray,
        beat_info: List[dict],
        threshold: float = 2.0
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        过滤异常心拍
        
        基于形态学相似性去除异常心拍 (如早搏、伪迹)
        
        Args:
            beats: 心拍数组
            beat_info: 心拍信息
            threshold: 异常阈值 (标准差倍数)
            
        Returns:
            filtered_beats: 过滤后的心拍
            filtered_info: 过滤后的信息
        """
        if len(beats) < 10:
            return beats, beat_info
        
        # 计算平均心拍模板
        mean_beat = np.mean(beats, axis=0)
        
        # 计算每个心拍与模板的相关系数
        correlations = []
        for beat in beats:
            corr = np.corrcoef(beat, mean_beat)[0, 1]
            correlations.append(corr)
        
        correlations = np.array(correlations)
        
        # 使用z-score检测异常
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        
        # 保留相关性高的心拍
        mask = correlations > (mean_corr - threshold * std_corr)
        
        filtered_beats = beats[mask]
        filtered_info = [info for info, m in zip(beat_info, mask) if m]
        
        n_removed = len(beats) - len(filtered_beats)
        logger.info(f"过滤掉 {n_removed} 个异常心拍 ({n_removed/len(beats)*100:.1f}%)")
        
        return filtered_beats, filtered_info
    
    def compute_morphological_features(
        self,
        beat: np.ndarray
    ) -> dict:
        """
        计算单个心拍的形态学特征
        
        Args:
            beat: 心拍波形
            
        Returns:
            特征字典
        """
        features = {}
        
        # R峰位置 (假设在中间)
        r_pos = len(beat) // 2
        
        # 基本统计特征
        features['amplitude'] = beat[r_pos]
        features['mean'] = np.mean(beat)
        features['std'] = np.std(beat)
        features['max'] = np.max(beat)
        features['min'] = np.min(beat)
        features['range'] = features['max'] - features['min']
        
        # QRS波群特征 (R峰附近±50ms)
        qrs_half = int(0.05 * self.sampling_rate)
        qrs_region = beat[r_pos-qrs_half:r_pos+qrs_half]
        features['qrs_amplitude'] = np.max(qrs_region) - np.min(qrs_region)
        features['qrs_duration'] = len(qrs_region) / self.sampling_rate * 1000  # ms
        
        # ST段特征 (R峰后50-150ms)
        st_start = r_pos + int(0.05 * self.sampling_rate)
        st_end = r_pos + int(0.15 * self.sampling_rate)
        if st_end < len(beat):
            st_segment = beat[st_start:st_end]
            features['st_elevation'] = np.mean(st_segment) - beat[r_pos-qrs_half]
            features['st_slope'] = np.polyfit(np.arange(len(st_segment)), st_segment, 1)[0]
        
        # T波特征 (R峰后150-350ms)
        t_start = r_pos + int(0.15 * self.sampling_rate)
        t_end = r_pos + int(0.35 * self.sampling_rate)
        if t_end < len(beat):
            t_region = beat[t_start:t_end]
            features['t_amplitude'] = np.max(t_region) - np.min(t_region)
            features['t_peak_pos'] = np.argmax(t_region) / self.sampling_rate * 1000  # ms from T start
        
        # 波形复杂度
        features['signal_energy'] = np.sum(beat ** 2)
        features['zero_crossings'] = np.sum(np.diff(np.sign(beat - np.mean(beat))) != 0)
        
        return features
    
    def extract_batch_features(
        self,
        beats: np.ndarray
    ) -> np.ndarray:
        """
        批量提取心拍形态学特征
        
        Args:
            beats: 心拍数组 (n_beats, beat_length)
            
        Returns:
            特征矩阵 (n_beats, n_features)
        """
        all_features = []
        
        for beat in beats:
            features = self.compute_morphological_features(beat)
            all_features.append(list(features.values()))
        
        return np.array(all_features)
