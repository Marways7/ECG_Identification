"""
特征提取管道
============

整合所有特征提取模块的统一接口
提供端到端的特征提取流程
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from loguru import logger

from .hrv_features import HRVFeatureExtractor
from .crc_features import CRCFeatureExtractor
from .morphological_features import MorphologicalFeatureExtractor


@dataclass
class FeatureConfig:
    """特征提取配置"""
    
    # 采样率 (ADS1292R默认250Hz)
    sampling_rate: float = 250.0
    
    # 特征类别开关
    enable_hrv: bool = True
    enable_crc: bool = True
    enable_morphological: bool = True
    enable_nonlinear: bool = True
    
    # HRV参数
    hrv_interpolation_rate: float = 4.0
    
    # CRC参数
    crc_resp_rate_range: Tuple[float, float] = (0.1, 0.5)


@dataclass
class FeatureResult:
    """特征提取结果"""
    
    # 特征向量
    features: Dict[str, float] = field(default_factory=dict)
    
    # 特征分组
    hrv_features: Dict[str, float] = field(default_factory=dict)
    crc_features: Dict[str, float] = field(default_factory=dict)
    morph_features: Dict[str, float] = field(default_factory=dict)
    
    # 元数据
    n_beats: int = 0
    heart_rate: float = 0.0
    feature_names: List[str] = field(default_factory=list)
    
    def to_vector(self) -> np.ndarray:
        """转换为特征向量"""
        return np.array(list(self.features.values()))
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        return pd.DataFrame([self.features])


class FeatureExtractor:
    """
    特征提取器
    
    整合HRV、CRC和形态学特征提取
    提供统一的特征提取接口
    
    Usage:
        extractor = FeatureExtractor(config)
        result = extractor.extract(beats, r_peaks, ecg_signal)
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        初始化特征提取器
        
        Args:
            config: 特征提取配置
        """
        self.config = config if config else FeatureConfig()
        
        # 初始化各特征提取器
        self.hrv_extractor = HRVFeatureExtractor(
            sampling_rate=self.config.sampling_rate,
            interpolation_rate=self.config.hrv_interpolation_rate
        )
        
        self.crc_extractor = CRCFeatureExtractor(
            sampling_rate=self.config.sampling_rate,
            resp_rate_range=self.config.crc_resp_rate_range
        )
        
        self.morph_extractor = MorphologicalFeatureExtractor(
            sampling_rate=self.config.sampling_rate
        )
        
        logger.info("特征提取器初始化完成")
    
    def extract(
        self,
        beats: np.ndarray,
        r_peaks: np.ndarray,
        ecg_signal: np.ndarray,
        resp_signal: Optional[np.ndarray] = None
    ) -> FeatureResult:
        """
        提取所有特征
        
        Args:
            beats: 心拍数组 (n_beats, beat_length)
            r_peaks: R峰位置索引
            ecg_signal: 完整ECG信号
            resp_signal: 呼吸信号 (可选)
            
        Returns:
            FeatureResult对象
        """
        result = FeatureResult()
        result.n_beats = len(beats)
        
        all_features = {}
        
        try:
            # 1. HRV特征
            if self.config.enable_hrv and len(r_peaks) > 10:
                logger.debug("提取HRV特征...")
                hrv_features = self.hrv_extractor.extract_all(
                    r_peaks,
                    include_nonlinear=self.config.enable_nonlinear
                )
                result.hrv_features = hrv_features
                all_features.update({f'hrv_{k}': v for k, v in hrv_features.items()})
                
                # 记录心率
                result.heart_rate = hrv_features.get('hr_mean', 0)
            
            # 2. CRC特征
            if self.config.enable_crc and len(r_peaks) > 30:
                logger.debug("提取CRC特征...")
                crc_features = self.crc_extractor.extract_all(
                    ecg_signal, r_peaks, resp_signal
                )
                result.crc_features = crc_features
                all_features.update({f'crc_{k}': v for k, v in crc_features.items()})
            
            # 3. 形态学特征
            if self.config.enable_morphological and len(beats) > 0:
                logger.debug("提取形态学特征...")
                morph_features = self.morph_extractor.extract_all(beats, ecg_signal)
                result.morph_features = morph_features
                all_features.update({f'morph_{k}': v for k, v in morph_features.items()})
            
            # 处理NaN和Inf
            for key, value in all_features.items():
                if np.isnan(value) or np.isinf(value):
                    all_features[key] = 0.0
            
            result.features = all_features
            result.feature_names = list(all_features.keys())
            
            logger.info(f"特征提取完成: {len(all_features)} 个特征")
            
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            raise
        
        return result
    
    def extract_batch(
        self,
        data_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        labels: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str], List[FeatureResult]]:
        """
        批量特征提取
        
        Args:
            data_list: [(beats, r_peaks, ecg_signal), ...]
            labels: 标签列表
            
        Returns:
            feature_matrix: 特征矩阵 (n_samples, n_features)
            feature_names: 特征名称列表
            results: FeatureResult列表
        """
        all_results = []
        feature_vectors = []
        feature_names = None
        
        for i, (beats, r_peaks, ecg_signal) in enumerate(data_list):
            label = labels[i] if labels else f"Sample_{i}"
            
            try:
                result = self.extract(beats, r_peaks, ecg_signal)
                all_results.append(result)
                
                if feature_names is None:
                    feature_names = result.feature_names
                
                feature_vectors.append(result.to_vector())
                
            except Exception as e:
                logger.error(f"样本 {label} 特征提取失败: {e}")
                # 添加零向量
                if feature_names:
                    feature_vectors.append(np.zeros(len(feature_names)))
                all_results.append(FeatureResult())
        
        if not feature_vectors:
            return np.array([]), [], []
        
        feature_matrix = np.vstack(feature_vectors)
        
        return feature_matrix, feature_names if feature_names else [], all_results
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        获取特征分组信息
        
        用于特征选择和解释
        
        Returns:
            特征分组字典
        """
        return {
            'hrv_time': ['hrv_rr_mean', 'hrv_rr_std', 'hrv_sdnn', 'hrv_rmssd', 
                        'hrv_pnn50', 'hrv_pnn20', 'hrv_hr_mean', 'hrv_hr_std'],
            'hrv_frequency': ['hrv_vlf_power', 'hrv_lf_power', 'hrv_hf_power',
                             'hrv_lf_hf_ratio', 'hrv_total_power'],
            'hrv_nonlinear': ['hrv_sd1', 'hrv_sd2', 'hrv_sample_entropy',
                             'hrv_approx_entropy', 'hrv_dfa_alpha1', 'hrv_dfa_alpha2'],
            'crc': ['crc_psi', 'crc_coherence_max', 'crc_rsa_amplitude',
                   'crc_mi_cardiac_resp', 'crc_te_cardiac_to_resp'],
            'morphological': ['morph_beat_amplitude', 'morph_qrs_amplitude',
                            'morph_t_amplitude', 'morph_st_level']
        }


class FeatureSelector:
    """
    特征选择器
    
    实现多种特征选择策略
    """
    
    def __init__(self, method: str = 'mutual_info'):
        """
        初始化特征选择器
        
        Args:
            method: 选择方法 ('mutual_info', 'f_classif', 'rfe', 'lasso')
        """
        self.method = method
        self.selected_features = None
        self.feature_scores = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_features: int = 50
    ) -> np.ndarray:
        """
        拟合特征选择器
        
        Args:
            X: 特征矩阵
            y: 标签
            n_features: 选择的特征数量
            
        Returns:
            选中的特征索引
        """
        from sklearn.feature_selection import (
            mutual_info_classif, f_classif, SelectKBest, RFE
        )
        from sklearn.ensemble import RandomForestClassifier
        
        if self.method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=n_features)
            selector.fit(X, y)
            self.feature_scores = selector.scores_
            self.selected_features = np.argsort(selector.scores_)[-n_features:]
            
        elif self.method == 'f_classif':
            selector = SelectKBest(f_classif, k=n_features)
            selector.fit(X, y)
            self.feature_scores = selector.scores_
            self.selected_features = np.argsort(selector.scores_)[-n_features:]
            
        elif self.method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=n_features)
            selector.fit(X, y)
            self.selected_features = np.where(selector.support_)[0]
            self.feature_scores = selector.ranking_
            
        else:
            raise ValueError(f"不支持的方法: {self.method}")
        
        return self.selected_features
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        应用特征选择
        
        Args:
            X: 特征矩阵
            
        Returns:
            选中的特征
        """
        if self.selected_features is None:
            raise ValueError("请先调用fit方法")
        
        return X[:, self.selected_features]
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_features: int = 50
    ) -> np.ndarray:
        """
        拟合并应用特征选择
        """
        self.fit(X, y, n_features)
        return self.transform(X)


class FeatureNormalizer:
    """
    特征归一化器
    
    实现多种归一化策略
    """
    
    def __init__(self, method: str = 'standard'):
        """
        初始化归一化器
        
        Args:
            method: 归一化方法 ('standard', 'minmax', 'robust')
        """
        self.method = method
        self.scaler = None
    
    def fit(self, X: np.ndarray) -> 'FeatureNormalizer':
        """拟合归一化器"""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"不支持的方法: {self.method}")
        
        self.scaler.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """应用归一化"""
        if self.scaler is None:
            raise ValueError("请先调用fit方法")
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """拟合并应用归一化"""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """逆变换"""
        if self.scaler is None:
            raise ValueError("请先调用fit方法")
        return self.scaler.inverse_transform(X)
