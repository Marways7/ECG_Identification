"""
数据加载模块
============

加载和预处理ECG数据文件
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger


class ECGDataLoader:
    """
    ECG数据加载器
    
    从CSV文件加载ECG数据
    """
    
    def __init__(self, data_dir: str):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.data_files = {}
        self.raw_data = {}
        
        self._scan_files()
    
    def _scan_files(self):
        """扫描数据目录"""
        if not os.path.exists(self.data_dir):
            logger.error(f"数据目录不存在: {self.data_dir}")
            return
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv'):
                # 解析文件名 (如 A1_processed.csv)
                name = filename.split('_')[0]
                subject_id = name[0]  # A, B, C, ...
                
                self.data_files[name] = os.path.join(self.data_dir, filename)
        
        logger.info(f"发现 {len(self.data_files)} 个数据文件")
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        加载所有数据文件
        
        Returns:
            数据字典 {subject_id: DataFrame}
        """
        for name, filepath in self.data_files.items():
            try:
                df = pd.read_csv(filepath)
                self.raw_data[name] = df
                logger.info(f"加载 {name}: {len(df)} 行")
            except Exception as e:
                logger.error(f"加载 {name} 失败: {e}")
        
        return self.raw_data
    
    def load_subject(self, subject_id: str) -> Optional[pd.DataFrame]:
        """
        加载单个被试的数据
        
        Args:
            subject_id: 被试ID (如 'A1')
            
        Returns:
            数据DataFrame
        """
        if subject_id in self.raw_data:
            return self.raw_data[subject_id]
        
        if subject_id in self.data_files:
            df = pd.read_csv(self.data_files[subject_id])
            self.raw_data[subject_id] = df
            return df
        
        logger.warning(f"未找到被试 {subject_id} 的数据")
        return None
    
    def get_ecg_signal(
        self,
        subject_id: str,
        channel: str = 'Channel 1'
    ) -> Optional[np.ndarray]:
        """
        获取ECG信号
        
        Args:
            subject_id: 被试ID
            channel: 通道名称
            
        Returns:
            ECG信号数组
        """
        df = self.load_subject(subject_id)
        if df is None or channel not in df.columns:
            return None
        
        return df[channel].values
    
    def estimate_sampling_rate(self, subject_id: str) -> float:
        """
        估计采样率
        
        Args:
            subject_id: 被试ID
            
        Returns:
            估计的采样率 (Hz)
        """
        df = self.load_subject(subject_id)
        if df is None:
            return 200.0  # 默认值
        
        timestamps = df['timestamp'].values
        time_span = timestamps.max() - timestamps.min() + 1
        n_samples = len(df)
        
        return n_samples / time_span
    
    def get_subjects(self) -> List[str]:
        """获取所有被试ID列表"""
        return list(self.data_files.keys())
    
    def prepare_training_data(
        self,
        sampling_rate: float = 200.0
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备训练数据
        
        返回所有被试的ECG信号和标签
        
        Args:
            sampling_rate: 目标采样率
            
        Returns:
            signals: 信号列表
            labels: 标签数组
            subject_ids: 被试ID列表
        """
        self.load_all()
        
        signals = []
        labels = []
        subject_ids = []
        
        for name, df in self.raw_data.items():
            subject_id = name[0]  # 取首字母作为身份标签
            signal = df['Channel 1'].values
            
            # 重采样到目标采样率
            actual_fs = self.estimate_sampling_rate(name)
            if abs(actual_fs - sampling_rate) > 10:
                from scipy import signal as sig
                n_samples = int(len(signal) * sampling_rate / actual_fs)
                signal = sig.resample(signal, n_samples)
            
            signals.append(signal)
            labels.append(subject_id)
            subject_ids.append(name)
        
        return signals, np.array(labels), subject_ids
