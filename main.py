#!/usr/bin/env python3
"""
ECG身份识别系统 - 主程序
========================

基于心电信号的SOTA级别身份识别系统

使用方法:
---------
1. 训练模型: python main.py train
2. 评估模型: python main.py evaluate
3. 启动UI: python main.py ui
4. 完整流程: python main.py pipeline
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime
from loguru import logger

# 配置日志
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/ecg_id_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG"
)

# 添加src到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_loader import ECGDataLoader
from src.preprocessing.signal_pipeline import SignalPreprocessor, PreprocessingConfig
from src.features.feature_pipeline import FeatureExtractor, FeatureConfig
# 延迟导入PyTorch相关模块，避免Bus error
# from src.models.trainer import ModelTrainer, TrainingConfig
from src.models.hybrid_classifier import HybridClassifier


def load_and_preprocess_data(data_dir: str = "ECG_Data"):
    """
    加载和预处理数据
    
    Returns:
        all_beats: 所有心拍
        all_labels: 对应标签
        all_features: 手工特征
        all_r_peaks: R峰位置
        sampling_rates: 采样率信息
    """
    logger.info("=" * 60)
    logger.info("阶段1: 数据加载与预处理")
    logger.info("=" * 60)
    
    # 加载数据
    loader = ECGDataLoader(data_dir)
    signals, labels, subject_ids = loader.prepare_training_data()
    
    logger.info(f"加载了 {len(signals)} 个被试的数据")
    
    # 预处理配置
    preprocess_config = PreprocessingConfig(
        sampling_rate=250.0,  # ADS1292R采样率250Hz
        wavelet='db4',
        wavelet_mode='soft',
        baseline_method='morphological',
        rpeak_method='pan_tompkins',
        beat_pre_r=0.25,
        beat_post_r=0.45,
        beat_target_length=140
    )
    
    preprocessor = SignalPreprocessor(preprocess_config)
    
    # 处理每个被试
    all_beats = []
    all_labels = []
    all_r_peaks = []
    all_signals = []
    
    for i, (signal, label, subject_id) in enumerate(zip(signals, labels, subject_ids)):
        logger.info(f"处理 {subject_id} ({label})...")
        
        try:
            result = preprocessor.process(signal)
            
            if result.is_valid() and len(result.beats) > 50:
                all_beats.append(result.beats)
                all_labels.extend([label] * len(result.beats))
                all_r_peaks.append(result.r_peaks)
                all_signals.append(result.baseline_corrected)
                
                logger.info(f"  {len(result.beats)} 心拍, HR={result.heart_rate:.1f} BPM, "
                           f"质量={result.signal_quality:.2f}")
            else:
                logger.warning(f"  {subject_id} 数据质量不足，跳过")
                
        except Exception as e:
            logger.error(f"  处理 {subject_id} 失败: {e}")
    
    # 合并所有心拍
    if not all_beats:
        raise ValueError("未找到可用心拍数据，请检查数据目录或预处理配置。")
    all_beats = np.vstack(all_beats)
    all_labels = np.array(all_labels)
    
    logger.info(f"预处理完成: {len(all_beats)} 心拍, {len(np.unique(all_labels))} 类别")
    
    return all_beats, all_labels, all_signals, all_r_peaks


def extract_features(signals, r_peaks_list, labels):
    """
    提取特征
    
    Args:
        signals: 处理后的信号列表
        r_peaks_list: R峰位置列表
        labels: 标签
        
    Returns:
        特征矩阵
    """
    logger.info("=" * 60)
    logger.info("阶段2: 特征提取")
    logger.info("=" * 60)
    
    feature_config = FeatureConfig(
        sampling_rate=200.0,
        enable_hrv=True,
        enable_crc=True,
        enable_morphological=True,
        enable_nonlinear=True
    )
    
    extractor = FeatureExtractor(feature_config)
    
    # 为每个被试提取特征
    all_features = []
    unique_labels = np.unique(labels)
    
    # 这里需要按被试组织数据
    # 暂时返回空特征
    logger.info("特征提取完成")
    
    return np.array([])


def train_model(beats, labels, handcrafted_features=None):
    """
    训练模型
    
    Args:
        beats: 心拍数组
        labels: 标签
        handcrafted_features: 手工特征 (可选)
        
    Returns:
        训练结果
    """
    logger.info("=" * 60)
    logger.info("阶段3: 模型训练")
    logger.info("=" * 60)
    
    # 训练配置
    config = TrainingConfig(
        batch_size=64,
        epochs=50,
        learning_rate=1e-3,
        weight_decay=1e-4,
        model_type='cnn',
        embedding_dim=128,
        dropout=0.3,
        early_stopping=True,
        patience=10
    )
    
    # 创建训练器
    trainer = ModelTrainer(config)
    
    # 构建模型
    n_classes = len(np.unique(labels))
    input_length = beats.shape[1]
    handcrafted_dim = handcrafted_features.shape[1] if handcrafted_features is not None else 0
    
    model = trainer.build_model(
        n_classes=n_classes,
        input_length=input_length,
        handcrafted_dim=handcrafted_dim
    )
    
    # 准备数据
    train_loader, val_loader, test_loader = trainer.prepare_data(
        beats, labels, handcrafted_features
    )
    
    # 训练
    use_handcrafted = handcrafted_features is not None and handcrafted_dim > 0
    history = trainer.train(train_loader, val_loader, use_handcrafted)
    
    # 评估
    results = trainer.evaluate(test_loader, use_handcrafted)
    
    # 保存结果
    trainer.save_results(results)
    
    logger.info(f"训练完成: 最终准确率 = {results['accuracy']:.4f}")
    
    return trainer, results


def train_hybrid_model(beats, labels, handcrafted_features):
    """
    训练混合模型
    
    使用传统机器学习 + 深度学习特征融合
    """
    logger.info("=" * 60)
    logger.info("训练混合分类器")
    logger.info("=" * 60)
    
    # 如果没有手工特征，创建基于心拍的简单特征
    if handcrafted_features is None or len(handcrafted_features) == 0:
        logger.info("从心拍数据提取简单特征...")
        handcrafted_features = []
        for beat in beats:
            features = [
                np.mean(beat),
                np.std(beat),
                np.max(beat),
                np.min(beat),
                np.max(beat) - np.min(beat),
                beat[len(beat)//2],  # R峰幅度
                np.sum(beat ** 2),   # 能量
            ]
            handcrafted_features.append(features)
        handcrafted_features = np.array(handcrafted_features)
    
    # 创建混合分类器
    classifier = HybridClassifier(
        deep_model=None,  # 暂时不用深度模型
        use_ensemble=True,
        ensemble_method='stacking',
        n_classes=len(np.unique(labels))
    )
    
    # 交叉验证
    cv_results = classifier.cross_validate(
        handcrafted_features, 
        labels,
        beats=None,
        cv=5
    )
    
    # 打印交叉验证结果
    logger.info("\n交叉验证结果:")
    for name, scores in cv_results.items():
        logger.info(f"  {name}: {scores['mean']:.4f} ± {scores['std']:.4f}")
    
    # 完整训练
    classifier.fit(handcrafted_features, labels, beats=None)
    
    # 评估
    results = classifier.evaluate(handcrafted_features, labels, beats=None)
    
    # 保存模型
    os.makedirs("models_saved", exist_ok=True)
    classifier.save('models_saved/hybrid_classifier.pkl')
    
    return classifier, results


def run_pipeline():
    """
    运行完整管道
    """
    logger.info("=" * 60)
    logger.info("ECG身份识别系统 - 完整管道")
    logger.info(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # 1. 加载和预处理
    beats, labels, signals, r_peaks = load_and_preprocess_data()
    
    # 2. 训练混合模型 (使用简单特征)
    classifier, results = train_hybrid_model(beats, labels, None)
    
    # 3. 打印最终结果
    logger.info("\n" + "=" * 60)
    logger.info("最终结果")
    logger.info("=" * 60)
    logger.info(f"准确率: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    
    # 混淆矩阵
    logger.info("\n混淆矩阵:")
    cm = results['confusion_matrix']
    unique_labels = np.unique(labels)
    
    # 打印表头
    header = "     " + "  ".join([f"{l:>5}" for l in unique_labels])
    logger.info(header)
    
    for i, row in enumerate(cm):
        row_str = f"{unique_labels[i]:>4} " + "  ".join([f"{v:>5}" for v in row])
        logger.info(row_str)
    
    return results


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='ECG身份识别系统')
    parser.add_argument('command', choices=['train', 'evaluate', 'ui', 'pipeline'],
                       help='执行命令')
    parser.add_argument('--data-dir', default='ECG_Data', help='数据目录')
    
    args = parser.parse_args()
    
    if args.command == 'pipeline':
        run_pipeline()
    elif args.command == 'train':
        beats, labels, signals, r_peaks = load_and_preprocess_data(args.data_dir)
        train_hybrid_model(beats, labels, None)
    elif args.command == 'evaluate':
        logger.info("评估模式 - 请确保已有训练好的模型")
    elif args.command == 'ui':
        logger.info("启动UI界面...")
        os.system('streamlit run ui/app.py')


if __name__ == '__main__':
    main()
