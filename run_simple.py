#!/usr/bin/env python3
"""
ECG身份识别系统 - 简化运行版本
==============================

使用传统机器学习方法（不依赖PyTorch）
"""

import os
import sys
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
    level="DEBUG"
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_loader import ECGDataLoader
from src.preprocessing.signal_pipeline import SignalPreprocessor, PreprocessingConfig

# 导入机器学习库（不依赖PyTorch）
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb


def load_and_preprocess():
    """加载和预处理数据"""
    logger.info("=" * 60)
    logger.info("阶段1: 数据加载与预处理")
    logger.info("=" * 60)
    
    loader = ECGDataLoader('ECG_Data')
    signals, labels, subject_ids = loader.prepare_training_data(sampling_rate=250.0)
    
    logger.info(f"加载了 {len(signals)} 个被试的数据")
    
    # 预处理配置 (250Hz采样率)
    config = PreprocessingConfig(
        sampling_rate=250.0,
        wavelet='db4',
        wavelet_mode='soft',
        baseline_method='morphological',
        rpeak_method='pan_tompkins',
        beat_pre_r=0.25,
        beat_post_r=0.45,
        beat_target_length=175  # 0.7s * 250Hz
    )
    
    preprocessor = SignalPreprocessor(config)
    
    all_beats = []
    all_labels = []
    
    for signal, label, subject_id in zip(signals, labels, subject_ids):
        logger.info(f"处理 {subject_id} ({label})...")
        
        try:
            result = preprocessor.process(signal)
            
            if result.is_valid() and len(result.beats) > 30:
                all_beats.append(result.beats)
                all_labels.extend([label] * len(result.beats))
                
                logger.info(f"  {len(result.beats)} 心拍, HR={result.heart_rate:.1f} BPM")
            else:
                logger.warning(f"  {subject_id} 数据不足，跳过")
                
        except Exception as e:
            logger.error(f"  处理 {subject_id} 失败: {e}")
    
    if not all_beats:
        raise ValueError("未找到可用心拍数据，请检查数据目录或预处理配置。")
    all_beats = np.vstack(all_beats)
    all_labels = np.array(all_labels)
    
    logger.info(f"预处理完成: {len(all_beats)} 心拍, {len(np.unique(all_labels))} 类别")
    
    return all_beats, all_labels


def extract_simple_features(beats):
    """提取简单的心拍特征"""
    logger.info("提取心拍特征...")
    
    features = []
    for beat in beats:
        n = len(beat)
        r_pos = n // 2
        
        # 基本统计特征
        feat = [
            np.mean(beat),
            np.std(beat),
            np.max(beat),
            np.min(beat),
            np.max(beat) - np.min(beat),
            beat[r_pos],  # R峰幅度
            np.sum(beat ** 2),  # 能量
            np.sum(np.abs(np.diff(beat))),  # 曲线长度
        ]
        
        # QRS区域特征
        qrs_half = int(0.04 * 250)  # 40ms
        qrs = beat[r_pos-qrs_half:r_pos+qrs_half]
        feat.extend([
            np.max(qrs) - np.min(qrs),
            np.sum(qrs ** 2),
        ])
        
        # 波形形状特征
        feat.extend([
            np.percentile(beat, 25),
            np.percentile(beat, 50),
            np.percentile(beat, 75),
            np.sum(np.diff(np.sign(beat - np.mean(beat))) != 0),  # 过零率
        ])
        
        # 小波能量特征
        import pywt
        coeffs = pywt.wavedec(beat, 'db4', level=4)
        for c in coeffs:
            feat.append(np.sum(c ** 2))
        
        features.append(feat)
    
    return np.array(features)


def train_ensemble(X, y):
    """训练集成分类器"""
    logger.info("=" * 60)
    logger.info("阶段2: 模型训练")
    logger.info("=" * 60)
    
    # 编码标签
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 处理NaN
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    logger.info(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
    
    # 定义分类器
    classifiers = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            use_label_encoder=False, eval_metric='mlogloss'
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
        ),
        'SVM': SVC(C=10, kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    }
    
    # 交叉验证评估
    logger.info("\n交叉验证结果:")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = {}
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
        cv_results[name] = {'mean': scores.mean(), 'std': scores.std()}
        logger.info(f"  {name}: {scores.mean():.4f} ± {scores.std():.4f}")
    
    # 训练投票分类器
    logger.info("\n训练集成分类器...")
    
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', classifiers['RandomForest']),
            ('xgb', classifiers['XGBoost']),
            ('lgb', classifiers['LightGBM']),
            ('svm', classifiers['SVM'])
        ],
        voting='soft',
        n_jobs=-1
    )
    
    voting_clf.fit(X_train, y_train)
    
    # 测试集评估
    y_pred = voting_clf.predict(X_test)
    y_pred_proba = voting_clf.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"\n测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    labels = le.classes_
    
    logger.info("\n混淆矩阵:")
    header = "     " + "  ".join([f"{l:>5}" for l in labels])
    logger.info(header)
    for i, row in enumerate(cm):
        row_str = f"{labels[i]:>4} " + "  ".join([f"{v:>5}" for v in row])
        logger.info(row_str)
    
    # 分类报告
    logger.info("\n分类报告:")
    report = classification_report(y_test, y_pred, target_names=labels)
    report_dict = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    for line in report.split('\n'):
        if line.strip():
            logger.info(line)
    
    # 收集所有结果
    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'labels': labels,
        'cv_results': cv_results,
        'classification_report': report_dict,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return voting_clf, scaler, le, results


def save_results(beats, labels, features, model, scaler, le, results, preprocess_info):
    """保存所有结果"""
    import joblib
    import json
    
    # 创建输出目录
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    logger.info("\n保存结果...")
    
    # 1. 保存模型
    model_path = os.path.join(output_dir, f"model_{timestamp}.joblib")
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'label_encoder': le
    }, model_path)
    logger.info(f"  模型已保存: {model_path}")
    
    # 2. 保存心拍数据
    beats_path = os.path.join(output_dir, f"beats_{timestamp}.npz")
    np.savez_compressed(beats_path, beats=beats, labels=labels, features=features)
    logger.info(f"  心拍数据已保存: {beats_path}")
    
    # 3. 保存评估结果
    eval_results = {
        'timestamp': timestamp,
        'accuracy': float(results['accuracy']),
        'labels': results['labels'].tolist(),
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'cv_results': {k: {'mean': float(v['mean']), 'std': float(v['std'])} 
                       for k, v in results['cv_results'].items()},
        'classification_report': results['classification_report'],
        'preprocess_info': preprocess_info,
        'total_beats': len(beats),
        'num_subjects': len(np.unique(labels))
    }
    
    results_path = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    logger.info(f"  评估结果已保存: {results_path}")
    
    # 4. 保存最新结果的软链接/复制
    latest_model = os.path.join(output_dir, "model_latest.joblib")
    latest_results = os.path.join(output_dir, "results_latest.json")
    latest_beats = os.path.join(output_dir, "beats_latest.npz")
    
    # 复制为latest版本
    import shutil
    shutil.copy(model_path, latest_model)
    shutil.copy(results_path, latest_results)
    shutil.copy(beats_path, latest_beats)
    
    logger.info(f"\n📁 所有文件已保存到 {output_dir}/ 目录")
    
    return output_dir


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("ECG身份识别系统 - 简化版本")
    logger.info(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("采样率: 250Hz")
    logger.info("=" * 60)
    
    # 1. 加载和预处理
    beats, labels = load_and_preprocess()
    
    # 收集预处理信息
    unique_labels, counts = np.unique(labels, return_counts=True)
    preprocess_info = {
        'subjects': {str(label): int(count) for label, count in zip(unique_labels, counts)},
        'sampling_rate': 250.0,
        'beat_length': beats.shape[1] if len(beats) > 0 else 0
    }
    
    # 2. 特征提取
    features = extract_simple_features(beats)
    logger.info(f"特征维度: {features.shape}")
    
    # 3. 训练模型
    model, scaler, le, results = train_ensemble(features, labels)
    
    # 4. 保存结果
    output_dir = save_results(beats, labels, features, model, scaler, le, results, preprocess_info)
    
    # 5. 打印最终结果
    accuracy = results['accuracy']
    logger.info("\n" + "=" * 60)
    logger.info("最终结果")
    logger.info("=" * 60)
    logger.info(f"✅ 准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"✅ 被试数量: {len(np.unique(labels))}")
    logger.info(f"✅ 总心拍数: {len(beats)}")
    logger.info(f"✅ 结果目录: {output_dir}/")
    
    return accuracy


if __name__ == '__main__':
    main()
