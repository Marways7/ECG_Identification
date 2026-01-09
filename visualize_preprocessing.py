#!/usr/bin/env python3
"""
ECG预处理可视化脚本
====================

在模型训练之前，展示完整的信号预处理流程：
1. 原始信号 vs 去噪后信号
2. 基线漂移校正前后对比
3. R峰检测结果
4. 心拍分割与叠加
5. 异常心拍检测
6. 信号质量评估
7. 频谱分析

用于答辩展示和技术报告

作者: ECG Identification System
日期: 2026-01-08
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import pywt
from scipy import signal
from scipy.fft import fft, fftfreq
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.wavelet_denoising import WaveletDenoiser
from src.preprocessing.baseline_correction import BaselineCorrector
from src.preprocessing.rpeak_detection import RPeakDetector
from src.preprocessing.beat_segmentation import BeatSegmenter

# 创建输出目录
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_dark_style():
    """设置深色主题样式"""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#0a0a0f',
        'axes.facecolor': '#0a0a0f',
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#00ffff',
        'text.color': '#e0e0e0',
        'xtick.color': '#808080',
        'ytick.color': '#808080',
        'grid.color': '#1a1a2e',
        'grid.alpha': 0.5,
        'lines.linewidth': 1.0,
        'figure.dpi': 150,
        'savefig.dpi': 200,
        'savefig.facecolor': '#0a0a0f',
        'savefig.edgecolor': '#0a0a0f'
    })


def load_ecg_data(subject='C', data_dir='ECG_Data'):
    """加载ECG数据"""
    filepath = os.path.join(data_dir, f'{subject}1_processed.csv')
    df = pd.read_csv(filepath)
    
    # Channel 1 是ECG信号
    ecg = df['Channel 1'].values.astype(np.float64)
    
    # 估算采样率
    timestamps = df['timestamp'].values
    unique_ts = np.unique(timestamps)
    if len(unique_ts) > 1:
        time_span = timestamps.max() - timestamps.min()
        fs = len(df) / time_span if time_span > 0 else 250.0
    else:
        fs = 250.0
    
    print(f"加载 {subject}1: {len(ecg)} 样本, 采样率 ≈ {fs:.1f} Hz")
    
    return ecg, fs, subject


def plot_raw_signal_overview(ecg, fs, subject, save=True):
    """绘制原始信号概览"""
    print("\n[1/8] 绘制原始信号概览...")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    # 完整信号
    t_full = np.arange(len(ecg)) / fs
    axes[0].plot(t_full, ecg, color='#00ff9f', linewidth=0.3, alpha=0.8)
    axes[0].set_title(f'Complete ECG Signal - Subject {subject}', fontsize=14, color='#00ffff')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (ADC)')
    axes[0].grid(True, alpha=0.3)
    
    # 10秒片段
    start_idx = int(10 * fs)
    end_idx = int(20 * fs)
    t_seg = np.arange(end_idx - start_idx) / fs + 10
    axes[1].plot(t_seg, ecg[start_idx:end_idx], color='#00ff9f', linewidth=0.8)
    axes[1].set_title('10-Second Segment (10s - 20s)', fontsize=14, color='#00ffff')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    # 3秒详细视图
    start_idx2 = int(15 * fs)
    end_idx2 = int(18 * fs)
    t_detail = np.arange(end_idx2 - start_idx2) / fs + 15
    axes[2].plot(t_detail, ecg[start_idx2:end_idx2], color='#00ff9f', linewidth=1.2)
    axes[2].set_title('3-Second Detail View (15s - 18s)', fontsize=14, color='#00ffff')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    
    # 标记PQRST区域（示意）
    axes[2].axhline(y=np.mean(ecg[start_idx2:end_idx2]), color='#ff00ff', 
                    linestyle='--', alpha=0.5, label='Baseline')
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(OUTPUT_DIR, f'01_raw_signal_{subject}.png')
        plt.savefig(filepath, bbox_inches='tight')
        print(f"  保存: {filepath}")
    
    plt.close()
    return fig


def plot_wavelet_denoising(ecg, fs, subject, save=True):
    """绘制小波去噪过程"""
    print("\n[2/8] 绘制小波去噪对比...")
    
    # 标准化
    ecg_norm = (ecg - np.mean(ecg)) / np.std(ecg)
    
    # 小波去噪
    denoiser = WaveletDenoiser(
        wavelet='db4',
        threshold_mode='soft',
        level=None  # 自动选择
    )
    ecg_denoised, _ = denoiser.denoise(ecg_norm)
    
    # 提取噪声
    noise = ecg_norm - ecg_denoised
    
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1])
    
    # 选择一个5秒片段
    start = int(20 * fs)
    end = int(25 * fs)
    t = np.arange(end - start) / fs
    
    # 原始信号
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, ecg_norm[start:end], color='#ff6b6b', linewidth=0.8, label='Original (Noisy)')
    ax1.set_title('Original ECG Signal (Normalized)', fontsize=14, color='#00ffff')
    ax1.set_ylabel('Amplitude')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 去噪后信号
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(t, ecg_denoised[start:end], color='#00ff9f', linewidth=0.8, label='Denoised')
    ax2.set_title('Wavelet Denoised ECG (db4, Soft Thresholding)', fontsize=14, color='#00ffff')
    ax2.set_ylabel('Amplitude')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 对比叠加
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(t, ecg_norm[start:end], color='#ff6b6b', linewidth=0.6, alpha=0.7, label='Original')
    ax3.plot(t, ecg_denoised[start:end], color='#00ff9f', linewidth=0.8, label='Denoised')
    ax3.set_title('Comparison: Original vs Denoised', fontsize=14, color='#00ffff')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 提取的噪声
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.plot(t, noise[start:end], color='#ffff00', linewidth=0.5, alpha=0.8)
    ax4.set_title('Extracted Noise', fontsize=12, color='#00ffff')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude')
    ax4.grid(True, alpha=0.3)
    
    # 噪声直方图
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.hist(noise, bins=100, color='#ffff00', alpha=0.7, density=True)
    ax5.axvline(x=0, color='#ff00ff', linestyle='--', linewidth=2)
    ax5.set_title(f'Noise Distribution (σ={np.std(noise):.4f})', fontsize=12, color='#00ffff')
    ax5.set_xlabel('Amplitude')
    ax5.set_ylabel('Density')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(OUTPUT_DIR, f'02_wavelet_denoising_{subject}.png')
        plt.savefig(filepath, bbox_inches='tight')
        print(f"  保存: {filepath}")
    
    plt.close()
    return ecg_denoised


def plot_wavelet_decomposition(ecg, fs, subject, save=True):
    """绘制小波分解细节"""
    print("\n[3/8] 绘制小波分解层次...")
    
    # 标准化
    ecg_norm = (ecg - np.mean(ecg)) / np.std(ecg)
    
    # 选择2秒片段
    start = int(20 * fs)
    end = int(22 * fs)
    segment = ecg_norm[start:end]
    t = np.arange(len(segment)) / fs
    
    # 小波分解
    wavelet = 'db4'
    max_level = pywt.dwt_max_level(len(segment), pywt.Wavelet(wavelet).dec_len)
    level = min(max_level, 6)
    coeffs = pywt.wavedec(segment, wavelet, level=level)
    
    fig, axes = plt.subplots(level + 2, 1, figsize=(16, 14))
    
    # 原始信号
    axes[0].plot(t, segment, color='#00ff9f', linewidth=0.8)
    axes[0].set_title('Original Signal', fontsize=12, color='#00ffff')
    axes[0].set_ylabel('Amp')
    axes[0].grid(True, alpha=0.3)
    
    # 近似系数 (cA)
    ca_len = len(coeffs[0])
    t_ca = np.linspace(0, t[-1], ca_len)
    axes[1].plot(t_ca, coeffs[0], color='#ff00ff', linewidth=0.8)
    axes[1].set_title(f'Approximation Coefficients (cA{level}) - Low Frequency', fontsize=12, color='#00ffff')
    axes[1].set_ylabel('Amp')
    axes[1].grid(True, alpha=0.3)
    
    # 细节系数 (cD)
    colors = ['#00ffff', '#ffff00', '#ff6b6b', '#9b59b6', '#3498db', '#e74c3c']
    for i, (detail, color) in enumerate(zip(coeffs[1:], colors)):
        ax = axes[i + 2]
        cd_len = len(detail)
        t_cd = np.linspace(0, t[-1], cd_len)
        ax.plot(t_cd, detail, color=color, linewidth=0.8)
        
        # 频率范围估算
        freq_low = fs / (2 ** (i + 2))
        freq_high = fs / (2 ** (i + 1))
        ax.set_title(f'Detail cD{level-i} ({freq_low:.1f}-{freq_high:.1f} Hz) - {"Noise" if i == 0 else "ECG Components"}', 
                    fontsize=10, color='#00ffff')
        ax.set_ylabel('Amp')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(OUTPUT_DIR, f'03_wavelet_decomposition_{subject}.png')
        plt.savefig(filepath, bbox_inches='tight')
        print(f"  保存: {filepath}")
    
    plt.close()


def plot_baseline_correction(ecg_denoised, fs, subject, save=True):
    """绘制基线漂移校正"""
    print("\n[4/8] 绘制基线校正对比...")
    
    # 基线校正
    corrector = BaselineCorrector(method='morphological', sampling_rate=fs)
    ecg_corrected, _ = corrector.correct(ecg_denoised)
    
    # 提取基线
    baseline = ecg_denoised - ecg_corrected
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    # 选择15秒片段
    start = int(30 * fs)
    end = int(45 * fs)
    t = np.arange(end - start) / fs
    
    # 原始（去噪后）
    axes[0].plot(t, ecg_denoised[start:end], color='#ff6b6b', linewidth=0.6)
    axes[0].set_title('After Wavelet Denoising (with Baseline Drift)', fontsize=14, color='#00ffff')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # 检测到的基线
    axes[1].plot(t, ecg_denoised[start:end], color='#ff6b6b', linewidth=0.5, alpha=0.5, label='Signal')
    axes[1].plot(t, baseline[start:end], color='#ffff00', linewidth=2, label='Estimated Baseline')
    axes[1].set_title('Baseline Estimation (Morphological Filter)', fontsize=14, color='#00ffff')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # 校正后
    axes[2].plot(t, ecg_corrected[start:end], color='#00ff9f', linewidth=0.6)
    axes[2].axhline(y=0, color='#ff00ff', linestyle='--', alpha=0.5, label='Zero Line')
    axes[2].set_title('After Baseline Correction', fontsize=14, color='#00ffff')
    axes[2].set_ylabel('Amplitude')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    # 对比
    axes[3].plot(t, ecg_denoised[start:end], color='#ff6b6b', linewidth=0.5, alpha=0.7, label='Before')
    axes[3].plot(t, ecg_corrected[start:end], color='#00ff9f', linewidth=0.6, label='After')
    axes[3].set_title('Comparison: Before vs After Baseline Correction', fontsize=14, color='#00ffff')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Amplitude')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(OUTPUT_DIR, f'04_baseline_correction_{subject}.png')
        plt.savefig(filepath, bbox_inches='tight')
        print(f"  保存: {filepath}")
    
    plt.close()
    return ecg_corrected


def plot_rpeak_detection(ecg_corrected, fs, subject, save=True):
    """绘制R峰检测结果"""
    print("\n[5/8] 绘制R峰检测结果...")
    
    # R峰检测
    detector = RPeakDetector(sampling_rate=fs, method='pan_tompkins')
    r_peaks, _ = detector.detect(ecg_corrected)
    
    # 计算RR间期
    rr_intervals = np.diff(r_peaks) / fs * 1000  # ms
    heart_rate = 60000 / rr_intervals  # BPM
    
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.5, 1, 1])
    
    # ECG with R峰标记
    ax1 = fig.add_subplot(gs[0, :])
    start = int(30 * fs)
    end = int(40 * fs)
    t = np.arange(end - start) / fs + 30
    
    ax1.plot(t, ecg_corrected[start:end], color='#00ff9f', linewidth=0.8, label='ECG')
    
    # 标记R峰
    peaks_in_range = r_peaks[(r_peaks >= start) & (r_peaks < end)]
    t_peaks = peaks_in_range / fs
    ax1.scatter(t_peaks, ecg_corrected[peaks_in_range], color='#ff00ff', s=80, 
                marker='v', zorder=5, label='R-peaks')
    
    # 标注RR间期
    for i in range(len(peaks_in_range) - 1):
        mid_t = (t_peaks[i] + t_peaks[i+1]) / 2
        rr = (peaks_in_range[i+1] - peaks_in_range[i]) / fs * 1000
        ax1.annotate(f'{rr:.0f}ms', xy=(mid_t, ax1.get_ylim()[1] * 0.8), 
                    fontsize=8, color='#ffff00', ha='center')
    
    ax1.set_title(f'R-Peak Detection (Pan-Tompkins Algorithm) - {len(r_peaks)} peaks detected', 
                 fontsize=14, color='#00ffff')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # RR间期时间序列
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(rr_intervals, color='#00ffff', linewidth=1, marker='o', markersize=3)
    ax2.axhline(y=np.mean(rr_intervals), color='#ff00ff', linestyle='--', 
               label=f'Mean: {np.mean(rr_intervals):.1f} ms')
    ax2.fill_between(range(len(rr_intervals)), 
                     np.mean(rr_intervals) - np.std(rr_intervals),
                     np.mean(rr_intervals) + np.std(rr_intervals),
                     alpha=0.3, color='#00ffff')
    ax2.set_title('RR Interval Time Series', fontsize=12, color='#00ffff')
    ax2.set_xlabel('Beat Number')
    ax2.set_ylabel('RR Interval (ms)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # RR间期直方图
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(rr_intervals, bins=30, color='#00ffff', alpha=0.7, edgecolor='#00ff9f')
    ax3.axvline(x=np.mean(rr_intervals), color='#ff00ff', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(rr_intervals):.1f} ms')
    ax3.axvline(x=np.median(rr_intervals), color='#ffff00', linestyle=':', linewidth=2,
               label=f'Median: {np.median(rr_intervals):.1f} ms')
    ax3.set_title(f'RR Interval Distribution (SDNN: {np.std(rr_intervals):.1f} ms)', 
                 fontsize=12, color='#00ffff')
    ax3.set_xlabel('RR Interval (ms)')
    ax3.set_ylabel('Count')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 心率变化
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(heart_rate, color='#ff6b6b', linewidth=1, marker='o', markersize=2)
    ax4.axhline(y=np.mean(heart_rate), color='#ff00ff', linestyle='--',
               label=f'Mean HR: {np.mean(heart_rate):.1f} BPM')
    ax4.set_title('Instantaneous Heart Rate', fontsize=12, color='#00ffff')
    ax4.set_xlabel('Beat Number')
    ax4.set_ylabel('Heart Rate (BPM)')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # Poincaré图
    ax5 = fig.add_subplot(gs[2, 1])
    if len(rr_intervals) > 1:
        rr_n = rr_intervals[:-1]
        rr_n1 = rr_intervals[1:]
        ax5.scatter(rr_n, rr_n1, c='#00ff9f', s=20, alpha=0.6)
        ax5.plot([min(rr_intervals), max(rr_intervals)], 
                [min(rr_intervals), max(rr_intervals)], 
                color='#ff00ff', linestyle='--', alpha=0.5)
    ax5.set_title('Poincaré Plot (RR_n vs RR_{n+1})', fontsize=12, color='#00ffff')
    ax5.set_xlabel('RR_n (ms)')
    ax5.set_ylabel('RR_{n+1} (ms)')
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(OUTPUT_DIR, f'05_rpeak_detection_{subject}.png')
        plt.savefig(filepath, bbox_inches='tight')
        print(f"  保存: {filepath}")
    
    plt.close()
    return r_peaks


def plot_beat_segmentation(ecg_corrected, r_peaks, fs, subject, save=True):
    """绘制心拍分割与叠加"""
    print("\n[6/8] 绘制心拍分割与叠加...")
    
    # 心拍分割
    segmenter = BeatSegmenter(
        sampling_rate=fs,
        pre_r=0.25,
        post_r=0.45,
        target_length=175
    )
    beats, valid_indices = segmenter.segment(ecg_corrected, r_peaks)
    
    # 过滤异常心拍 (创建简单的beat_info)
    beat_info = [{'idx': i} for i in range(len(beats))]
    filtered_beats, filtered_info = segmenter.filter_abnormal_beats(beats, beat_info)
    
    # 计算正常和异常的索引
    normal_indices = np.array([info['idx'] for info in filtered_info])
    all_indices = set(range(len(beats)))
    abnormal_indices = np.array(list(all_indices - set(normal_indices)))
    
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    # 所有心拍叠加
    ax1 = fig.add_subplot(gs[0, 0])
    t_beat = np.linspace(-0.25, 0.45, beats.shape[1])
    for i, beat in enumerate(beats[:100]):  # 最多显示100个
        ax1.plot(t_beat, beat, color='#00ff9f', alpha=0.1, linewidth=0.5)
    ax1.axvline(x=0, color='#ff00ff', linestyle='--', alpha=0.5, label='R-peak')
    ax1.set_title(f'All Beats Overlay ({len(beats)} beats)', fontsize=12, color='#00ffff')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 正常心拍叠加
    ax2 = fig.add_subplot(gs[0, 1])
    for beat in filtered_beats[:100]:
        ax2.plot(t_beat, beat, color='#00ff9f', alpha=0.15, linewidth=0.5)
    # 平均模板
    mean_beat = np.mean(filtered_beats, axis=0)
    ax2.plot(t_beat, mean_beat, color='#ff00ff', linewidth=2, label='Mean Template')
    ax2.axvline(x=0, color='#ffff00', linestyle='--', alpha=0.5)
    ax2.set_title(f'Normal Beats ({len(filtered_beats)} beats, {len(abnormal_indices)} rejected)', 
                 fontsize=12, color='#00ffff')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 异常心拍示例
    ax3 = fig.add_subplot(gs[1, 0])
    abnormal_beats = beats[abnormal_indices] if len(abnormal_indices) > 0 else []
    if len(abnormal_beats) > 0:
        for beat in abnormal_beats[:20]:
            ax3.plot(t_beat, beat, color='#ff6b6b', alpha=0.5, linewidth=0.8)
        ax3.plot(t_beat, mean_beat, color='#00ff9f', linewidth=2, linestyle='--', label='Normal Template')
    ax3.axvline(x=0, color='#ffff00', linestyle='--', alpha=0.5)
    ax3.set_title(f'Rejected Abnormal Beats ({len(abnormal_indices)} beats)', fontsize=12, color='#00ffff')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude')
    if len(abnormal_beats) > 0:
        ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 心拍模板（聚类后）
    ax4 = fig.add_subplot(gs[1, 1])
    templates, template_indices = segmenter.get_beat_templates(filtered_beats, n_clusters=3)
    colors = ['#00ff9f', '#00ffff', '#ff00ff']
    for i, (template, color) in enumerate(zip(templates, colors)):
        count = np.sum(template_indices == i)
        ax4.plot(t_beat, template, color=color, linewidth=2, label=f'Template {i+1} (n={count})')
    ax4.axvline(x=0, color='#ffff00', linestyle='--', alpha=0.5)
    ax4.set_title('Beat Templates (K-means Clustering)', fontsize=12, color='#00ffff')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # 心拍形态统计
    ax5 = fig.add_subplot(gs[2, 0])
    beat_amplitudes = np.max(filtered_beats, axis=1) - np.min(filtered_beats, axis=1)
    ax5.hist(beat_amplitudes, bins=30, color='#00ffff', alpha=0.7, edgecolor='#00ff9f')
    ax5.axvline(x=np.mean(beat_amplitudes), color='#ff00ff', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(beat_amplitudes):.3f}')
    ax5.set_title('Beat Amplitude Distribution', fontsize=12, color='#00ffff')
    ax5.set_xlabel('Peak-to-Peak Amplitude')
    ax5.set_ylabel('Count')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    # 相关性分布
    ax6 = fig.add_subplot(gs[2, 1])
    correlations = []
    for beat in beats:
        corr = np.corrcoef(beat, mean_beat)[0, 1]
        correlations.append(corr)
    correlations = np.array(correlations)
    
    ax6.hist(correlations, bins=30, color='#00ff9f', alpha=0.7, edgecolor='#00ffff')
    ax6.axvline(x=0.85, color='#ff00ff', linestyle='--', linewidth=2, label='Threshold (0.85)')
    ax6.set_title('Beat Correlation Distribution', fontsize=12, color='#00ffff')
    ax6.set_xlabel('Correlation with Mean Template')
    ax6.set_ylabel('Count')
    ax6.legend(loc='upper left')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(OUTPUT_DIR, f'06_beat_segmentation_{subject}.png')
        plt.savefig(filepath, bbox_inches='tight')
        print(f"  保存: {filepath}")
    
    plt.close()
    return filtered_beats


def plot_frequency_analysis(ecg, ecg_denoised, fs, subject, save=True):
    """绘制频谱分析"""
    print("\n[7/8] 绘制频谱分析...")
    
    # 标准化
    ecg_norm = (ecg - np.mean(ecg)) / np.std(ecg)
    
    # 选择一段信号进行FFT
    n_samples = min(len(ecg_norm), int(30 * fs))  # 30秒
    
    # FFT
    yf_orig = np.abs(fft(ecg_norm[:n_samples]))[:n_samples//2]
    yf_clean = np.abs(fft(ecg_denoised[:n_samples]))[:n_samples//2]
    xf = fftfreq(n_samples, 1/fs)[:n_samples//2]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 原始信号频谱
    axes[0, 0].semilogy(xf, yf_orig, color='#ff6b6b', linewidth=0.5, alpha=0.8)
    axes[0, 0].axvspan(0.5, 40, alpha=0.2, color='#00ff9f', label='ECG Band (0.5-40Hz)')
    axes[0, 0].axvline(x=50, color='#ffff00', linestyle='--', alpha=0.5, label='Power Line (50Hz)')
    axes[0, 0].set_title('Original Signal Spectrum', fontsize=12, color='#00ffff')
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('Magnitude (log)')
    axes[0, 0].set_xlim([0, 100])
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 去噪后频谱
    axes[0, 1].semilogy(xf, yf_clean, color='#00ff9f', linewidth=0.5, alpha=0.8)
    axes[0, 1].axvspan(0.5, 40, alpha=0.2, color='#00ff9f', label='ECG Band')
    axes[0, 1].set_title('Denoised Signal Spectrum', fontsize=12, color='#00ffff')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude (log)')
    axes[0, 1].set_xlim([0, 100])
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 频谱对比（线性）
    axes[1, 0].plot(xf, yf_orig, color='#ff6b6b', linewidth=0.5, alpha=0.7, label='Original')
    axes[1, 0].plot(xf, yf_clean, color='#00ff9f', linewidth=0.5, alpha=0.9, label='Denoised')
    axes[1, 0].set_title('Spectrum Comparison (Linear Scale)', fontsize=12, color='#00ffff')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].set_xlim([0, 60])
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 功率谱密度 (PSD)
    f_psd, psd_clean = signal.welch(ecg_denoised[:n_samples], fs, nperseg=int(fs*2))
    axes[1, 1].semilogy(f_psd, psd_clean, color='#00ffff', linewidth=1)
    axes[1, 1].axvspan(0.04, 0.15, alpha=0.3, color='#ff00ff', label='VLF (0.04-0.15Hz)')
    axes[1, 1].axvspan(0.15, 0.4, alpha=0.3, color='#00ff9f', label='LF (0.15-0.4Hz)')
    axes[1, 1].axvspan(0.4, 1.5, alpha=0.3, color='#ffff00', label='HF (0.4-1.5Hz)')
    axes[1, 1].set_title('Power Spectral Density (HRV Frequency Bands)', fontsize=12, color='#00ffff')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('PSD (log)')
    axes[1, 1].set_xlim([0, 2])
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(OUTPUT_DIR, f'07_frequency_analysis_{subject}.png')
        plt.savefig(filepath, bbox_inches='tight')
        print(f"  保存: {filepath}")
    
    plt.close()


def plot_preprocessing_summary(ecg, ecg_denoised, ecg_corrected, r_peaks, beats, fs, subject, save=True):
    """绘制预处理总结图"""
    print("\n[8/8] 绘制预处理流程总结...")
    
    # 标准化原始信号
    ecg_norm = (ecg - np.mean(ecg)) / np.std(ecg)
    
    fig = plt.figure(figsize=(18, 16))
    
    # 使用GridSpec创建布局
    gs = gridspec.GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 1, 1.2])
    
    # 选择5秒片段用于详细展示
    start = int(25 * fs)
    end = int(30 * fs)
    t = np.arange(end - start) / fs
    
    # Step 1: 原始信号
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, ecg_norm[start:end], color='#ff6b6b', linewidth=0.8)
    ax1.set_title('Step 1: Raw ECG Signal (Normalized)', fontsize=14, color='#ff6b6b', fontweight='bold')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim([t[0], t[-1]])
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.95, '① Original', transform=ax1.transAxes, fontsize=12, 
            color='#ff6b6b', verticalalignment='top', fontweight='bold')
    
    # Step 2: 小波去噪
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(t, ecg_denoised[start:end], color='#ffff00', linewidth=0.8)
    ax2.set_title('Step 2: After Wavelet Denoising (db4, Soft Threshold)', fontsize=14, color='#ffff00', fontweight='bold')
    ax2.set_ylabel('Amplitude')
    ax2.set_xlim([t[0], t[-1]])
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.95, '② Denoised', transform=ax2.transAxes, fontsize=12,
            color='#ffff00', verticalalignment='top', fontweight='bold')
    
    # Step 3: 基线校正
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(t, ecg_corrected[start:end], color='#00ff9f', linewidth=0.8)
    ax3.axhline(y=0, color='#ff00ff', linestyle='--', alpha=0.5)
    ax3.set_title('Step 3: After Baseline Correction (Morphological Filter)', fontsize=14, color='#00ff9f', fontweight='bold')
    ax3.set_ylabel('Amplitude')
    ax3.set_xlim([t[0], t[-1]])
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.95, '③ Baseline Corrected', transform=ax3.transAxes, fontsize=12,
            color='#00ff9f', verticalalignment='top', fontweight='bold')
    
    # 标记R峰
    peaks_in_range = r_peaks[(r_peaks >= start) & (r_peaks < end)]
    t_peaks = (peaks_in_range - start) / fs
    ax3.scatter(t_peaks, ecg_corrected[peaks_in_range], color='#ff00ff', s=100, 
               marker='v', zorder=5, label='R-peaks')
    ax3.legend(loc='upper right')
    
    # Step 4: 心拍分割结果
    ax4 = fig.add_subplot(gs[3, 0])
    t_beat = np.linspace(-0.25, 0.45, beats.shape[1])
    for beat in beats[:50]:
        ax4.plot(t_beat, beat, color='#00ffff', alpha=0.2, linewidth=0.5)
    mean_beat = np.mean(beats, axis=0)
    ax4.plot(t_beat, mean_beat, color='#ff00ff', linewidth=2.5, label='Mean Template')
    ax4.axvline(x=0, color='#ffff00', linestyle='--', alpha=0.7)
    ax4.set_title('Step 4: Beat Segmentation', fontsize=12, color='#00ffff', fontweight='bold')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.text(0.02, 0.95, '④ Beats', transform=ax4.transAxes, fontsize=12,
            color='#00ffff', verticalalignment='top', fontweight='bold')
    
    # 统计信息
    ax5 = fig.add_subplot(gs[3, 1])
    rr_intervals = np.diff(r_peaks) / fs * 1000
    stats_text = f"""
    Processing Statistics
    ━━━━━━━━━━━━━━━━━━━━━━
    
    Signal Duration: {len(ecg)/fs:.1f} s
    Sampling Rate: {fs:.1f} Hz
    
    R-peaks Detected: {len(r_peaks)}
    Valid Beats: {len(beats)}
    
    Mean Heart Rate: {60000/np.mean(rr_intervals):.1f} BPM
    SDNN: {np.std(rr_intervals):.1f} ms
    RMSSD: {np.sqrt(np.mean(np.diff(rr_intervals)**2)):.1f} ms
    
    Signal Quality: {'Good' if len(beats) > 100 else 'Fair'}
    """
    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace', color='#e0e0e0')
    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 1])
    ax5.axis('off')
    ax5.set_title('Statistics', fontsize=12, color='#00ffff', fontweight='bold')
    
    # 处理流程图
    ax6 = fig.add_subplot(gs[3, 2])
    pipeline_text = """
    ╔═══════════════════════════╗
    ║     PREPROCESSING         ║
    ║       PIPELINE            ║
    ╠═══════════════════════════╣
    ║                           ║
    ║  ┌─────────────────────┐  ║
    ║  │  1. Raw Signal      │  ║
    ║  └──────────┬──────────┘  ║
    ║             ▼             ║
    ║  ┌─────────────────────┐  ║
    ║  │  2. Wavelet Denoise │  ║
    ║  └──────────┬──────────┘  ║
    ║             ▼             ║
    ║  ┌─────────────────────┐  ║
    ║  │  3. Baseline Corr.  │  ║
    ║  └──────────┬──────────┘  ║
    ║             ▼             ║
    ║  ┌─────────────────────┐  ║
    ║  │  4. R-peak Detect   │  ║
    ║  └──────────┬──────────┘  ║
    ║             ▼             ║
    ║  ┌─────────────────────┐  ║
    ║  │  5. Beat Segment    │  ║
    ║  └─────────────────────┘  ║
    ║                           ║
    ╚═══════════════════════════╝
    """
    ax6.text(0.05, 0.95, pipeline_text, transform=ax6.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace', color='#00ff9f')
    ax6.set_xlim([0, 1])
    ax6.set_ylim([0, 1])
    ax6.axis('off')
    ax6.set_title('Pipeline', fontsize=12, color='#00ffff', fontweight='bold')
    
    plt.suptitle(f'ECG Preprocessing Pipeline Summary - Subject {subject}', 
                fontsize=18, color='#ff00ff', fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save:
        filepath = os.path.join(OUTPUT_DIR, f'08_preprocessing_summary_{subject}.png')
        plt.savefig(filepath, bbox_inches='tight')
        print(f"  保存: {filepath}")
    
    plt.close()


def plot_multi_subject_comparison(subjects=['B', 'C', 'D', 'E', 'F'], save=True):
    """绘制多被试对比图"""
    print("\n[额外] 绘制多被试心拍对比...")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    colors = ['#00ff9f', '#00ffff', '#ff00ff', '#ffff00', '#ff6b6b', '#9b59b6']
    
    for idx, subject in enumerate(subjects[:6]):
        try:
            # 加载和处理
            ecg, fs, _ = load_ecg_data(subject)
            ecg_norm = (ecg - np.mean(ecg)) / np.std(ecg)
            
            # 去噪
            denoiser = WaveletDenoiser()
            ecg_denoised, _ = denoiser.denoise(ecg_norm)
            
            # 基线校正
            corrector = BaselineCorrector(method='morphological', sampling_rate=fs)
            ecg_corrected, _ = corrector.correct(ecg_denoised)
            
            # R峰检测
            detector = RPeakDetector(sampling_rate=fs, method='pan_tompkins')
            r_peaks, _ = detector.detect(ecg_corrected)
            
            # 心拍分割
            segmenter = BeatSegmenter(sampling_rate=fs)
            beats, _ = segmenter.segment(ecg_corrected, r_peaks)
            beat_info = [{'idx': i} for i in range(len(beats))]
            filtered_beats, _ = segmenter.filter_abnormal_beats(beats, beat_info)
            
            # 绘制
            ax = axes[idx]
            t_beat = np.linspace(-0.25, 0.45, filtered_beats.shape[1])
            
            for beat in filtered_beats[:30]:
                ax.plot(t_beat, beat, color=colors[idx], alpha=0.15, linewidth=0.5)
            
            mean_beat = np.mean(filtered_beats, axis=0)
            ax.plot(t_beat, mean_beat, color='white', linewidth=2)
            ax.axvline(x=0, color='#ff00ff', linestyle='--', alpha=0.5)
            
            rr = np.diff(r_peaks) / fs * 1000
            hr = 60000 / np.mean(rr)
            
            ax.set_title(f'Subject {subject}\n{len(filtered_beats)} beats, HR={hr:.0f} BPM', 
                        fontsize=12, color=colors[idx])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"  处理 {subject} 失败: {e}")
            axes[idx].text(0.5, 0.5, f'Subject {subject}\nNo Data', 
                          ha='center', va='center', fontsize=14, color='#ff6b6b')
            axes[idx].axis('off')
    
    # 隐藏多余的子图
    if len(subjects) < 6:
        for i in range(len(subjects), 6):
            axes[i].axis('off')
    
    plt.suptitle('Multi-Subject ECG Beat Comparison', fontsize=16, color='#00ffff', fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save:
        filepath = os.path.join(OUTPUT_DIR, '09_multi_subject_comparison.png')
        plt.savefig(filepath, bbox_inches='tight')
        print(f"  保存: {filepath}")
    
    plt.close()


def main():
    """主函数"""
    print("=" * 60)
    print("ECG预处理可视化")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输出目录: {OUTPUT_DIR}/")
    print("=" * 60)
    
    # 设置深色主题
    set_dark_style()
    
    # 选择一个被试进行详细展示
    subject = 'C'  # 选择数据质量较好的被试C
    
    # 加载数据
    ecg, fs, subject = load_ecg_data(subject)
    
    # 1. 原始信号概览
    plot_raw_signal_overview(ecg, fs, subject)
    
    # 2. 小波去噪
    ecg_denoised = plot_wavelet_denoising(ecg, fs, subject)
    
    # 3. 小波分解层次
    plot_wavelet_decomposition(ecg, fs, subject)
    
    # 4. 基线校正
    ecg_corrected = plot_baseline_correction(ecg_denoised, fs, subject)
    
    # 5. R峰检测
    r_peaks = plot_rpeak_detection(ecg_corrected, fs, subject)
    
    # 6. 心拍分割
    # 分割心拍
    segmenter = BeatSegmenter(sampling_rate=fs)
    beats, valid_indices = segmenter.segment(ecg_corrected, r_peaks)
    beat_info = [{'idx': i} for i in range(len(beats))]
    filtered_beats, _ = segmenter.filter_abnormal_beats(beats, beat_info)
    
    plot_beat_segmentation(ecg_corrected, r_peaks, fs, subject)
    
    # 7. 频谱分析
    plot_frequency_analysis(ecg, ecg_denoised, fs, subject)
    
    # 8. 预处理总结
    plot_preprocessing_summary(ecg, ecg_denoised, ecg_corrected, r_peaks, filtered_beats, fs, subject)
    
    # 9. 多被试对比
    plot_multi_subject_comparison(['B', 'C', 'D', 'E', 'F'])
    
    print("\n" + "=" * 60)
    print("✅ 所有图像已生成!")
    print(f"📁 输出目录: {OUTPUT_DIR}/")
    print("=" * 60)
    
    # 列出生成的文件
    print("\n生成的文件:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            filepath = os.path.join(OUTPUT_DIR, f)
            size = os.path.getsize(filepath) / 1024
            print(f"  📊 {f} ({size:.1f} KB)")


if __name__ == '__main__':
    main()
