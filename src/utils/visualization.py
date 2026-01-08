"""
可视化模块
==========

ECG信号和分析结果的可视化
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class ECGVisualizer:
    """
    ECG可视化器
    
    生成各种交互式可视化图表
    """
    
    # 赛博朋克配色方案
    CYBERPUNK_COLORS = {
        'primary': '#00ff9f',      # 霓虹绿
        'secondary': '#ff00ff',    # 霓虹粉
        'accent': '#00ffff',       # 青色
        'warning': '#ffff00',      # 黄色
        'danger': '#ff0040',       # 红色
        'background': '#0a0a0f',   # 深色背景
        'surface': '#1a1a2e',      # 表面色
        'text': '#e0e0e0',         # 文字色
        'grid': '#2a2a3e'          # 网格色
    }
    
    SUBJECT_COLORS = {
        'A': '#00ff9f',
        'B': '#ff00ff',
        'C': '#00ffff',
        'D': '#ffff00',
        'E': '#ff0040',
        'F': '#9d00ff'
    }
    
    def __init__(self, theme: str = 'cyberpunk'):
        """
        初始化可视化器
        
        Args:
            theme: 主题 ('cyberpunk', 'medical', 'default')
        """
        self.theme = theme
        self.colors = self.CYBERPUNK_COLORS
    
    def plot_ecg_signal(
        self,
        signal: np.ndarray,
        sampling_rate: float = 200.0,
        title: str = "ECG Signal",
        r_peaks: Optional[np.ndarray] = None,
        highlight_range: Optional[Tuple[int, int]] = None
    ) -> go.Figure:
        """
        绘制ECG信号
        
        Args:
            signal: ECG信号
            sampling_rate: 采样率
            title: 标题
            r_peaks: R峰位置
            highlight_range: 高亮区域
            
        Returns:
            Plotly Figure
        """
        time = np.arange(len(signal)) / sampling_rate
        
        fig = go.Figure()
        
        # 主信号
        fig.add_trace(go.Scatter(
            x=time,
            y=signal,
            mode='lines',
            name='ECG',
            line=dict(color=self.colors['primary'], width=1.5),
            hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.2f}<extra></extra>'
        ))
        
        # R峰标记
        if r_peaks is not None and len(r_peaks) > 0:
            r_times = r_peaks / sampling_rate
            r_values = signal[r_peaks]
            
            fig.add_trace(go.Scatter(
                x=r_times,
                y=r_values,
                mode='markers',
                name='R-peaks',
                marker=dict(
                    color=self.colors['danger'],
                    size=8,
                    symbol='triangle-up'
                )
            ))
        
        # 高亮区域
        if highlight_range:
            start, end = highlight_range
            fig.add_vrect(
                x0=start / sampling_rate,
                x1=end / sampling_rate,
                fillcolor=self.colors['accent'],
                opacity=0.2,
                line_width=0
            )
        
        # 布局
        fig.update_layout(
            title=dict(text=title, font=dict(color=self.colors['text'])),
            xaxis_title='Time (s)',
            yaxis_title='Amplitude',
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['surface'],
            font=dict(color=self.colors['text']),
            xaxis=dict(gridcolor=self.colors['grid'], showgrid=True),
            yaxis=dict(gridcolor=self.colors['grid'], showgrid=True),
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def plot_beats_overlay(
        self,
        beats: np.ndarray,
        sampling_rate: float = 200.0,
        title: str = "Beat Overlay"
    ) -> go.Figure:
        """
        绘制心拍叠加图
        
        Args:
            beats: 心拍数组 (n_beats, beat_length)
            sampling_rate: 采样率
            title: 标题
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        beat_length = beats.shape[1]
        time = np.arange(beat_length) / sampling_rate * 1000  # ms
        
        # 绘制所有心拍 (半透明)
        for i, beat in enumerate(beats):
            fig.add_trace(go.Scatter(
                x=time,
                y=beat,
                mode='lines',
                name=f'Beat {i+1}',
                line=dict(color=self.colors['primary'], width=0.5),
                opacity=0.3,
                showlegend=False
            ))
        
        # 绘制平均心拍
        mean_beat = np.mean(beats, axis=0)
        fig.add_trace(go.Scatter(
            x=time,
            y=mean_beat,
            mode='lines',
            name='Mean Beat',
            line=dict(color=self.colors['secondary'], width=3)
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(color=self.colors['text'])),
            xaxis_title='Time (ms)',
            yaxis_title='Amplitude',
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['surface'],
            font=dict(color=self.colors['text']),
            height=400
        )
        
        return fig
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        labels: List[str],
        title: str = "Confusion Matrix"
    ) -> go.Figure:
        """
        绘制混淆矩阵
        
        Args:
            confusion_matrix: 混淆矩阵
            labels: 类别标签
            title: 标题
            
        Returns:
            Plotly Figure
        """
        # 归一化
        cm_normalized = confusion_matrix.astype(float) / confusion_matrix.sum(axis=1, keepdims=True)
        
        # 创建文本注释
        text = []
        for i in range(len(labels)):
            row = []
            for j in range(len(labels)):
                row.append(f"{confusion_matrix[i, j]}<br>({cm_normalized[i, j]:.1%})")
            text.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm_normalized,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=14, color='white'),
            colorscale=[
                [0, self.colors['surface']],
                [0.5, self.colors['accent']],
                [1, self.colors['primary']]
            ],
            showscale=True,
            colorbar=dict(title='Ratio')
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(color=self.colors['text'])),
            xaxis_title='Predicted',
            yaxis_title='Actual',
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['surface'],
            font=dict(color=self.colors['text']),
            height=500,
            width=600
        )
        
        return fig
    
    def plot_hrv_dashboard(
        self,
        hrv_features: Dict[str, float],
        title: str = "HRV Dashboard"
    ) -> go.Figure:
        """
        绘制HRV仪表盘
        
        Args:
            hrv_features: HRV特征字典
            title: 标题
            
        Returns:
            Plotly Figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "indicator"}, {"type": "indicator"}]
            ],
            subplot_titles=['Heart Rate', 'SDNN', 'RMSSD', 'LF/HF Ratio']
        )
        
        # 心率指示器
        hr = hrv_features.get('hrv_hr_mean', 0)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=hr,
            title={'text': "BPM", 'font': {'color': self.colors['text']}},
            gauge={
                'axis': {'range': [40, 120], 'tickcolor': self.colors['text']},
                'bar': {'color': self.colors['primary']},
                'bgcolor': self.colors['surface'],
                'bordercolor': self.colors['grid'],
                'steps': [
                    {'range': [40, 60], 'color': self.colors['accent']},
                    {'range': [60, 100], 'color': self.colors['surface']},
                    {'range': [100, 120], 'color': self.colors['danger']}
                ]
            },
            number={'font': {'color': self.colors['primary']}}
        ), row=1, col=1)
        
        # SDNN指示器
        sdnn = hrv_features.get('hrv_sdnn', 0)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=sdnn,
            title={'text': "ms", 'font': {'color': self.colors['text']}},
            gauge={
                'axis': {'range': [0, 200]},
                'bar': {'color': self.colors['secondary']},
                'bgcolor': self.colors['surface'],
            },
            number={'font': {'color': self.colors['secondary']}}
        ), row=1, col=2)
        
        # RMSSD指示器
        rmssd = hrv_features.get('hrv_rmssd', 0)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=rmssd,
            title={'text': "ms", 'font': {'color': self.colors['text']}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': self.colors['accent']},
                'bgcolor': self.colors['surface'],
            },
            number={'font': {'color': self.colors['accent']}}
        ), row=2, col=1)
        
        # LF/HF比值
        lf_hf = hrv_features.get('hrv_lf_hf_ratio', 0)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=lf_hf,
            title={'text': "Ratio", 'font': {'color': self.colors['text']}},
            gauge={
                'axis': {'range': [0, 5]},
                'bar': {'color': self.colors['warning']},
                'bgcolor': self.colors['surface'],
            },
            number={'font': {'color': self.colors['warning']}}
        ), row=2, col=2)
        
        fig.update_layout(
            title=dict(text=title, font=dict(color=self.colors['text'])),
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            height=600
        )
        
        return fig
    
    def plot_poincare(
        self,
        rr_intervals: np.ndarray,
        title: str = "Poincaré Plot"
    ) -> go.Figure:
        """
        绘制Poincaré图
        
        Args:
            rr_intervals: RR间期序列 (ms)
            title: 标题
            
        Returns:
            Plotly Figure
        """
        rr_n = rr_intervals[:-1]
        rr_n1 = rr_intervals[1:]
        
        fig = go.Figure()
        
        # 散点图
        fig.add_trace(go.Scatter(
            x=rr_n,
            y=rr_n1,
            mode='markers',
            marker=dict(
                color=self.colors['primary'],
                size=5,
                opacity=0.6
            ),
            name='RR intervals'
        ))
        
        # 对角线
        min_val = min(rr_n.min(), rr_n1.min())
        max_val = max(rr_n.max(), rr_n1.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color=self.colors['secondary'], dash='dash'),
            name='Identity line'
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(color=self.colors['text'])),
            xaxis_title='RR(n) (ms)',
            yaxis_title='RR(n+1) (ms)',
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['surface'],
            font=dict(color=self.colors['text']),
            height=500,
            width=500
        )
        
        return fig
    
    def plot_training_history(
        self,
        history: Dict,
        title: str = "Training History"
    ) -> go.Figure:
        """
        绘制训练历史
        
        Args:
            history: 训练历史字典
            title: 标题
            
        Returns:
            Plotly Figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Loss', 'Accuracy'],
            vertical_spacing=0.15
        )
        
        epochs = list(range(1, len(history['train_loss']) + 1))
        
        # Loss
        fig.add_trace(go.Scatter(
            x=epochs, y=history['train_loss'],
            mode='lines', name='Train Loss',
            line=dict(color=self.colors['primary'])
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=epochs, y=history['val_loss'],
            mode='lines', name='Val Loss',
            line=dict(color=self.colors['secondary'])
        ), row=1, col=1)
        
        # Accuracy
        fig.add_trace(go.Scatter(
            x=epochs, y=history['train_acc'],
            mode='lines', name='Train Acc',
            line=dict(color=self.colors['primary'])
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=epochs, y=history['val_acc'],
            mode='lines', name='Val Acc',
            line=dict(color=self.colors['secondary'])
        ), row=2, col=1)
        
        # 标记最佳epoch
        best_epoch = history.get('best_epoch', 0)
        if best_epoch > 0:
            fig.add_vline(
                x=best_epoch + 1,
                line_dash="dash",
                line_color=self.colors['accent'],
                annotation_text="Best"
            )
        
        fig.update_layout(
            title=dict(text=title, font=dict(color=self.colors['text'])),
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['surface'],
            font=dict(color=self.colors['text']),
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance: np.ndarray,
        top_n: int = 20,
        title: str = "Feature Importance"
    ) -> go.Figure:
        """
        绘制特征重要性
        
        Args:
            feature_names: 特征名称
            importance: 重要性值
            top_n: 显示前n个特征
            title: 标题
            
        Returns:
            Plotly Figure
        """
        # 排序并选择top_n
        indices = np.argsort(importance)[-top_n:]
        top_features = [feature_names[i] for i in indices]
        top_importance = importance[indices]
        
        fig = go.Figure(go.Bar(
            x=top_importance,
            y=top_features,
            orientation='h',
            marker=dict(
                color=top_importance,
                colorscale=[
                    [0, self.colors['surface']],
                    [0.5, self.colors['accent']],
                    [1, self.colors['primary']]
                ]
            )
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(color=self.colors['text'])),
            xaxis_title='Importance',
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['surface'],
            font=dict(color=self.colors['text']),
            height=600
        )
        
        return fig
    
    def plot_probability_distribution(
        self,
        probabilities: np.ndarray,
        labels: List[str],
        predicted_class: int,
        title: str = "Prediction Confidence"
    ) -> go.Figure:
        """
        绘制预测概率分布
        
        Args:
            probabilities: 各类别概率
            labels: 类别标签
            predicted_class: 预测类别索引
            title: 标题
            
        Returns:
            Plotly Figure
        """
        colors = [
            self.colors['primary'] if i == predicted_class else self.colors['surface']
            for i in range(len(labels))
        ]
        
        fig = go.Figure(go.Bar(
            x=labels,
            y=probabilities * 100,
            marker=dict(
                color=colors,
                line=dict(color=self.colors['accent'], width=2)
            ),
            text=[f'{p*100:.1f}%' for p in probabilities],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(color=self.colors['text'])),
            xaxis_title='Identity',
            yaxis_title='Confidence (%)',
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['surface'],
            font=dict(color=self.colors['text']),
            height=400,
            yaxis=dict(range=[0, 110])
        )
        
        return fig
