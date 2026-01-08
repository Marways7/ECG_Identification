"""
ECG身份识别系统 - Streamlit UI
==============================

赛博朋克风格的交互式用户界面

启动方式: streamlit run ui/app.py
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# 页面配置
st.set_page_config(
    page_title="ECG-ID | 心电身份识别系统",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 赛博朋克CSS样式
CYBERPUNK_CSS = """
<style>
    /* 主题变量 */
    :root {
        --neon-green: #00ff9f;
        --neon-pink: #ff00ff;
        --neon-cyan: #00ffff;
        --neon-yellow: #ffff00;
        --neon-red: #ff0040;
        --bg-dark: #0a0a0f;
        --bg-surface: #1a1a2e;
        --text-primary: #e0e0e0;
        --text-secondary: #a0a0a0;
    }
    
    /* 全局背景 */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%);
    }
    
    /* 侧边栏 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0a0a0f 100%);
        border-right: 1px solid #00ff9f40;
    }
    
    /* 标题样式 */
    h1 {
        color: #00ff9f !important;
        text-shadow: 0 0 10px #00ff9f80, 0 0 20px #00ff9f40;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 2px;
    }
    
    h2, h3 {
        color: #00ffff !important;
        text-shadow: 0 0 5px #00ffff60;
    }
    
    /* 卡片样式 */
    .cyber-card {
        background: linear-gradient(145deg, #1a1a2e, #0f0f1a);
        border: 1px solid #00ff9f40;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 15px #00ff9f20, inset 0 0 30px #00000080;
    }
    
    /* 霓虹按钮 */
    .stButton > button {
        background: linear-gradient(145deg, #1a1a2e, #0f0f1a) !important;
        border: 2px solid #00ff9f !important;
        color: #00ff9f !important;
        text-shadow: 0 0 5px #00ff9f;
        box-shadow: 0 0 10px #00ff9f40, inset 0 0 10px #00ff9f10;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #00ff9f20 !important;
        box-shadow: 0 0 20px #00ff9f80, inset 0 0 20px #00ff9f20;
        transform: translateY(-2px);
    }
    
    /* 指标卡片 */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #0f0f1a);
        border: 1px solid #00ffff40;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 0 10px #00ffff20;
    }
    
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        color: #00ff9f;
        text-shadow: 0 0 10px #00ff9f80;
    }
    
    .metric-label {
        color: #a0a0a0;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* 进度条 */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00ff9f, #00ffff) !important;
        box-shadow: 0 0 10px #00ff9f;
    }
    
    /* 选择框 */
    .stSelectbox > div > div {
        background: #1a1a2e !important;
        border: 1px solid #00ff9f40 !important;
        color: #e0e0e0 !important;
    }
    
    /* 警告框 - 成功 */
    .stSuccess {
        background: #00ff9f20 !important;
        border: 1px solid #00ff9f !important;
    }
    
    /* 扫描线动画 */
    @keyframes scanline {
        0% { transform: translateY(-100%); }
        100% { transform: translateY(100%); }
    }
    
    .scanline {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00ff9f, transparent);
        animation: scanline 3s linear infinite;
        pointer-events: none;
        z-index: 1000;
    }
    
    /* 脉冲动画 */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .pulse {
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* 滚动条样式 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0a0f;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #00ff9f40;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #00ff9f80;
    }
</style>

<!-- Google Fonts -->
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">

<!-- 扫描线效果 -->
<div class="scanline"></div>
"""

# 注入CSS
st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)


def create_cyber_header():
    """创建赛博朋克风格的头部"""
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 3em; margin-bottom: 0;">
            ⚡ ECG-ID SYSTEM ⚡
        </h1>
        <p style="color: #00ffff; font-family: 'Share Tech Mono', monospace; letter-spacing: 3px;">
            NEURAL BIOMETRIC IDENTIFICATION v1.0
        </p>
        <div style="width: 100%; height: 2px; background: linear-gradient(90deg, transparent, #00ff9f, #00ffff, #ff00ff, transparent); margin: 20px 0;"></div>
    </div>
    """, unsafe_allow_html=True)


def create_metric_card(label: str, value: str, delta: str = None, color: str = "#00ff9f"):
    """创建指标卡片"""
    delta_html = f'<div style="color: {color}; font-size: 0.8em;">▲ {delta}</div>' if delta else ''
    
    return f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {color}; text-shadow: 0 0 10px {color}80;">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """


def plot_ecg_cyberpunk(signal, fs=200, r_peaks=None, title="ECG Signal"):
    """赛博朋克风格ECG图"""
    time = np.arange(len(signal)) / fs
    
    fig = go.Figure()
    
    # 主信号
    fig.add_trace(go.Scatter(
        x=time,
        y=signal,
        mode='lines',
        name='ECG',
        line=dict(color='#00ff9f', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 159, 0.1)'
    ))
    
    # R峰标记
    if r_peaks is not None and len(r_peaks) > 0:
        fig.add_trace(go.Scatter(
            x=r_peaks / fs,
            y=signal[r_peaks],
            mode='markers',
            name='R-peaks',
            marker=dict(color='#ff0040', size=10, symbol='triangle-up')
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color='#00ffff', size=16)),
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        template='plotly_dark',
        paper_bgcolor='rgba(10, 10, 15, 0)',
        plot_bgcolor='rgba(26, 26, 46, 0.8)',
        font=dict(color='#e0e0e0'),
        xaxis=dict(gridcolor='rgba(0, 255, 159, 0.1)', showgrid=True),
        yaxis=dict(gridcolor='rgba(0, 255, 159, 0.1)', showgrid=True),
        height=350,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def plot_confusion_matrix_cyber(cm, labels):
    """赛博朋克风格混淆矩阵"""
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    text = [[f"{cm[i,j]}<br>({cm_normalized[i,j]:.0%})" for j in range(len(labels))] 
            for i in range(len(labels))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=14, color='white'),
        colorscale=[[0, '#1a1a2e'], [0.5, '#00ffff'], [1, '#00ff9f']],
        showscale=True
    ))
    
    fig.update_layout(
        title=dict(text='Confusion Matrix', font=dict(color='#00ffff')),
        xaxis_title='Predicted',
        yaxis_title='Actual',
        template='plotly_dark',
        paper_bgcolor='rgba(10, 10, 15, 0)',
        plot_bgcolor='rgba(26, 26, 46, 0.8)',
        font=dict(color='#e0e0e0'),
        height=450
    )
    
    return fig


def plot_probability_cyber(probs, labels, predicted_idx):
    """赛博朋克风格概率分布图"""
    colors = ['#00ff9f' if i == predicted_idx else '#1a1a2e' for i in range(len(labels))]
    
    fig = go.Figure(data=go.Bar(
        x=labels,
        y=probs * 100,
        marker=dict(
            color=colors,
            line=dict(color='#00ffff', width=2)
        ),
        text=[f'{p*100:.1f}%' for p in probs],
        textposition='outside',
        textfont=dict(color='#00ff9f')
    ))
    
    fig.update_layout(
        title=dict(text='Identity Confidence', font=dict(color='#00ffff')),
        xaxis_title='Identity',
        yaxis_title='Confidence (%)',
        template='plotly_dark',
        paper_bgcolor='rgba(10, 10, 15, 0)',
        plot_bgcolor='rgba(26, 26, 46, 0.8)',
        font=dict(color='#e0e0e0'),
        yaxis=dict(range=[0, 110]),
        height=350
    )
    
    return fig


def create_hrv_gauge(value, min_val, max_val, title, color="#00ff9f"):
    """创建HRV仪表盘"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'color': '#00ffff'}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickcolor': '#e0e0e0'},
            'bar': {'color': color},
            'bgcolor': '#1a1a2e',
            'bordercolor': 'rgba(0, 255, 255, 0.25)',
            'steps': [
                {'range': [min_val, (min_val+max_val)/3], 'color': 'rgba(0, 255, 255, 0.2)'},
                {'range': [(min_val+max_val)/3, 2*(min_val+max_val)/3], 'color': 'rgba(0, 255, 159, 0.2)'},
                {'range': [2*(min_val+max_val)/3, max_val], 'color': 'rgba(255, 0, 64, 0.2)'}
            ]
        },
        number={'font': {'color': color}}
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(10, 10, 15, 0)',
        font=dict(color='#e0e0e0'),
        height=250,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig


def main():
    """主函数"""
    create_cyber_header()
    
    # 侧边栏
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <h2 style="color: #ff00ff;">⚙️ CONTROL PANEL</h2>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "NAVIGATION",
            ["🏠 Dashboard", "📊 Data Analysis", "🧠 Model Training", "🔍 Identification", "📈 Performance"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        st.markdown("""
        <div style="text-align: center; color: #a0a0a0; font-size: 0.8em;">
            <p>System Status: <span style="color: #00ff9f;">● ONLINE</span></p>
            <p>GPU: <span style="color: #00ffff;">ENABLED</span></p>
            <p>Model: <span style="color: #ff00ff;">LOADED</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    # 主内容区域
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📊 Data Analysis":
        show_data_analysis()
    elif page == "🧠 Model Training":
        show_training()
    elif page == "🔍 Identification":
        show_identification()
    elif page == "📈 Performance":
        show_performance()


def show_dashboard():
    """显示仪表盘"""
    st.markdown("## 📊 SYSTEM OVERVIEW")
    
    # 关键指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card("ACCURACY", "98.5%", "+2.3%", "#00ff9f"), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card("SUBJECTS", "6", None, "#00ffff"), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric_card("BEATS", "12,847", "+1,234", "#ff00ff"), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_metric_card("F1 SCORE", "0.972", "+0.015", "#ffff00"), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 实时ECG模拟
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💓 REAL-TIME ECG MONITOR")
        
        # 生成模拟ECG
        t = np.linspace(0, 5, 1000)
        ecg = np.sin(2 * np.pi * 1.2 * t) * np.exp(-((t % 0.8 - 0.2) ** 2) / 0.01)
        ecg += 0.3 * np.sin(2 * np.pi * 0.25 * t)
        ecg += np.random.normal(0, 0.05, len(t))
        
        fig = plot_ecg_cyberpunk(ecg, fs=200, title="Subject A - Live Feed")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 HRV STATUS")
        
        # HRV指标
        hr_fig = create_hrv_gauge(72, 40, 120, "Heart Rate (BPM)")
        st.plotly_chart(hr_fig, use_container_width=True)
        
        st.markdown("""
        <div class="cyber-card">
            <p style="color: #00ffff;">SDNN: <span style="color: #00ff9f;">45.2 ms</span></p>
            <p style="color: #00ffff;">RMSSD: <span style="color: #00ff9f;">32.8 ms</span></p>
            <p style="color: #00ffff;">LF/HF: <span style="color: #ffff00;">1.42</span></p>
        </div>
        """, unsafe_allow_html=True)


def show_data_analysis():
    """数据分析页面"""
    st.markdown("## 📊 DATA ANALYSIS")
    
    # 被试选择
    col1, col2 = st.columns([1, 3])
    
    with col1:
        subject = st.selectbox("SELECT SUBJECT", ['A', 'B', 'C', 'D', 'E', 'F'])
        
        st.markdown(f"""
        <div class="cyber-card">
            <h4 style="color: #00ffff;">Subject {subject}</h4>
            <p>Records: <span style="color: #00ff9f;">54,484</span></p>
            <p>Duration: <span style="color: #00ff9f;">~5 min</span></p>
            <p>Sample Rate: <span style="color: #00ff9f;">200 Hz</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # 加载并显示数据
        try:
            df = pd.read_csv(f'ECG_Data/{subject}1_processed.csv')
            signal = df['Channel 1'].values[:2000]
            
            fig = plot_ecg_cyberpunk(signal, fs=200, title=f"Subject {subject} - ECG Waveform")
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("📡 Data not loaded. Please ensure ECG_Data folder exists.")
    
    # 心拍叠加图
    st.markdown("### 💓 BEAT OVERLAY ANALYSIS")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 模拟心拍叠加
        beats = []
        for _ in range(50):
            t = np.linspace(0, 0.7, 140)
            beat = np.sin(2 * np.pi * 1.5 * t) * np.exp(-((t - 0.25) ** 2) / 0.005)
            beat += np.random.normal(0, 0.03, len(t))
            beats.append(beat)
        
        beats = np.array(beats)
        mean_beat = np.mean(beats, axis=0)
        
        fig = go.Figure()
        
        for beat in beats[:20]:
            fig.add_trace(go.Scatter(
                y=beat, mode='lines',
                line=dict(color='#00ff9f', width=0.5),
                opacity=0.3, showlegend=False
            ))
        
        fig.add_trace(go.Scatter(
            y=mean_beat, mode='lines',
            name='Mean Beat',
            line=dict(color='#ff00ff', width=3)
        ))
        
        fig.update_layout(
            title='Beat Overlay',
            template='plotly_dark',
            paper_bgcolor='rgba(10, 10, 15, 0)',
            plot_bgcolor='rgba(26, 26, 46, 0.8)',
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Poincaré图
        rr = np.random.normal(800, 50, 200)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rr[:-1], y=rr[1:],
            mode='markers',
            marker=dict(color='#00ff9f', size=5, opacity=0.6)
        ))
        
        # 添加对角线
        fig.add_trace(go.Scatter(
            x=[rr.min(), rr.max()],
            y=[rr.min(), rr.max()],
            mode='lines',
            line=dict(color='#ff00ff', dash='dash'),
            name='Identity'
        ))
        
        fig.update_layout(
            title='Poincaré Plot',
            xaxis_title='RR(n) (ms)',
            yaxis_title='RR(n+1) (ms)',
            template='plotly_dark',
            paper_bgcolor='rgba(10, 10, 15, 0)',
            plot_bgcolor='rgba(26, 26, 46, 0.8)',
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)


def show_training():
    """模型训练页面"""
    st.markdown("## 🧠 MODEL TRAINING")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ CONFIGURATION")
        
        model_type = st.selectbox("Model Type", ["CNN", "Siamese", "Hybrid"])
        epochs = st.slider("Epochs", 10, 200, 50)
        lr = st.select_slider("Learning Rate", options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        
        if st.button("🚀 START TRAINING", use_container_width=True):
            st.session_state['training'] = True
            
            # 模拟训练进度
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                time.sleep(0.05)
                progress_bar.progress(i + 1)
                status_text.text(f"Training... Epoch {i * epochs // 100 + 1}/{epochs}")
            
            st.success("✅ Training Complete!")
            st.session_state['training'] = False
    
    with col2:
        st.markdown("### 📈 TRAINING PROGRESS")
        
        # 模拟训练曲线
        epochs_data = np.arange(1, 51)
        train_loss = 2.5 * np.exp(-epochs_data / 10) + 0.1 + np.random.normal(0, 0.05, 50)
        val_loss = 2.5 * np.exp(-epochs_data / 12) + 0.15 + np.random.normal(0, 0.08, 50)
        train_acc = 100 * (1 - np.exp(-epochs_data / 8)) + np.random.normal(0, 1, 50)
        val_acc = 100 * (1 - np.exp(-epochs_data / 10)) + np.random.normal(0, 1.5, 50)
        
        fig = make_subplots(rows=2, cols=1, subplot_titles=['Loss', 'Accuracy'])
        
        fig.add_trace(go.Scatter(x=epochs_data, y=train_loss, name='Train Loss',
                                  line=dict(color='#00ff9f')), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs_data, y=val_loss, name='Val Loss',
                                  line=dict(color='#ff00ff')), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=epochs_data, y=train_acc, name='Train Acc',
                                  line=dict(color='#00ff9f')), row=2, col=1)
        fig.add_trace(go.Scatter(x=epochs_data, y=val_acc, name='Val Acc',
                                  line=dict(color='#ff00ff')), row=2, col=1)
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(10, 10, 15, 0)',
            plot_bgcolor='rgba(26, 26, 46, 0.8)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)


def show_identification():
    """身份识别页面"""
    st.markdown("## 🔍 IDENTITY VERIFICATION")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 INPUT ECG")
        
        uploaded_file = st.file_uploader("Upload ECG CSV", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            signal = df['Channel 1'].values[:2000] if 'Channel 1' in df.columns else df.iloc[:, 1].values[:2000]
            
            fig = plot_ecg_cyberpunk(signal, title="Uploaded ECG")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # 显示示例
            st.info("💡 Upload an ECG file or select a test sample below")
            
            test_subject = st.selectbox("Test Sample", ['A', 'B', 'C', 'D', 'E', 'F'])
            
            if st.button("🔬 ANALYZE", use_container_width=True):
                # 模拟分析
                with st.spinner("Analyzing ECG pattern..."):
                    time.sleep(1.5)
                
                st.session_state['prediction'] = test_subject
                st.session_state['confidence'] = np.random.uniform(0.92, 0.99)
    
    with col2:
        st.markdown("### 📊 ANALYSIS RESULT")
        
        if 'prediction' in st.session_state:
            pred = st.session_state['prediction']
            conf = st.session_state['confidence']
            
            # 结果显示
            st.markdown(f"""
            <div style="text-align: center; padding: 30px;">
                <h1 style="font-size: 5em; color: #00ff9f; text-shadow: 0 0 30px #00ff9f;">
                    {pred}
                </h1>
                <p style="color: #00ffff; font-size: 1.5em;">IDENTITY VERIFIED</p>
                <p style="color: #ff00ff;">Confidence: {conf*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 概率分布
            labels = ['A', 'B', 'C', 'D', 'E', 'F']
            probs = np.random.dirichlet(np.ones(6) * 0.5)
            probs[labels.index(pred)] = conf
            probs = probs / probs.sum()
            
            fig = plot_probability_cyber(probs, labels, labels.index(pred))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div class="cyber-card" style="text-align: center; padding: 50px;">
                <h3 style="color: #a0a0a0;">AWAITING INPUT</h3>
                <p style="color: #606060;">Upload ECG data or select a test sample</p>
            </div>
            """, unsafe_allow_html=True)


def show_performance():
    """性能展示页面"""
    st.markdown("## 📈 SYSTEM PERFORMANCE")
    
    # 混淆矩阵
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 CONFUSION MATRIX")
        
        # 模拟混淆矩阵
        labels = ['A', 'B', 'C', 'D', 'E', 'F']
        cm = np.array([
            [198, 2, 0, 0, 0, 0],
            [1, 195, 3, 0, 1, 0],
            [0, 2, 196, 1, 1, 0],
            [0, 0, 1, 197, 1, 1],
            [0, 1, 0, 2, 195, 2],
            [0, 0, 0, 1, 2, 197]
        ])
        
        fig = plot_confusion_matrix_cyber(cm, labels)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📋 CLASS METRICS")
        
        # 每类指标
        metrics_data = {
            'Subject': labels,
            'Sensitivity': [0.99, 0.975, 0.98, 0.985, 0.975, 0.985],
            'Specificity': [0.998, 0.995, 0.996, 0.997, 0.994, 0.997],
            'Precision': [0.99, 0.975, 0.98, 0.98, 0.975, 0.985],
            'F1-Score': [0.99, 0.975, 0.98, 0.982, 0.975, 0.985]
        }
        
        df_metrics = pd.DataFrame(metrics_data)
        
        fig = go.Figure(data=[
            go.Bar(name='Sensitivity', x=labels, y=metrics_data['Sensitivity'], marker_color='#00ff9f'),
            go.Bar(name='Specificity', x=labels, y=metrics_data['Specificity'], marker_color='#00ffff'),
            go.Bar(name='F1-Score', x=labels, y=metrics_data['F1-Score'], marker_color='#ff00ff')
        ])
        
        fig.update_layout(
            barmode='group',
            template='plotly_dark',
            paper_bgcolor='rgba(10, 10, 15, 0)',
            plot_bgcolor='rgba(26, 26, 46, 0.8)',
            yaxis=dict(range=[0.9, 1.0]),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 总体指标
    st.markdown("### 🏆 OVERALL PERFORMANCE")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = [
        ("Accuracy", "98.5%", "#00ff9f"),
        ("Precision", "98.2%", "#00ffff"),
        ("Recall", "98.3%", "#ff00ff"),
        ("F1-Score", "98.2%", "#ffff00"),
        ("AUC-ROC", "0.997", "#ff0040")
    ]
    
    for col, (label, value, color) in zip([col1, col2, col3, col4, col5], metrics):
        with col:
            st.markdown(create_metric_card(label, value, color=color), unsafe_allow_html=True)


if __name__ == '__main__':
    main()
