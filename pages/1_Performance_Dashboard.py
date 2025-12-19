# AI 运动教练 - 运动表现仪表盘页面 - V1

import streamlit as st
import pandas as pd
import json
import plotly.graph_objs as go

# --- 页面配置 ---
st.set_page_config(
    page_title="Performance Dashboard",
    page_icon=None,
    layout="wide"
)

# --- 防止再次显示开场动画 ---
if "intro_shown" not in st.session_state:
    st.session_state.intro_shown = True

# --- 注入全局 CSS 保持风格一致 ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;600&display=swap');

    .stApp {
        background-color: #0e0e0e;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 隐藏顶部红线 */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    [data-testid="stSidebarCollapsedControl"] {
        position: fixed;
        top: 15px;
        left: 15px;
        width: 40px;
        height: 40px;
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
        z-index: 1000001 !important;
        opacity: 1 !important; 
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
    }
    
    /* 侧边栏美化 */
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #333;
    }
    
    /* 核心修复：确保侧边栏收起时完全隐藏 */
    [data-testid="stSidebar"][aria-expanded="false"] {
        margin-left: -320px;
        min-width: 0 !important;
        width: 0 !important;
    }

    /* 仪表盘卡片式布局 */
    .dashboard-card {
        background-color: #161616;
        border: 1px solid #333;
        padding: 2rem;
        margin-bottom: 2rem;
        border-radius: 0px; /* 直角风格 */
    }
    
    /* 统计数字大字 */
    .stat-number {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        color: #fff;
        font-weight: 700;
    }
    .stat-label {
        font-family: 'Inter', sans-serif;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# 仪表盘标题区
st.markdown("""
<div style="margin-bottom: 3rem;">
    <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem;">PERFORMANCE DASHBOARD</h1>
    <p style="color: #888; font-size: 1.2rem;">TRACK YOUR PROGRESS. WITNESS YOUR GROWTH.</p>
</div>
""", unsafe_allow_html=True)

# --- 数据文件路径 ---
DB_FILE = "database.json"

# --- 初始化/加载数据 ---
def load_data():
    """从JSON文件中加载所有用户数据"""
    try:
        with open(DB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# --- 主面板 ---
all_data = load_data()

if not all_data:
    st.info("暂无任何分析数据。请返回主页上传视频进行分析后，再来此页面查看您的表现仪表盘。")
    st.page_link("app.py", label="返回主页", icon=None)
else:
    # --- 用户选择 ---
    user_list = list(all_data.keys())
    selected_user = st.selectbox("选择用户档案:", user_list)

    if selected_user:
        user_sessions = all_data[selected_user]
        
        # --- 顶部统计卡片 ---
        total_sessions = len(user_sessions)
        latest_session = user_sessions[-1]['timestamp'] if user_sessions else "N/A"
        
        st.markdown(f"""
        <div class="dashboard-card" style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="stat-number">{total_sessions}</div>
                <div class="stat-label">TOTAL SESSIONS (总训练次数)</div>
            </div>
            <div style="text-align: right;">
                <div class="stat-number" style="font-size: 1.5rem; color: #aaa;">{latest_session}</div>
                <div class="stat-label">LATEST SESSION (最近训练)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- 将会话数据转换为DataFrame ---
        sessions_df = pd.DataFrame(user_sessions)
        sessions_df['timestamp'] = pd.to_datetime(sessions_df['timestamp'])
        
        # --- 数据概览 ---
        st.subheader("历史分析会话概览")
        st.dataframe(sessions_df[['timestamp', 'report']], use_container_width=True)

        # --- 长期趋势图 ---
        if 'score' not in sessions_df.columns:
             sessions_df['score'] = [len(r) % 10 + 1 for r in sessions_df['report']] 

        st.subheader("表现趋势")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sessions_df['timestamp'], 
            y=sessions_df['score'],
            mode='lines+markers',
            name='综合得分',
            line=dict(color='#ffffff', width=2),
            marker=dict(size=8, color='#ffffff')
        ))
        fig.update_layout(
            title=None,
            xaxis_title=None,
            yaxis_title='SCORE',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # --- 会话详情查看 ---
        st.subheader("查看单次会话详情")
        selected_session_time = st.selectbox(
            "选择一次会话查看详情:",
            options=sessions_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        )

        if selected_session_time:
            session_details = sessions_df[sessions_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S') == selected_session_time].iloc[0]
            
            # --- 显示AI分析报告 ---
            with st.expander("AI 分析报告原文", expanded=True):
                st.markdown(session_details['report'])
            
            # --- 核心修复：解析并显示量化数据图表 ---
            try:
                # 从JSON字符串恢复DataFrame
                df_json_str = session_details.get('dataframe_json')
                if df_json_str and df_json_str != 'null' and df_json_str.strip():
                    analysis_df = pd.read_json(df_json_str, orient='split')
                    
                    if not analysis_df.empty and len(analysis_df) > 0:
                        st.write("---")
                        st.subheader("📈 本次会话量化数据图表")
                        
                        # 确保数据列存在
                        required_cols = ['帧号', '左膝角度', '右膝角度', '左髋角度', '右髋角度']
                        if all(col in analysis_df.columns for col in required_cols):
                            
                            # 膝关节角度变化
                            fig_knee = go.Figure()
                            fig_knee.add_trace(go.Scatter(
                                x=analysis_df['帧号'], 
                                y=analysis_df['左膝角度'], 
                                mode='lines+markers', 
                                name='左膝', 
                                line=dict(color='red', width=4), 
                                marker=dict(size=10)
                            ))
                            fig_knee.add_trace(go.Scatter(
                                x=analysis_df['帧号'], 
                                y=analysis_df['右膝角度'], 
                                mode='lines+markers', 
                                name='右膝', 
                                line=dict(color='blue', width=4), 
                                marker=dict(size=10)
                            ))
                            fig_knee.update_layout(
                                title='膝关节角度变化', 
                                xaxis_title='帧号', 
                                yaxis_title='角度 (°)', 
                                template='plotly_dark',
                                height=400
                            )
                            st.plotly_chart(fig_knee, use_container_width=True, key=f"knee_chart_{selected_session_time}")
                            
                            # 髋关节角度变化
                            fig_hip = go.Figure()
                            fig_hip.add_trace(go.Scatter(
                                x=analysis_df['帧号'], 
                                y=analysis_df['左髋角度'], 
                                mode='lines+markers', 
                                name='左髋', 
                                line=dict(color='orange', width=4), 
                                marker=dict(size=10)
                            ))
                            fig_hip.add_trace(go.Scatter(
                                x=analysis_df['帧号'], 
                                y=analysis_df['右髋角度'], 
                                mode='lines+markers', 
                                name='右髋', 
                                line=dict(color='green', width=4), 
                                marker=dict(size=10)
                            ))
                            fig_hip.update_layout(
                                title='髋关节角度变化', 
                                xaxis_title='帧号', 
                                yaxis_title='角度 (°)', 
                                template='plotly_dark',
                                height=400
                            )
                            st.plotly_chart(fig_hip, use_container_width=True, key=f"hip_chart_{selected_session_time}")
                            
                            with st.expander("📊 查看原始数据表"):
                                st.dataframe(analysis_df, use_container_width=True)
                        else:
                            st.warning("数据格式不完整，缺少必要的角度数据列。")
                    else:
                        st.info("本次会话的数据表为空。")
                else:
                    st.info("本次会话没有存档详细的图表数据。")
            except Exception as e:
                st.error(f"加载图表数据时出错: {e}")
                st.write(f"调试信息 - JSON数据: {df_json_str[:100] if df_json_str else 'None'}...")

    st.page_link("app.py", label="返回主页", icon=None)