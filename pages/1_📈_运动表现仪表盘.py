import streamlit as st
import pandas as pd
import json
import plotly.graph_objs as go

# --- 页面配置 ---
st.set_page_config(
    page_title="运动表现仪表盘",
    page_icon="📈",
    layout="wide"
)

st.title("📈 运动表现仪表盘")
st.caption("在这里追踪您的每一次进步，见证自己的成长。")

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
    st.page_link("app.py", label="返回主页", icon="🏠")
else:
    # --- 用户选择 ---
    user_list = list(all_data.keys())
    selected_user = st.selectbox("选择要查看的用户档案:", user_list)

    if selected_user:
        user_sessions = all_data[selected_user]
        st.header(f"用户: {selected_user} 的表现报告")

        # --- 将会话数据转换为DataFrame ---
        sessions_df = pd.DataFrame(user_sessions)
        sessions_df['timestamp'] = pd.to_datetime(sessions_df['timestamp'])
        
        # --- 数据概览 ---
        st.subheader("历史分析会话概览")
        st.dataframe(sessions_df[['timestamp', 'report']], use_container_width=True)

        # --- 长期趋势图 (示例：假设报告中可以提取一个'综合得分') ---
        # 注意：这需要您的AI分析报告中有一个可以被正则提取的稳定得分项
        # for simplicity, let's plot a dummy score for now.
        # you would need to parse the 'report' column to get real scores.
        if 'score' not in sessions_df.columns:
             sessions_df['score'] = [len(r) % 10 + 1 for r in sessions_df['report']] # Dummy score

        st.subheader("表现趋势")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sessions_df['timestamp'], 
            y=sessions_df['score'],
            mode='lines+markers',
            name='综合得分'
        ))
        fig.update_layout(
            title='综合得分长期趋势',
            xaxis_title='日期',
            yaxis_title='得分 (1-10)',
            template='plotly_dark'
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
            with st.expander("AI 分析报告原文", expanded=True):
                st.markdown(session_details['report'])
            
            # 您可以在这里添加更多关于单次会话的详细数据展示，例如图表等
            st.write("---")

    st.page_link("app.py", label="返回主页", icon="🏠") 