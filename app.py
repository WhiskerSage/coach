# AI 运动教练 - V1
# 你的随身运动教练
#
# 启动方式：
# 先切换端口（如需代理）：set HTTPS_PROXY=http://127.0.0.1:7897
# streamlit run app.py 来启动

import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import time
from PIL import Image
import io
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import re
import base64
import json
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import TextLoader
import os
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from typing import List, Any

# --- 页面配置和标题 ---
st.set_page_config(
    page_title="AI COACH V1",
    page_icon=None,
    layout="wide"
)

# --- 高级 UI 设计与 CSS 注入 (仿 Bienville Capital 极简奢华风格) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;600&display=swap');

    /* 全局背景与字体 */
    .stApp {
        background-color: #0e0e0e; /* 更深邃的黑色 */
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }

    /* 隐藏 Streamlit 默认元素，但保留 Header 用于显示侧边栏按钮 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 隐藏顶部红线 */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* 调整侧边栏收缩后的按钮位置，确保它贴合左上角 */
    [data-testid="stSidebarCollapsedControl"] {
        position: fixed;
        top: 15px;
        left: 15px;
        width: 40px;
        height: 40px;
    }
    
    /* 侧边栏开关按钮默认隐藏，通过动画延迟显示 */
    [data-testid="stSidebarCollapsedControl"] {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 4px;
        transition: all 0.3s;
        z-index: 1000001 !important;
        opacity: 0;
        animation: fadeInButton 1s ease-in-out forwards;
        animation-delay: 7s; /* 延迟直到开场动画结束 */
    }
    
    @keyframes fadeInButton {
        to { opacity: 1; }
    }
    
    [data-testid="stSidebarCollapsedControl"]:hover {
        background-color: rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
    }

    /* 标题排版 */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    h1 {
        font-size: 3.5rem !important;
        color: #ffffff;
        margin-bottom: 0.5rem !important;
    }

    /* 侧边栏美化 - 恢复默认宽度，保持质感 */
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #333;
        /* 移除强制宽度设置，恢复 Streamlit 默认行为 */
    }

    /* 核心修复：确保侧边栏收起时完全隐藏，不留残影 */
    [data-testid="stSidebar"][aria-expanded="false"] {
        display: none !important;
    }

    /* 确保侧边栏收起时，主内容区域占满全宽 */
    [data-testid="stSidebar"][aria-expanded="false"] ~ .main,
    section[data-testid="stSidebar"][aria-expanded="false"] ~ .main {
        margin-left: 0 !important;
    }
    
    /* 侧边栏内的组件间距优化 */
    [data-testid="stSidebar"] .block-container {
        padding-top: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* 侧边栏标题 */
    [data-testid="stSidebar"] h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem !important;
        letter-spacing: 1px;
        margin-bottom: 2rem;
        color: #fff;
    }
    
    /* 侧边栏输入框与按钮 */
    [data-testid="stSidebar"] input {
        background-color: #1f1f1f !important;
        border: 1px solid #444 !important;
        padding: 10px !important;
    }
    
    /* File Uploader 美化 */
    [data-testid="stFileUploader"] {
        border: 1px dashed #555;
        border-radius: 4px;
        padding: 20px;
        background-color: #161616;
        transition: border-color 0.3s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #fff;
    }
    
    /* Expander 美化 */
    .streamlit-expanderHeader {
        background-color: #161616 !important;
        color: #ccc !important;
        border: 1px solid #333;
    }
    
    /* 自定义按钮风格 - 侧边栏触发器 */
    .sidebar-trigger {
        position: fixed;
        top: 20px;
        left: 20px;
        z-index: 9999;
        cursor: pointer;
        color: #fff;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        mix-blend-mode: difference;
    }
    
    /* 重点内容放大处理 */
    .highlight-text {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        color: #ffffff;
        line-height: 1.6;
        margin: 2rem 0;
        border-left: 3px solid #fff;
        padding-left: 20px;
    }
    
    /* 中文适配优化 */
    .cn-text {
        font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
    }

    /* 按钮美化 - 极简黑白 */
    .stButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 0px !important; /* 锐利直角 */
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 0.6rem 2rem !important;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #cccccc !important;
        transform: translateY(-2px);
    }

    /* 输入框美化 */
    .stTextInput > div > div > input {
        background-color: #1a1a1a;
        color: #fff;
        border: 1px solid #333;
        border-radius: 0px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #fff;
    }

    /* 优化主内容区域布局 - 完美居中且宽屏 */
    .stApp {
        display: flex;
        flex-direction: row;
    }

    .main {
        flex: 1;
        margin-left: 0 !important;
        padding-left: 0 !important;
        width: 100% !important;
        display: flex;
        justify-content: center;
    }

    .main .block-container {
        max-width: 90rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-top: 2rem !important;
        margin: 0 auto !important;
        width: 100% !important;
    }

    /* 视频容器居中 */
    video {
        width: 100% !important;
        display: block;
        margin: 0 auto;
    }

    /* 修复 Streamlit 可能的内部元素限制 */
    .stApp > header {
        background-color: transparent !important;
    }

    /* 自定义 Landing Page 容器 */
    .landing-container {
        padding: 4rem 2rem;
        text-align: left;
        width: 100%; /* 确保容器占满可用宽度 */
        max-width: 1000px; /* 限制内容最大宽度 */
        margin: 0 auto; /* 关键：在全宽父容器中居中 */
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 85vh;
    }
    
    .landing-hero-text {
        font-family: 'Playfair Display', serif;
        font-size: 5rem; /* 字体加大 */
        line-height: 1.1;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 2rem;
        max-width: 100%; /* 允许文字横向铺满 */
    }
    
    .landing-sub-text {
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        line-height: 1.6;
        color: #a0a0a0;
        margin-bottom: 3rem;
        max-width: 800px; /* 副标题可以稍窄，保持阅读舒适度 */
        border-left: 2px solid #fff;
        padding-left: 1.5rem;
    }
    
    .feature-list {
        display: flex;
        gap: 4rem; /* 增加间距 */
        flex-wrap: wrap;
        margin-top: 3rem;
        justify-content: flex-start; /* 靠左对齐或均匀分布 */
    }
    .feature-item {
        flex: 1;
        min-width: 200px;
    }
    .feature-title {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #666;
        margin-bottom: 0.5rem;
    }
    .feature-desc {
        font-size: 1.1rem;
        color: #ddd;
    }
    
    /* 模拟加载动画容器 - 全屏 Overlay */
    .loader-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-color: #000000;
        z-index: 999999; /* 确保覆盖所有 Streamlit 元素 */
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        text-align: center;
    }
    .loader-text {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        color: #fff;
        animation: fadeIn 1.5s ease-in-out;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }

</style>
""", unsafe_allow_html=True)


# --- Intro Animation Logic ---
if "intro_shown" not in st.session_state:
    st.session_state.intro_shown = False

def show_intro_animation():
    placeholder = st.empty()
    
    # Sequence 1
    html_1 = """
<div class="loader-overlay">
<div class="loader-text">YOUR BODY DOESN'T NEED MORE EXERCISE.</div>
</div>
"""
    placeholder.markdown(html_1, unsafe_allow_html=True)
    time.sleep(2.5)
    
    # Sequence 2
    html_2 = """
<div class="loader-overlay">
<div class="loader-text" style="color: #a0a0a0;">IT NEEDS <span style="color: #fff; font-style: italic;">SMARTER MOVEMENT.</span></div>
</div>
"""
    placeholder.markdown(html_2, unsafe_allow_html=True)
    time.sleep(2.5)
    
    # Sequence 3
    html_3 = """
<div class="loader-overlay">
<div class="loader-text" style="font-size: 5rem; letter-spacing: 5px;">AI COACH</div>
</div>
"""
    placeholder.markdown(html_3, unsafe_allow_html=True)
    time.sleep(2.0)
    
    placeholder.empty()
    st.session_state.intro_shown = True

# Run intro only once
if not st.session_state.intro_shown:
    show_intro_animation()
    st.rerun() # Rerun to load the main UI cleanly

# --- Main App Logic Starts Here ---
# st.title("🏃‍♂️ AI 运动教练 V1") # 移除默认标题，使用自定义Landing Page
# st.caption("你的随身运动教练，专业姿态分析与改进建议") # 移除默认caption



# --- Gemini API 配置 (自动从 secrets 读取) ---
api_key = st.secrets.get('GEMINI_API_KEY', None)
if not api_key:
    st.error("未检测到 Gemini API 密钥，请在 .streamlit/secrets.toml 中配置 GEMINI_API_KEY。")
    st.stop()

# --- 定义安全设置 ---
safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- 数据库功能 (JSON) ---
DB_FILE = "database.json"

def load_data():
    """从JSON文件中加载所有用户数据"""
    try:
        with open(DB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_data(user: str, report: str, df: pd.DataFrame):
    """保存一次分析会话到JSON文件"""
    try:
        all_data = load_data()
        if user not in all_data:
            all_data[user] = []

        # --- 核心修复：确保数据类型是JSON可序列化的 ---
        # 创建一个副本以避免修改原始的 session_state df
        df_copy = df.copy()
        # 遍历所有列，如果数据是浮点数类型，则转换为Python内置的float
        for col in df_copy.columns:
            if pd.api.types.is_float_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].astype(float)
        
        session_data = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "report": report,
            "dataframe_json": df_copy.to_json(orient='split')
        }
        all_data[user].append(session_data)

        with open(DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
        
        print(f"成功保存数据到 {DB_FILE}，用户: {user}")
    except Exception as e:
        print(f"保存数据时出错: {e}")
        raise e

# --- 金牌功能：定义数据分析工具 ---
@tool
def get_angle_extremes(joint: str) -> dict:
    """
    获取指定关节在运动过程中的最大和最小角度。
    当用户询问关于某个关节的"活动范围"、"最大弯曲角度"或"最小角度"时使用此工具。
    参数 `joint` 的可选值为 '膝' 或 '髋'。
    """
    if 'analysis_df' not in st.session_state or st.session_state.analysis_df.empty:
        return {"error": "无可用分析数据。请先上传并分析一个视频。"}
    df = st.session_state.analysis_df
    try:
        left_col, right_col = f'左{joint}角度', f'右{joint}角度'
        min_angle = df[[left_col, right_col]].min().min()
        max_angle = df[[left_col, right_col]].max().max()
        return {"关节": joint, "最小角度": f"{min_angle:.2f}°", "最大角度": f"{max_angle:.2f}°"}
    except KeyError:
        return {"error": f"数据格式错误，找不到'{joint}'关节的角度数据。"}

@tool
def get_max_angle_difference(joint: str) -> dict:
    """
    计算在整个运动过程中，左右同名关节（如左膝和右膝）在任意一帧的最大角度差。
    当用户询问关于身体"对称性"、"左右差异"或"两边不一致"等问题时，使用此工具。
    参数 `joint` 的可选值为 '膝' 或 '髋'。
    """
    if 'analysis_df' not in st.session_state or st.session_state.analysis_df.empty:
        return {"error": "无可用分析数据。请先上传并分析一个视频。"}
    df = st.session_state.analysis_df
    try:
        left_col, right_col = f'左{joint}角度', f'右{joint}角度'
        df['diff'] = (df[left_col] - df[right_col]).abs()
        max_diff = df['diff'].max()
        return {"关节": joint, "最大角度差": f"{max_diff:.2f}°"}
    except KeyError:
        return {"error": f"数据格式错误，找不到'{joint}'关节的角度数据。"}


# --- RAG Setup: 创建并缓存Retriever ---
@st.cache_resource
def get_retriever(api_key):
    """
    创建知识库检索器（可选功能）
    如果创建失败，不影响主要的视频分析功能
    """
    try:
        # 手动加载 knowledge_base 目录下的所有 .md 文件
        kb_path = './knowledge_base/'
        md_files = glob.glob(os.path.join(kb_path, '**/*.md'), recursive=True)

        if not md_files:
            print("知识库为空，RAG功能将不会生效。")
            return None

        documents = []
        for file_path in md_files:
            try:
                # 使用 TextLoader 加载每个文件，指定编码为 utf-8
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {e}")
                continue

        if not documents:
            print("无法加载知识库文件，RAG功能将不会生效。")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        print(f"正在创建知识库向量索引（这可能需要一些时间）...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        # 使用FAISS创建向量存储
        vectorstore = FAISS.from_documents(texts, embeddings)
        print(f"✓ 成功加载 {len(md_files)} 个知识库文件，共 {len(texts)} 个文本块")
        return vectorstore.as_retriever()
    except Exception as e:
        # 配额超限或其他错误时，只打印警告，不中断应用
        error_msg = str(e)
        if "quota" in error_msg.lower() or "429" in error_msg:
            print(f"⚠ RAG功能暂时不可用: API配额已达上限")
            print(f"  提示: RAG功能需要 embedding API 配额，主要的视频分析功能不受影响")
        else:
            print(f"⚠ RAG功能初始化失败: {error_msg}")
        return None

# --- LangChain 初始化 ---
try:
    if "llm" not in st.session_state:
        st.session_state.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", 
            google_api_key=api_key,
            safety_settings=safety_settings,
        )
    if "history" not in st.session_state:
        st.session_state.history = []  # LangChain的消息历史
    if "analysis_df" not in st.session_state:
        st.session_state.analysis_df = pd.DataFrame() # 用于存储分析数据
    if "retriever" not in st.session_state:
        # 初始化RAG（可选功能，失败不影响主要功能）
        with st.spinner("正在初始化知识库..."):
            st.session_state.retriever = get_retriever(api_key)
            if st.session_state.retriever is None:
                st.info("💡 提示：知识库功能当前不可用（可能是API配额限制），但不影响视频分析功能。")
    if "agent_executor" not in st.session_state:
        # --- 将RAG包装成一个工具 ---
        retriever_tool = None
        if st.session_state.retriever:
            # 创建一个可以处理历史对话的检索器
            contextualize_q_system_prompt = """
            Given a chat history and the latest user question 
            which might reference context in the chat history, 
            formulate a standalone question which can be understood 
            without the chat history. Do NOT answer the question, 
            just reformulate it if needed and otherwise return it as is.
            """
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            history_aware_retriever = create_history_aware_retriever(
                st.session_state.llm, st.session_state.retriever, contextualize_q_prompt
            )
            # 创建文档链
            qa_system_prompt = """
            You are a professional AI sports coach. Please answer the user's questions strictly based on the "Knowledge Base Context" provided below.
            If the context does not contain enough information to answer the question, politely inform the user "Based on my current knowledge, I cannot answer this question yet," and do not attempt to fabricate an answer.
            All your answers must be in English.
            
            Knowledge Base Context:
            {context}
            """
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", qa_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            question_answer_chain = create_stuff_documents_chain(st.session_state.llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            # 定义工具
            @tool
            def knowledge_base_retriever(input: str, chat_history: List[Any]) -> str:
                """
                Use this tool when the user asks general questions about sports, fitness, nutrition, recovery, etc.
                Do not use it to answer questions about specific video analysis results.
                """
                response = rag_chain.invoke({"input": input, "chat_history": chat_history})
                return response['answer']
            retriever_tool = knowledge_base_retriever
        
        # --- 创建Agent ---
        tools = [get_angle_extremes, get_max_angle_difference]
        # 暂时注释掉RAG工具以避免参数问题
        # if retriever_tool:
        #     tools.append(retriever_tool)

        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一位专业的AI运动教练。你可以调用工具来查询知识库或分析数据。你的所有回答都必须使用简体中文。"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_tool_calling_agent(st.session_state.llm, tools, agent_prompt)
        st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

except Exception as e:
    st.error(f"模型或Agent初始化失败: {e}")
    st.stop()


# --- 侧边栏 UI ---
with st.sidebar:
    st.header("控制面板") 
    
    # --- 用户系统 ---
    username = st.text_input("用户名称", placeholder="输入您的名字用于存档")
    
    with st.expander("训练目标 (可选)"):
        user_goal = st.text_input("我的目标:", placeholder="例如：改善深蹲时的膝盖内扣")

    st.divider()

    st.subheader("上传与分析")
    # --- 优化点：增加用户引导 ---
    uploaded_file = st.file_uploader(
        "上传视频",
        help="建议：上传 5-15 秒的短视频以获得最佳效果。"
    )

    # --- 核心修复：手动进行文件类型验证以绕过Streamlit的bug ---
    is_valid_file = False
    if uploaded_file is not None:
        # 获取文件名
        file_name = uploaded_file.name
        # 检查文件扩展名
        allowed_extensions = ['.mp4', '.mov', '.avi']
        if any(file_name.lower().endswith(ext) for ext in allowed_extensions):
            is_valid_file = True
        else:
            st.error(f"格式无效。支持的格式: {', '.join(allowed_extensions)}")
    
    desired_frames = st.number_input(
        "分析帧数",
        min_value=2, max_value=30, value=6, step=1,
        help="提取的关键帧数量。建议 5-20 帧。"
    )
    
    analyze_button = st.button(
        "开始分析", 
        use_container_width=True, 
        disabled=not (uploaded_file and username and is_valid_file)
    )
    if not username:
        st.warning("请输入您的名字以启用分析。")


# --- 辅助函数：加载视频 ---
@st.cache_resource
def get_video_base64(video_path):
    try:
        with open(video_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# --- 主聊天界面 ---


# --- 优化点：增加欢迎页/引导区，避免冷启动 ---
if not st.session_state.history:
    # 加载宣传视频
    video_path = "/Users/boannn/codes/coach/宣传片.mp4"
    video_b64 = get_video_base64(video_path)
    video_html = ""
    if video_b64:
        video_html = f"""
<div style="margin: 3rem 0; border-radius: 0px; overflow: hidden; border: 1px solid #333;">
<video autoplay loop muted playsinline width="100%" style="display: block; opacity: 0.8;">
<source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
</video>
</div>
"""

    html_content = f"""
<div class="landing-container">
<div class="landing-hero-text">
YOUR BODY DOESN'T NEED MORE EXERCISE.<br>
IT NEEDS <span style="font-style: italic; color: #888;">SMARTER MOVEMENT.</span>
</div>
<div class="highlight-text cn-text">
我们不仅仅是记录运动，更是在解码人体力学。<br>
结合计算机视觉与生成式 AI，为您提供可量化的专业洞察。
</div>
{video_html}
<div class="landing-divider"></div>
<div class="feature-list">
<div class="feature-item">
<div class="feature-title">01. CAPTURE (拍摄)</div>
<div class="feature-desc cn-text">上传您的运动视频，支持任意角度与动作。</div>
</div>
<div class="feature-item">
<div class="feature-title">02. ANALYZE (分析)</div>
<div class="feature-desc cn-text">精准的骨骼追踪与关节角度量化。</div>
</div>
<div class="feature-item">
<div class="feature-title">03. EVOLVE (进化)</div>
<div class="feature-desc cn-text">获取 AI 提供的专业改进建议与训练计划。</div>
</div>
</div>
<div style="margin-top: 4rem; color: #666; font-size: 0.8rem; letter-spacing: 1px;" class="cn-text">
请点击左上角箭头展开侧边栏开始体验 &rarr;
</div>
</div>
"""
    st.markdown(html_content, unsafe_allow_html=True)


# 显示历史对话记录
for message in st.session_state.history:
    # 不显示初始的多模态用户消息，只显示AI回复和后续文本对话
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content, unsafe_allow_html=True)
    elif isinstance(message, HumanMessage) and isinstance(message.content, str):
         with st.chat_message("user"):
            st.markdown(message.content, unsafe_allow_html=True)

# --- 金牌功能：新增角度计算函数 ---
def calculate_angle(a, b, c):
    """计算由三点a, b, c构成的角度（b为顶点），返回0-180之间的角度值"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- 核心处理逻辑 ---
if analyze_button:
    # --- 金牌功能：初始化用于存储量化数据的字典 ---
    quantitative_data = {
        "帧号": [],
        "左膝角度": [],
        "右膝角度": [],
        "左髋角度": [],
        "右髋角度": []
    }
    with st.spinner("处理中..."):
        st.info("AI 正在分析...")
        # --- 视频处理逻辑 ---
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            st.error("无法读取视频，请检查文件是否损坏。")
            st.stop()
        
        frame_interval = max(total_frames // desired_frames, 1)
        st.write(f"视频总帧数: {total_frames}。正在提取 {desired_frames} 个关键帧。")
        
        sampled_frames_pil = []
        frame_indices_to_extract = [i * frame_interval for i in range(desired_frames)]
        
        with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            for frame_index in frame_indices_to_extract:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                annotated_image = frame.copy()
                
                if results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        annotated_image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                    )
                
                # --- 金牌功能：计算角度并存储 ---
                try:
                    landmarks = results.pose_landmarks.landmark
                    left_hip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_ankle = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y]
                    left_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    right_hip = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_ankle = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    right_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    lk_angle = calculate_angle(left_hip, left_knee, left_ankle)
                    rk_angle = calculate_angle(right_hip, right_knee, right_ankle)
                    lh_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                    rh_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                    quantitative_data["帧号"].append(frame_index)
                    quantitative_data["左膝角度"].append(lk_angle)
                    quantitative_data["右膝角度"].append(rk_angle)
                    quantitative_data["左髋角度"].append(lh_angle)
                    quantitative_data["右髋角度"].append(rh_angle)
                except Exception as e:
                    print(f"在帧 {frame_index} 计算角度时出错: {e}")
                    pass
                
                pil_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
                sampled_frames_pil.append(pil_image)
        
        cap.release()
        st.write(f"提取完成。共捕获 {len(sampled_frames_pil)} 帧。")

        if not sampled_frames_pil:
            st.error("未提取到有效帧。")
        else:
            st.session_state.analysis_df = pd.DataFrame(quantitative_data)
            df = st.session_state.analysis_df
            # --- 华丽关键帧横向大图展示 ---
            st.markdown("#### 关键帧预览")
            cols = st.columns(len(sampled_frames_pil))
            for i, img in enumerate(sampled_frames_pil):
                with cols[i]:
                    st.image(img, caption=f"第 {i+1} 帧", use_container_width=True)

            # --- LangChain Prompt & Invocation ---
            focus_prompt = f"用户的训练目标是{user_goal}。如有相关问题请适当关注。" if user_goal else ""
            data_prompt = f"\n以下为部分帧的量化数据，仅供你分析时参考，重点请结合视频帧的多模态理解进行综合判断：\n{df.to_markdown(index=False)}\n" if not df.empty else ""
            prompt_text = f"""
            You are a top-tier sports biomechanics expert and AI coach. All your responses must be in Simplified Chinese.
            Your task is to provide a professional, in-depth analysis report based primarily on the user-uploaded video frames, with some quantitative data (for reference only) and a focus on multi-modal understanding.
            {focus_prompt}
            {data_prompt}

            **Output Format Requirements:**
            Please strictly organize your response into the following three sections:

            - **【综合评估与得分】**: 
              First, you **must use a Markdown table** to clearly display scores for four dimensions. The table should include "评估维度" (Assessment Dimension) and "得分 (满分10)" (Score out of 10) columns.
              Then, provide a brief overall evaluation below the table.

            - **【多模态诊断】**: 
              This is the core of the report. Please **focus on the multi-modal understanding of video frames, using quantitative data only as auxiliary support**, and explain the basis for your scoring item by item.
              For example: "在第X帧中观察到...". Do not just discuss quantitative data.

            - **【核心改进建议】**: 
              Provide the most critical and actionable training suggestions targeting the lower-scoring dimensions and the user's goals.

            Your tone should be professional, rigorous, and encouraging. Please begin your analysis.
            """
            
            # 将PIL图像转换为LangChain所需格式
            image_messages = []
            for img in sampled_frames_pil:
                img.thumbnail((768, 768))
                buf = io.BytesIO()
                img.save(buf, format='JPEG')
                base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')
                image_messages.append({
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                })

            # 创建LangChain的多模态消息
            user_message = HumanMessage(
                content=[{"type": "text", "text": prompt_text}] + image_messages
            )
            
            # 清空历史，开始新的分析会话
            st.session_state.history = []

            try:
                st.info("正在生成报告...")
                with st.chat_message("assistant"):
                    response_container = st.empty()
                    collected_messages = ""
                    # 使用LangChain LLM进行流式调用
                    response = st.session_state.llm.stream([user_message])
                    for chunk in response:
                        if chunk.content:
                            collected_messages += chunk.content
                            response_container.markdown(collected_messages, unsafe_allow_html=True)

                    # 将AI的完整回复存入历史记录
                    st.session_state.history.append(AIMessage(content=collected_messages))

                    # --- 核心修复：自动存档分析结果 ---
                    if collected_messages and username:
                        try:
                            save_data(username, collected_messages, df)
                            st.success(f"已为用户 {username} 存档分析结果")
                        except Exception as e:
                            st.error(f"存档失败: {e}")
                            print(f"存档错误详情: {e}")

                    # --- 分析完成后的图表和下载按钮 ---
                    if not df.empty:
                        st.write("---")
                        st.subheader("详细数据指标")
                        
                        # 膝关节角度变化
                        fig_knee = go.Figure()
                        fig_knee.add_trace(go.Scatter(
                            x=df['帧号'], 
                            y=df['左膝角度'], 
                            mode='lines+markers', 
                            name='左膝', 
                            line=dict(color='red', width=4), 
                            marker=dict(size=10)
                        ))
                        fig_knee.add_trace(go.Scatter(
                            x=df['帧号'], 
                            y=df['右膝角度'], 
                            mode='lines+markers', 
                            name='右膝', 
                            line=dict(color='blue', width=4), 
                            marker=dict(size=10)
                        ))
                        fig_knee.update_layout(
                            title='膝关节角度', 
                            xaxis_title='帧号', 
                            yaxis_title='角度 (°)', 
                            template='plotly_dark',
                            height=400
                        )
                        st.plotly_chart(fig_knee, use_container_width=True, key="knee_chart")
                        
                        # 髋关节角度变化
                        fig_hip = go.Figure()
                        fig_hip.add_trace(go.Scatter(
                            x=df['帧号'], 
                            y=df['左髋角度'], 
                            mode='lines+markers', 
                            name='左髋', 
                            line=dict(color='orange', width=4), 
                            marker=dict(size=10)
                        ))
                        fig_hip.add_trace(go.Scatter(
                            x=df['帧号'], 
                            y=df['右髋角度'], 
                            mode='lines+markers', 
                            name='右髋', 
                            line=dict(color='green', width=4), 
                            marker=dict(size=10)
                        ))
                        fig_hip.update_layout(
                            title='髋关节角度', 
                            xaxis_title='帧号', 
                            yaxis_title='角度 (°)', 
                            template='plotly_dark',
                            height=400
                        )
                        st.plotly_chart(fig_hip, use_container_width=True, key="hip_chart")
                        
                        with st.expander("查看原始数据"):
                            st.dataframe(df, use_container_width=True)
                    
                    # 只保留下载功能
                    if collected_messages:
                        st.download_button(
                            label="下载报告",
                            data=collected_messages,
                            file_name=f"ai_coach_report_{username}_{time.strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                st.success("分析完成！")
            except Exception as e:
                error_str = str(e)
                if "safety" in error_str.lower() or "blocked" in error_str.lower():
                     st.error("请求被安全设置阻止。请尝试其他视频。")
                else:
                     st.error(f"AI 错误: {e}")
                print(e)

# 仅在AI有回复后（即分析完成后）显示输入框
if st.session_state.history and isinstance(st.session_state.history[-1], AIMessage):
    if prompt := st.chat_input('关于分析结果，您想问 AI 什么...'):
        user_prompt_message = HumanMessage(content=prompt)
        st.session_state.history.append(user_prompt_message)
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_container = st.empty()
            collected_messages = ""
            
            # --- Agent 逻辑 ---
            try:
                response_stream = st.session_state.agent_executor.stream({
                    "input": prompt,
                    "chat_history": st.session_state.history
                })
                for chunk in response_stream:
                    if "output" in chunk:
                        collected_messages = chunk['output']
                        response_container.markdown(collected_messages, unsafe_allow_html=True)
            except Exception as e:
                collected_messages = f"调用Agent时出错: {e}"
                response_container.markdown(collected_messages)

            # 将后续回复也加入历史
            st.session_state.history.append(AIMessage(content=collected_messages))

