# 先切换端口
# set HTTPS_PROXY=http://127.0.0.1:7897 一定要在cmd里切换到你vpn的端口，不然连不上。一定要在cmd里！！！
# streamlit run app.py 来启动
# --- 最终优化与修正版本 v4 ---

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
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 页面配置和标题 ---
st.set_page_config(
    page_title="AI运动教练 Demo",
    page_icon="🤖",
    layout="wide"
)
st.title("🤖 AI 运动教练")
st.caption("上传一段运动视频，让AI为你分析姿态。支持连续对话。")


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

# --- RAG Setup: 创建并缓存Retriever ---
@st.cache_resource
def get_retriever(api_key):
    try:
        loader = DirectoryLoader('./knowledge_base/', glob="**/*.md", show_progress=True)
        documents = loader.load()
        if not documents:
            st.warning("知识库为空，RAG功能将不会生效。请在 knowledge_base 文件夹中添加Markdown文件。")
            return None
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vectorstore = FAISS.from_documents(texts, embeddings)
        return vectorstore.as_retriever()
    except Exception as e:
        st.error(f"创建RAG索引时出错: {e}")
        return None

# --- LangChain 初始化 ---
try:
    if "llm" not in st.session_state:
        st.session_state.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            safety_settings=safety_settings,
        )
    if "history" not in st.session_state:
        st.session_state.history = []  # LangChain的消息历史
    if "analysis_df" not in st.session_state:
        st.session_state.analysis_df = pd.DataFrame() # 用于存储分析数据
    if "retriever" not in st.session_state:
        st.session_state.retriever = get_retriever(api_key) # 初始化RAG
except Exception as e:
    st.error(f"模型初始化失败，请检查API Key和网络代理。错误: {e}")
    st.stop()


# --- 侧边栏 UI ---
with st.sidebar:
    st.header("⚙️ 控制面板") # 优化点：使用更通用的图标
    
    # --- 优化点：改进"新建对话"交互 ---
    if st.button("✨ 新建对话", use_container_width=True):
        st.session_state.history = []
        st.session_state.analysis_df = pd.DataFrame()
        st.success("新的对话已开始！")
        time.sleep(0.5) # 短暂显示成功信息，然后刷新
        st.rerun()

    # --- 铜牌功能：增加用户目标输入框 ---
    user_goal = st.text_input("我的训练目标:", placeholder="例如：改善深蹲时膝盖内扣")
    st.divider()

    # --- 优化点：增加用户引导 ---
    uploaded_file = st.file_uploader(
        "上传你的运动视频",
        type=["mp4", "mov", "avi"],
        help="建议上传5-15秒的短视频，以获得最佳分析速度和效果。"
    )
    
    desired_frames = st.slider(
        "分析强度 (帧数)", # 优化点：标签更易懂
        min_value=2, max_value=10, value=6, step=1,
        help="选择从视频中抽取的关键画面数量。数量越多，分析越精细，但处理时间也更长。"
    )
    
    analyze_button = st.button("开始分析", use_container_width=True, disabled=not uploaded_file)


# --- 主聊天界面 ---

# --- 优化点：增加欢迎页/引导区，避免冷启动 ---
if not st.session_state.history:
    st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem;">
            <h2 style="font-weight: bold;">欢迎使用 AI 运动教练</h2>
            <p>我是您的专属AI教练，可以分析您上传的运动视频，提供专业的姿态评估和改进建议。</p>
            <p><strong>请按以下步骤开始：</strong></p>
            <ol style="display: inline-block; text-align: left; margin-top: 1rem;">
                <li>在左侧的 <strong>控制面板</strong> 上传您的运动视频。</li>
                <li>（可选）调整您希望分析的 <strong>关键帧数量</strong>。</li>
                <li>点击 <strong>"开始分析"</strong> 按钮，稍等片刻即可获得报告。</li>
            </ol>
            <p>期待看到您的精彩表现！</p>
        </div>
    """, unsafe_allow_html=True)

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
    with st.spinner("处理中，请稍候..."):
        st.info("AI正在分析...请稍候")
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
        st.write(f"视频总帧数: {total_frames}，将均匀抽取 {desired_frames} 帧进行分析。")
        
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
        st.write(f"图像提取完成！共提取 {len(sampled_frames_pil)} 帧。正在准备发送...")

        if not sampled_frames_pil:
            st.error("无法从视频中提取任何有效帧。")
        else:
            st.session_state.analysis_df = pd.DataFrame(quantitative_data)
            df = st.session_state.analysis_df
            # --- 华丽关键帧横向大图展示 ---
            st.markdown("#### 关键帧预览")
            cols = st.columns(len(sampled_frames_pil))
            for i, img in enumerate(sampled_frames_pil):
                with cols[i]:
                    st.image(img, caption=f"帧 {i+1}", use_container_width=True)

            # --- LangChain Prompt & Invocation ---
            focus_prompt = f"用户的训练目标是{user_goal}。如有相关问题请适当关注。" if user_goal else ""
            data_prompt = f"\n以下为部分帧的量化数据，仅供你分析时参考，重点请结合视频帧的多模态理解进行综合判断：\n{df.to_markdown(index=False)}\n" if not df.empty else ""
            prompt_text = f"""
            你是一位顶级的运动生物力学专家和AI教练。你的所有回答都必须使用简体中文。
            你的任务是基于用户上传的视频帧为主，结合部分量化数据（仅作辅助参考），提供一份专业、深入、以多模态理解为核心的分析报告。
            {focus_prompt}
            {data_prompt}

            **输出格式要求：**
            请严格按照以下三个部分进行组织：

            - **【综合评估与得分】**: 
              首先，**必须使用Markdown表格**清晰地展示四个维度的得分。表格应包含"评估维度"和"得分 (满分10)"两列。
              然后，在表格下方给出一个简短的总体评价。

            - **【多模态诊断】**: 
              这是报告的核心。请**以视频帧的多模态理解为主，量化数据仅作辅助**，逐项解释打分依据。
              例如："在第X帧图像中观察到..."。不要只围绕量化数据展开。

            - **【核心改进建议】**: 
              针对得分较低的维度和用户目标，提供最关键、最可操作的训练建议。

            你的语气应专业、严谨且富有鼓励性。请开始分析。
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
                st.info("AI大模型正在生成分析报告...")
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

                    # --- 分析完成后的图表和下载按钮 ---
                    if not df.empty:
                        with st.expander("📈 详细数据图表", expanded=True):
                            # 膝关节角度变化
                            fig_knee = go.Figure()
                            fig_knee.add_trace(go.Scatter(x=df['帧号'], y=df['左膝角度'], mode='lines+markers', name='左膝', line=dict(color='red', width=4), marker=dict(size=10)))
                            fig_knee.add_trace(go.Scatter(x=df['帧号'], y=df['右膝角度'], mode='lines+markers', name='右膝', line=dict(color='blue', width=4), marker=dict(size=10)))
                            fig_knee.update_layout(title='膝关节角度变化', xaxis_title='帧号', yaxis_title='角度 (°)', template='plotly_dark')
                            st.plotly_chart(fig_knee, use_container_width=True)
                            # 髋关节角度变化
                            fig_hip = go.Figure()
                            fig_hip.add_trace(go.Scatter(x=df['帧号'], y=df['左髋角度'], mode='lines+markers', name='左髋', line=dict(color='orange', width=4), marker=dict(size=10)))
                            fig_hip.add_trace(go.Scatter(x=df['帧号'], y=df['右髋角度'], mode='lines+markers', name='右髋', line=dict(color='green', width=4), marker=dict(size=10)))
                            fig_hip.update_layout(title='髋关节角度变化', xaxis_title='帧号', yaxis_title='角度 (°)', template='plotly_dark')
                            st.plotly_chart(fig_hip, use_container_width=True)
                            st.dataframe(df)
                    
                    if collected_messages:
                        st.download_button(
                            label="📥 下载本次分析报告",
                            data=collected_messages,
                            file_name=f"ai_coach_report_{time.strftime('%Y%m%d-%H%M%S')}.md",
                            mime="text/markdown",
                        )
                st.success("分析完成！")
            except Exception as e:
                error_str = str(e)
                if "safety" in error_str.lower() or "blocked" in error_str.lower():
                     st.error("请求被安全策略阻止，可能是图像或文本内容被误判。请尝试更换视频或调整提示。")
                else:
                     st.error(f"调用AI模型时发生错误: {e}")
                print(e)

# 仅在AI有回复后（即分析完成后）显示输入框
if st.session_state.history and isinstance(st.session_state.history[-1], AIMessage):
    if prompt := st.chat_input('可以继续向AI提问，例如"我的左腿应该注意什么？"'):
        user_prompt_message = HumanMessage(content=prompt)
        st.session_state.history.append(user_prompt_message)
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_container = st.empty()
            collected_messages = ""
            
            # --- RAG + Chat 逻辑 ---
            # 如果RAG可用，则使用RAG链；否则退回普通对话
            if st.session_state.retriever:
                rag_prompt_template = ChatPromptTemplate.from_template("""
                **请注意：** 你是一位专业的AI运动教练。请严格根据下面提供的"知识库上下文"来回答用户的问题。
                如果上下文中没有足够的信息来回答问题，请礼貌地告知用户"根据我现有的知识，我还无法回答这个问题"，不要尝试编造答案。
                你的所有回答都必须使用简体中文。

                **知识库上下文:**
                {context}

                **用户问题:**
                {input}
                """)
                
                document_chain = create_stuff_documents_chain(st.session_state.llm, rag_prompt_template)
                retrieval_chain = create_retrieval_chain(st.session_state.retriever, document_chain)
                
                # 流式传输RAG链的响应
                response_stream = retrieval_chain.stream({"input": prompt, "history": st.session_state.history})
                
                for chunk in response_stream:
                    if "answer" in chunk:
                        collected_messages += chunk["answer"]
                        response_container.markdown(collected_messages, unsafe_allow_html=True)
            else:
                 # RAG不可用时的普通对话
                response = st.session_state.llm.stream(st.session_state.history)
                for chunk in response:
                    if chunk.content:
                        collected_messages += chunk.content
                        response_container.markdown(collected_messages, unsafe_allow_html=True)
            
            # 将后续回复也加入历史
            st.session_state.history.append(AIMessage(content=collected_messages))

