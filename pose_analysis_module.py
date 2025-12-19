"""
姿态分析模块

该模块提供一个函数 analyze_video，用于接收一个视频文件路径，
执行姿态估计、量化分析，并调用AI模型生成分析报告。

用法示例:
    from pose_analysis_module import analyze_video

    video_path = "path/to/your/video.mp4"
    result = analyze_video(video_path)
    print(result['report'])
    # result['dataframe'] 包含量化数据
    # result['sampled_frames_pil'] 包含关键帧图像列表
"""

import os
import sys
import cv2
import mediapipe as mp
import tempfile
import time
import io
import numpy as np
import pandas as pd
import base64
import json
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional

# --- 导入必要的 LangChain 和 Google GenAI 组件 ---
# 注意：运行此脚本需要安装 requirements.txt 中列出的依赖项
# pip install -r requirements.txt

# 为了简化，我们直接在这里导入。在实际项目中，你可能需要配置API密钥管理。
# 请确保在调用此模块前，环境变量或配置中已设置 GEMINI_API_KEY。

from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# --- 定义安全设置 ---
safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- 角度计算函数 ---
def calculate_angle(a: List[float], b: List[float], c: List[float]) -> float:
    """
    计算由三点a, b, c构成的角度（b为顶点），返回0-180之间的角度值。
    参数 a, b, c 是包含 [x, y] 坐标的列表。
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- 核心姿态分析函数 ---
def analyze_video(
    video_path: str, 
    desired_frames: int = 6,
    api_key: Optional[str] = None,
    user_goal: str = ""
) -> Dict[str, Any]:
    """
    分析给定视频文件的运动姿态。

    Args:
        video_path (str): 视频文件的路径。
        desired_frames (int): 希望抽取的关键帧数量。默认为6。
        api_key (str, optional): Google Gemini API 密钥。如果未提供，将尝试从环境变量 GEMINI_API_KEY 获取。
        user_goal (str): 用户的训练目标，用于AI分析时参考。

    Returns:
        dict: 包含分析结果的字典。
            - 'report' (str): AI生成的分析报告文本。
            - 'dataframe' (pd.DataFrame): 包含量化数据的DataFrame。
            - 'sampled_frames_pil' (List[Image.Image]): 关键帧图像列表。
            - 'success' (bool): 分析是否成功。
            - 'error' (str, optional): 如果失败，包含错误信息。
            
    Raises:
        Exception: 如果在处理过程中发生不可恢复的错误。
    """
    
    # 1. 初始化返回结果
    result = {
        'report': '',
        'dataframe': pd.DataFrame(),
        'sampled_frames_pil': [],
        'success': False,
        'error': None
    }

    try:
        # 2. 验证视频文件
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        if not api_key:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("未提供 Google Gemini API 密钥。请通过参数传递或设置环境变量 GEMINI_API_KEY。")

        # 3. 初始化用于存储量化数据的字典
        quantitative_data = {
            "帧号": [],
            "左膝角度": [],
            "右膝角度": [],
            "左髋角度": [],
            "右髋角度": []
        }
        
        print(f"开始处理视频: {video_path}")
        print(f"目标关键帧数: {desired_frames}")

        # 4. 视频处理逻辑
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError("无法读取视频，请检查文件是否损坏。")
        
        frame_interval = max(total_frames // desired_frames, 1)
        print(f"视频总帧数: {total_frames}，将均匀抽取 {desired_frames} 帧进行分析。")
        
        sampled_frames_pil = []
        frame_indices_to_extract = [i * frame_interval for i in range(desired_frames)]
        
        with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            for frame_index in frame_indices_to_extract:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    print(f"警告: 无法读取帧 {frame_index}")
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
                
                # 计算角度并存储
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
                    # 即使一帧出错，也继续处理其他帧
                    pass
                
                pil_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
                sampled_frames_pil.append(pil_image)
        
        cap.release()
        print(f"图像提取完成！共提取 {len(sampled_frames_pil)} 帧。")

        if not sampled_frames_pil:
            raise ValueError("无法从视频中提取任何有效帧。")
            
        # 5. 准备数据
        df = pd.DataFrame(quantitative_data)
        result['dataframe'] = df
        result['sampled_frames_pil'] = sampled_frames_pil
        
        # 6. 准备发送给AI模型的数据
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
            # 缩略图以减小大小
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
        
        # 7. 调用AI模型
        print("正在调用AI模型生成分析报告...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",  # 使用 -latest 后缀
            google_api_key=api_key,
            safety_settings=safety_settings,
        )
        
        collected_messages = ""
        response = llm.stream([user_message])
        for chunk in response:
            if chunk.content:
                collected_messages += chunk.content
        
        result['report'] = collected_messages
        result['success'] = True
        print("分析完成！")
        return result

    except Exception as e:
        error_msg = f"分析过程中发生错误: {e}"
        print(error_msg)
        result['error'] = error_msg
        return result


# --- 示例用法 (如果直接运行此脚本) ---
if __name__ == "__main__":
    # 注意：直接运行此脚本需要设置环境变量 GEMINI_API_KEY
    # 示例: set GEMINI_API_KEY=your_actual_api_key_here && python pose_analysis_module.py
    
    # 你可以在这里替换为你的视频路径进行测试
    test_video_path = "跑步.mp4" # 假设在项目根目录
    
    if os.path.exists(test_video_path):
        analysis_result = analyze_video(test_video_path)
        if analysis_result['success']:
            print("\n--- AI分析报告 ---")
            print(analysis_result['report'])
            print("\n--- 量化数据 ---")
            print(analysis_result['dataframe'])
            print(f"\n--- 提取的关键帧数量: {len(analysis_result['sampled_frames_pil'])} ---")
        else:
            print(f"分析失败: {analysis_result['error']}")
    else:
        print(f"测试视频文件 {test_video_path} 不存在，请提供一个有效的视频路径。")