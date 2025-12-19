# 姿态分析模块使用指南

该模块 (`pose_analysis_module.py`) 提供了一个独立的函数 `analyze_video`，用于对运动视频进行姿态估计和AI智能分析。它可以方便地集成到其他项目中，例如无人机录像系统。

---

## 功能概述

1.  **视频处理**: 接收一个视频文件路径。
2.  **姿态估计**: 使用 `MediaPipe` 库自动抽取关键帧，并计算核心关节（膝、髋）的角度。
3.  **AI分析**: 将关键帧图像和量化数据发送给 `Google Gemini 1.5 Flash` 模型，生成一份专业的、可执行的运动表现分析报告。
4.  **结果返回**: 返回分析报告、量化数据和关键帧图像。

---

## 安装依赖

确保安装了项目所需的依赖包。如果此模块是作为 "AI运动教练" 项目的一部分，可以直接使用项目根目录下的 `requirements.txt` 文件。

```bash
pip install -r requirements.txt
```

核心依赖包括：
*   `opencv-python`
*   `mediapipe`
*   `Pillow`
*   `numpy`
*   `pandas`
*   `langchain`
*   `langchain-google-genai`

---

## API 密钥配置

该模块需要一个 Google AI Studio (Gemini) API 密钥才能调用AI模型。

1.  **获取API密钥**: 访问 [Google AI Studio](https://aistudio.google.com/) 获取你的API密钥。
2.  **配置密钥**: 有两种方式提供API密钥给模块：
    *   **方式一 (推荐)**: 在调用模块之前，设置环境变量 `GEMINI_API_KEY`。
        ```bash
        # Windows (CMD)
        set GEMINI_API_KEY=YOUR_ACTUAL_API_KEY_HERE
        # Windows (PowerShell)
        $env:GEMINI_API_KEY="YOUR_ACTUAL_API_KEY_HERE"
        # Linux/macOS
        export GEMINI_API_KEY=YOUR_ACTUAL_API_KEY_HERE
        ```
    *   **方式二**: 在调用 `analyze_video` 函数时，通过 `api_key` 参数直接传入。

---

## 使用方法

### 1. 导入模块

在你的Python脚本或项目中导入该模块。

```python
from pose_analysis_module import analyze_video
```

### 2. 调用 `analyze_video` 函数

```python
# 视频文件路径 (来自无人机系统)
video_file_path = "path/to/your/drone_recorded_video.mp4"

# 可选: 设置用户训练目标，AI分析时会参考
user_training_goal = "改善跑步时的步频与步幅协调性"

# 可选: 指定要分析的关键帧数量 (默认为6)
num_frames_to_analyze = 8

# 可选: 直接提供API密钥 (如果未通过环境变量设置)
# gemini_api_key = "YOUR_ACTUAL_API_KEY_HERE"

# --- 调用分析函数 ---
result = analyze_video(
    video_path=video_file_path,
    desired_frames=num_frames_to_analyze,
    # api_key=gemini_api_key, # 如果通过参数传递
    user_goal=user_training_goal
)

# --- 处理返回结果 ---
if result['success']:
    print("视频分析成功完成！")
    
    # 1. 获取AI生成的分析报告 (Markdown格式)
    ai_report = result['report']
    print("\n--- AI分析报告 ---")
    print(ai_report)

    # 2. 获取量化数据 (pandas DataFrame)
    quantified_data = result['dataframe']
    print("\n--- 量化数据 ---")
    print(quantified_data)
    # 你可以对 DataFrame 进行进一步处理，例如保存为CSV
    # quantified_data.to_csv("analysis_data.csv", index=False)

    # 3. 获取关键帧图像 (PIL Image对象列表)
    keyframe_images = result['sampled_frames_pil']
    print(f"\n--- 提取的关键帧数量: {len(keyframe_images)} ---")
    # 例如，保存第一张关键帧
    # if keyframe_images:
    #     keyframe_images[0].save("关键帧_1.jpg")

    # --- 示例：将报告保存到文件 ---
    with open("运动分析报告.md", "w", encoding="utf-8") as f:
        f.write(ai_report)
    print("\n分析报告已保存到 '运动分析报告.md'")

else:
    # 分析失败，打印错误信息
    print(f"分析失败: {result['error']}")

```

### 3. 函数参数详解

| 参数 | 类型 | 是否必需 | 默认值 | 描述 |
| --- | --- | --- | --- | --- |
| `video_path` | `str` | 是 | - | 需要分析的视频文件的完整路径。 |
| `desired_frames` | `int` | 否 | `6` | 指定从视频中均匀抽取的关键帧数量。建议范围 5-20。 |
| `api_key` | `str` | 否 | - | Google Gemini API 密钥。如果未提供，函数将尝试从环境变量 `GEMINI_API_KEY` 获取。 |
| `user_goal` | `str` | 否 | `""` | 用户的训练目标。提供此信息可以帮助AI生成更具针对性的分析和建议。 |


### 4. 返回值详解

`analyze_video` 函数返回一个字典 (`dict`)，包含以下键值：

| 键 | 类型 | 描述 |
| --- | --- | --- |
| `success` | `bool` | 指示分析过程是否成功。 |
| `report` | `str` | 如果成功，此键包含由AI模型生成的完整分析报告文本 (Markdown格式)。 |
| `dataframe` | `pd.DataFrame` | 如果成功，此键包含一个 `pandas.DataFrame`，其中包含了每帧的量化角度数据。 |
| `sampled_frames_pil` | `List[Image.Image]` | 如果成功，此键包含一个 `PIL.Image` 对象的列表，代表从视频中提取并标注了姿态的关键帧。 |
| `error` | `str` | 如果 `success` 为 `False`，此键将包含描述失败原因的错误信息字符串。 |

---

通过遵循以上步骤，你可以轻松地将此姿态分析功能集成到你的无人机或其他项目中，为用户提供便捷的运动表现分析服务。
