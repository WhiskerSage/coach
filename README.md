# 🏃‍♂️ AI 运动教练

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/your_username/your_repository/main/app.py)

这是一个结合了多模态大语言模型、计算机视觉和智能代理（Agent）技术的AI运动教练应用。它能够分析用户上传的运动视频，提供专业的姿态评估、量化数据分析、知识问答和长期的运动表现追踪。



---

## 🌟 项目亮点 (Key Features)

- **🤖 多模态视频分析**: 采用 Google `Gemini 1.5 Flash` 模型，能够直接理解视频内容，从多个关键帧中提取视觉信息，生成综合性的运动分析报告。
- **🦾 计算机视觉量化**: 集成 `MediaPipe` 框架，自动识别和追踪人体的关键骨骼点，实时计算膝关节、髋关节等关键角度，为评估提供数据支撑。
- **🧠 LangChain 智能代理 (Agent)**:
    - **RAG 知识库**: 搭载了基于本地知识库的检索增强生成（RAG）系统。当用户提出通用的运动健康问题时，AI可以从`knowledge_base`目录下的文档中检索信息，提供有据可依的答案。
    - **Function Calling 数据工具**: AI 能够智能调用Python函数（工具）来执行特定的数据查询任务，例如查询某个动作的最大、最小角度，或计算左右肢体的对称性差异。
- **📈 长期表现追踪**:
    - **多用户数据存储**: 分析结果会以用户名进行区分，并保存在本地的 `database.json` 文件中。
    - **运动表现仪表盘**: 应用内置了一个独立的仪表盘页面，可以读取和展示指定用户的历史分析数据，方便用户追踪自己的进步轨迹。
- **🎨 可定制的 Streamlit 前端**:
    - 基于 Streamlit 构建，界面简洁、交互友好。
    - 通过 `.streamlit/config.toml` 文件轻松定制应用的主题、颜色和字体。

---

## 🛠️ 技术栈 (Tech Stack)

- **前端**: Streamlit
- **AI 框架**: LangChain
- **大语言模型 (LLM)**: Google Gemini 1.5 Flash
- **嵌入模型 (Embedding)**: Google `embedding-001`
- **计算机视觉**: OpenCV, MediaPipe
- **向量数据库**: FAISS (本地)
- **核心依赖**: `langchain-google-genai`, `streamlit`, `opencv-python`, `mediapipe`, `faiss-cpu`

---

## 🚀 如何运行 (Getting Started)

### 1. 先决条件
- Python 3.8+
- 一个配置好的 Google Cloud 项目，并已启用 "Generative Language API"。

### 2. 克隆仓库
```bash
git clone https://github.com/WhiskerSage/coach
```

### 3. 创建并激活虚拟环境 (推荐)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. 安装依赖
项目所需的所有依赖都已在 `requirements.txt` 中列出。
```bash
pip install -r requirements.txt
```
如果在使用RAG功能时遇到 `unstructured` 相关的错误，请确保安装了Markdown解析的附加依赖：
```bash
pip install "unstructured[md]"
```

### 5. 配置 API 密钥
在项目根目录下创建一个名为 `.streamlit` 的文件夹（如果尚不存在），然后在其中创建一个名为 `secrets.toml` 的文件。填入你的 Gemini API 密钥：

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "你的Google_API_密钥"
```

### 6. 运行应用
```bash
streamlit run app.py
```
应用将在你的本地浏览器中自动打开。

---

## 📂 项目结构 (Project Structure)

```
coach/
│
├── .streamlit/
│   ├── config.toml         # Streamlit 主题配置文件
│   └── secrets.toml        # 存放 API 密钥
│
├── app.py                  # 主应用文件：视频上传、分析、聊天交互
│
├── pages/
│   └── 1_📈_运动表现仪表盘.py # 仪表盘页面，用于展示历史数据
│
├── knowledge_base/
│   └── 运动常识.md         # RAG 知识库的源文件
│
├── database.json           # 存储用户分析结果的数据库文件
│
├── requirements.txt        # Python 依赖列表
│
└── README.md               # 项目说明文件
```
---


