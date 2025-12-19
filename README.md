# 🏃‍♂️ AI 运动教练 V1

**你的随身运动教练** - 这是一个结合了计算机视觉与大语言模型的AI驱动的运动姿态分析应用。用户可以上传自己的运动视频，应用会通过姿态估计算法量化关键关节角度，并利用Google Gemini多模态模型生成一份专业的、可执行的运动表现分析报告。

## 🌟 核心功能

- **视频姿态分析:** 上传运动视频（如跑步、深蹲等），应用会自动提取关键帧。
- **量化数据提取:** 利用`MediaPipe`识别人体关键点，计算出核心关节（如膝、髋）的角度变化，并将数据可视化。
- **多模态AI报告:** 将视频帧和量化数据一同提交给`Google Gemini`模型，生成包含**综合得分、多模态诊断、核心改进建议**的专业报告。
- **AI Agent智能问答:**
    - **数据查询:** 在报告生成后，可以继续向AI提问关于本次运动的具体数据（如"我的膝盖最大弯曲了多少度？"），Agent会自动调用工具查询并回答。
    - **知识库检索 (RAG):** 可以提问通用的运动知识（如"跑步后如何拉伸？"），AI会从本地知识库 (`knowledge_base`目录) 中检索相关信息，提供更可靠的答案。
- **历史数据仪表盘:**
    - **自动存档:** 每次分析结果都会自动为用户存档。
    - **表现追踪:** 在"运动表现仪表盘"页面，可以查看所有历史分析报告及对应的详细数据图表，追踪长期进步。
- **灵活的分析配置:** 支持用户自定义希望分析的视频帧数。

## 🛠️ 技术栈

- **前端:** `Streamlit`
- **计算机视觉:** `OpenCV`, `MediaPipe`
- **AI 框架:**
    - **核心模型:** `Google Gemini 2.5 Flash Lite`
    - **Embedding 模型:** `Google text-embedding-004`
    - **编排框架:** `LangChain` (Agent、Tool Calling、RAG)
- **数据处理与可视化:** `Pandas`, `NumPy`, `Plotly`
- **向量存储:** `FAISS`
- **数据存储:** 本地 `JSON` 文件

## 🚀 如何运行

### Windows 快速启动（推荐）

项目提供了便捷的批处理脚本：

1. **安装依赖**: 双击 `安装依赖.bat`
2. **启动应用**: 双击 `启动应用.bat`

详细说明请查看 `快速启动指南.md`

### 手动启动

1.  **克隆/下载项目代码**

2.  **创建并激活Python环境** (推荐使用 Conda)
    ```bash
    conda create -n ai_coach python=3.9
    conda activate ai_coach
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

4.  **配置API密钥**
    - 在项目根目录下创建 `.streamlit` 文件夹。
    - 在该文件夹内创建一个名为 `secrets.toml` 的文件。
    - 在 `secrets.toml` 文件中添加以下内容，并填入你的Google AI Studio API密钥:
      ```toml
      GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
      ```

5.  **设置网络代理 (如果需要)**
    如果直接访问Google API存在网络问题，请先打开你的代理/VPN客户端，然后在**运行应用的终端**中设置代理环境变量：
    ```bash
    # Windows
    set HTTPS_PROXY=http://127.0.0.1:7897
    ```

6.  **启动应用**
    ```bash
    streamlit run app.py
    ```

## 📂 项目结构

```
coach/
│
├── .streamlit/
│   └── secrets.toml        # 存放API密钥等敏感信息
│
├── knowledge_base/
│   └── 运动常识.md         # RAG知识库的源文件
│
├── pages/
│   └── 1_📈_运动表现仪表盘.py # Streamlit多页面，用于展示历史数据
│
├── app.py                  # 主应用文件
├── database.json           # JSON数据库，用于存档分析结果
├── requirements.txt        # 项目依赖
├── README.md               # 项目说明文件
├── 安装依赖.bat             # Windows快速安装脚本
├── 启动应用.bat             # Windows快速启动脚本
└── 快速启动指南.md          # 快速启动说明
```

---

**版本:** V1
**项目定位:** 你的随身运动教练