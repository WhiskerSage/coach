"""
测试 pose_analysis_module.py 模块的脚本。

此脚本用于验证 analyze_video 函数是否能正常工作。
它会分析项目根目录下的 '跑步.mp4' 文件（如果存在）。
请确保已设置环境变量 GEMINI_API_KEY 或在此脚本中填写你的API密钥。
"""

import os
from pose_analysis_module import analyze_video

# --- 配置 ---
# 1. 指定要分析的视频文件路径
#    这里我们测试项目自带的 '跑步.mp4' 文件
test_video_file = "跑步.mp4"

# 2. (可选) 设置你的 Google Gemini API 密钥
#    推荐通过环境变量设置，避免将密钥硬编码在代码中。
#    在运行此脚本前，请执行 (Windows):
#    set GEMINI_API_KEY=your_actual_api_key_here
#    或者 (Linux/macOS):
#    export GEMINI_API_KEY=your_actual_api_key_here
#
# 如果你选择在这里硬编码，请务必在使用后删除或保密：
os.environ['GEMINI_API_KEY'] = 'AIzaSyAj1nMpq05G-NYdR1YqwKB3S24jzS_F6N8'

# 3. (可选) 设置用户训练目标
user_goal = "改善跑步姿态，特别是步频和着地技术"

# 4. (可选) 设置分析的关键帧数量
num_frames = 6

# --- 执行测试 ---
if __name__ == "__main__":
    print(f"开始测试姿态分析模块...")
    print(f"目标视频文件: {test_video_file}")

    # 检查视频文件是否存在
    if not os.path.exists(test_video_file):
        print(f"错误: 找不到视频文件 '{test_video_file}'。请确认文件路径是否正确。")
        exit(1)

    # 检查API密钥
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("警告: 环境变量 'GEMINI_API_KEY' 未设置。")
        print("请先设置API密钥，例如 (Windows): set GEMINI_API_KEY=your_actual_api_key_here")
        exit(1)
    
    print("API密钥已找到。")

    try:
        # 调用分析函数
        print(f"正在分析视频 '{test_video_file}'...")
        result = analyze_video(
            video_path=test_video_file,
            desired_frames=num_frames,
            user_goal=user_goal
            # api_key=api_key # 如果不使用环境变量，取消注释此项并传入密钥
        )

        # 检查并打印结果
        if result['success']:
            print("\n--- 分析成功! ---")
            
            print("\n--- AI分析报告 ---")
            print(result['report'])
            
            print("\n--- 量化数据 (前5行) ---")
            print(result['dataframe'].head())
            
            print(f"\n--- 提取的关键帧数量: {len(result['sampled_frames_pil'])} ---")
            
            # (可选) 保存报告和数据
            with open("测试报告_输出.md", "w", encoding="utf-8") as f:
                f.write(result['report'])
            result['dataframe'].to_csv("测试数据_输出.csv", index=False)
            print("\n报告和数据已分别保存到 '测试报告_输出.md' 和 '测试数据_输出.csv'")
            
        else:
            print(f"\n--- 分析失败 ---")
            print(f"错误信息: {result['error']}")

    except Exception as e:
        print(f"\n--- 测试过程中发生未预期的错误 ---")
        print(f"异常信息: {e}")
        import traceback
        traceback.print_exc()