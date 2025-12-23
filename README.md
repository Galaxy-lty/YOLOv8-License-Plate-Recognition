# 🚗 基于深度学习的车牌检测与识别系统 (License Plate Recognition System)

## 📖 项目介绍
本项目是深度学习课程的期末大作业。项目实现了一个端到端的车牌识别系统，能够从复杂的自然场景中精准定位车牌并识别字符。
系统采用 **YOLOv8** 进行目标检测，结合 **PaddleOCR** 进行光学字符识别，并使用 **Streamlit** 开发了友好的 Web 交互界面。

## 🛠️ 技术栈 (Tech Stack)
- **核心语言**: Python 3.9
- **目标检测 (Detection)**: YOLOv8 (Ultralytics) - 用于定位车牌位置
- **文字识别 (Recognition)**: PaddleOCR (CRNN) - 用于识别车牌号码
- **应用框架 (Web UI)**: Streamlit - 用于构建可视化演示系统
- **训练平台**: Kaggle GPU (T4 x2)

## 🚀 功能特性
1. **高精度检测**: 经微调的 YOLOv8 模型，能够适应倾斜、遮挡、暗光等复杂环境。
2. **中文支持**: 集成 PaddleOCR，完美支持中国车牌（蓝牌、绿牌）的汉字识别。
3. **可视化界面**: 提供一键上传图片、实时推理、结果可视化的 Web 界面。
4. **端到端流程**: 包含从数据预处理、模型训练到系统部署的完整代码。

## 📊 效果展示
*(请在此处插入你的软件运行截图，例如 screenshot.jpg)*

<img width="2400" height="1200" alt="fig1_loss" src="https://github.com/user-attachments/assets/ca6fbdf9-3b99-424b-8910-15d58f9cfcde" />
![fig4_val](https://github.com/user-attachments/assets/b1b15bb9-60fa-4268-8f28-204f376027ea)
<img width="2250" height="1500" alt="fig3_pr" <img width="1580" height="1401" alt="fig5_system" src="https://github.com/user-attachments/assets/6bc4f75f-10bb-4199-99ce-a20decc3ded8" />
src="https://github.com/user-attachments/assets/b691c8b2-e6bc-4ed4-9608-d90d112f5a0d" />
<img width="3000" height="2250" alt="fig2_matrix" src="https://github.com/user-attachments/assets/d51c54a0-b219-4c3d-a397-de55ddc3642a" />

## 📂 项目结构
```text
├── weights/
│   └── best.pt          # YOLOv8 训练好的权重文件
├── app.py               # Streamlit 系统主程序
├── requirements.txt     # 项目依赖库
├── README.md            # 项目说明文档
└── test_images/         # 测试图片文件夹
