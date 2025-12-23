import streamlit as st
from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# --- 页面配置 ---
st.set_page_config(page_title="车牌检测与识别系统", layout="wide")

# --- 侧边栏 ---
st.sidebar.title("设置")
conf_threshold = st.sidebar.slider("检测置信度 (Confidence)", 0.1, 1.0, 0.25)
st.sidebar.info("模型加载自:best.pt")

# --- 标题 ---
st.title("🚗 深度学习大作业 - 车牌检测与识别系统")
st.markdown("### 基于 YOLOv8 (目标检测) + PaddleOCR (文字识别)")


# --- 加载模型 (加缓存装饰器，防止每次刷新都重新加载) ---
@st.cache_resource
def load_models():
    # 加载你从Kaggle训练好的YOLO模型
    det_model = YOLO('best.pt')
    # 加载OCR模型 (自动下载轻量级模型)
    ocr_model = PaddleOCR(use_angle_cls=True, lang="ch")
    return det_model, ocr_model


with st.spinner('正在加载模型，请稍候...'):
    model, ocr = load_models()

# --- 上传图片 ---
uploaded_file = st.file_uploader("请上传一张包含车牌的图片 (支持 JPG, PNG)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # 1. 读取图片
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)  # BGR格式 (OpenCV标准)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB格式 (用于显示)

    # 分列显示
    col1, col2 = st.columns(2)

    with col1:
        st.image(image_rgb, caption="原始图片", use_column_width=True)

    # 2. 点击检测按钮
    if st.button('开始检测与识别', type="primary"):
        with st.spinner('正在进行深度学习推理...'):
            # YOLO推理
            results = model(image, conf=conf_threshold)

            # 用于在原图上画框
            img_with_box = image.copy()

            recognized_text = []

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()

                if len(boxes) == 0:
                    st.warning("未检测到车牌，请调整置信度或更换图片。")

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)

                    # 裁剪车牌区域
                    plate_crop = image[y1:y2, x1:x2]

                    # OCR 识别
                    # 这里的 cls=True 表示启用方向分类，防止车牌歪了读不准
                    ocr_res = ocr.ocr(plate_crop, cls=True)

                    # 处理OCR结果
                    txt = "未识别"
                    score = 0.0
                    if ocr_res and ocr_res[0]:
                        txt = ocr_res[0][0][1][0]
                        score = ocr_res[0][0][1][1]

                    recognized_text.append(f"📍 内容: **{txt}** (可信度: {score:.2f})")

                    # 在图上画框和文字
                    cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    # 为了防止文字乱码，图片上只画框，文字在右侧显示

            # 显示结果
            with col2:
                st.image(cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB), caption="检测结果", use_column_width=True)

            # 显示识别到的文本列表
            if recognized_text:
                st.success("检测完成！")
                for info in recognized_text:

                    st.markdown(info)

