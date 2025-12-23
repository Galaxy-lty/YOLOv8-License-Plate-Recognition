import streamlit as st
from ultralytics import YOLO
# from paddleocr import PaddleOCR  <-- 注释掉OCR库
import cv2
import numpy as np
from PIL import Image

# --- 页面配置 ---
st.set_page_config(page_title="车牌检测系统", layout="wide")

# --- 侧边栏 ---
st.sidebar.title("设置")
conf_threshold = st.sidebar.slider("检测置信度 (Confidence)", 0.1, 1.0, 0.25)
st.sidebar.info("模型加载自: best.pt")

# --- 标题 ---
st.title("🚗 深度学习大作业 - 车牌检测系统")
st.markdown("### 基于 YOLOv8 (仅目标检测)")
st.warning("注：由于云端服务器资源限制，在线演示仅展示【车牌定位】功能。")

# --- 加载模型 ---
@st.cache_resource
def load_models():
    # 加载你从Kaggle训练好的YOLO模型
    det_model = YOLO('best.pt') 
    
    # --- 这里是改动的关键点 ---
    # 我们不加载 OCR 模型了，直接返回 None
    return det_model, None 

with st.spinner('正在加载模型...'):
    model, ocr = load_models()

# --- 上传图片 ---
uploaded_file = st.file_uploader("请上传车牌图片", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # 读取图片
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1) 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgb, caption="原始图片", use_column_width=True)

    # 按钮触发
    if st.button('开始检测', type="primary"):
        with st.spinner('正在检测...'):
            # YOLO推理
            results = model(image, conf=conf_threshold)
            
            img_with_box = image.copy()
            
            # 统计检测到的数量
            count = 0

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                count += len(boxes)
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # --- 这里是改动的关键点 ---
                    # 不再调用 ocr.ocr()，直接给一个固定文本
                    txt = "License Plate"
                    
                    # 画框
                    cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    # 画标签背景
                    cv2.rectangle(img_with_box, (x1, y1-30), (x1+150, y1), (0, 255, 0), -1)
                    # 写字
                    cv2.putText(img_with_box, txt, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            with col2:
                st.image(cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB), caption="检测结果", use_column_width=True)
                
            if count > 0:
                st.success(f"检测完成！共发现 {count} 个车牌。")
            else:
                st.warning("未检测到车牌。")
