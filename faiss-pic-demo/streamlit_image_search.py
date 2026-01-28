import os
# 设置环境变量以避免OpenMP错误
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
import faiss
import json
from pathlib import Path
import os

# 设置页面标题和配置
st.set_page_config(
    page_title="图像相似度检索系统",
    page_icon="🔍",
    layout="wide"
)

# 页面标题
st.title("🔍 基于深度学习的图像相似度检索系统")
st.markdown("---")

# 初始化设备
@st.cache_resource
def load_model():
    """加载预训练模型并初始化特征提取器"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载预训练ResNet模型并修改为特征提取器
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights).to(device)
    # 去除最后两层（全局平均池化层后直接输出特征，无需分类层）
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()  # 进入评估模式，禁用Dropout等
    
    # 图像预处理（与预训练模型要求一致）
    transform = transforms.Compose([
        transforms.Resize(256),  # 先缩放到256（短边），保持比例
        transforms.CenterCrop(224),  # 中心裁剪到224×224（ResNet标准输入）
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet均值
                             std=[0.229, 0.224, 0.225])   # ImageNet标准差
    ])
    
    return feature_extractor, transform, device

# 加载模型和相关组件
feature_extractor, transform, device = load_model()

def extract_image_feature(image):
    """提取单张图像的特征向量"""
    try:
        # 将上传的PIL图像转换为RGB格式
        image = image.convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)  # 增加批次维度
        
        # 无梯度计算加速
        with torch.no_grad():
            feature = feature_extractor(input_tensor)
        
        # 特征向量处理（展平为1D向量并归一化）
        feature_vector = feature.squeeze().cpu().numpy()
        # L2归一化（增加边界条件，避免除以0）
        norm = np.linalg.norm(feature_vector)
        feature_vector = feature_vector / norm if norm > 1e-6 else feature_vector
        return feature_vector.astype(np.float32)
    except Exception as e:
        st.error(f"特征提取失败：{e}")
        return None

# 加载图像检索数据库
@st.cache_resource
def load_image_database():
    """加载图像索引和元数据"""
    db_dir = Path("./image_search_db")
    index_path = db_dir / "image_index.index"
    metadata_path = db_dir / "image_metadata.json"
    
    if not index_path.exists() or not metadata_path.exists():
        st.error("图像检索数据库不存在，请先构建数据库！")
        return None, []
    
    # 加载索引与元数据
    loaded_index = faiss.read_index(str(index_path))
    with open(metadata_path, "r", encoding="utf-8") as f:
        loaded_img_metadata = json.load(f)
    
    return loaded_index, loaded_img_metadata

# 侧边栏
with st.sidebar:
    st.header("⚙️ 参数设置")
    top_k = st.slider("返回相似图像数量", min_value=1, max_value=10, value=3)
    st.markdown("---")
    st.info("💡 使用说明：\n1. 在下方上传待检索的图像\n2. 系统会自动提取图像特征\n3. 与数据库中的图像进行相似度比较\n4. 显示最相似的图像结果")

# 主界面 - 图像上传区域
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 上传查询图像")
    uploaded_file = st.file_uploader(
        "请选择一张图像进行相似图像检索", 
        type=["jpg", "jpeg", "png"],
        help="支持JPG、JPEG、PNG格式的图像文件"
    )

    if uploaded_file is not None:
        # 显示上传的图像
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的查询图像")

with col2:
    st.subheader("📊 检索结果")
    
    if uploaded_file is not None:
        # 加载数据库
        index, img_metadata = load_image_database()
        
        if index is not None:
            # 提取上传图像的特征
            with st.spinner('正在提取图像特征...'):
                query_feature = extract_image_feature(image)
            
            if query_feature is not None:
                with st.spinner('正在检索相似图像...'):
                    # 转换为合适的形状和类型
                    query_feature = query_feature.reshape(1, -1).astype(np.float32)
                    
                    # 执行检索
                    distances, indices = index.search(query_feature, top_k)
                
                # 显示检索结果
                st.success(f"找到 {top_k} 个最相似的图像")
                
                # 展示结果
                for i in range(top_k):
                    idx = indices[0][i]
                    distance = distances[0][i]
                    
                    # 防止索引越界
                    if idx < 0 or idx >= len(img_metadata):
                        st.warning(f"排名 {i+1}: 无匹配结果")
                        continue
                    
                    # 获取匹配图像信息
                    matched_img_info = img_metadata[idx]
                    
                    # 创建结果卡片
                    with st.container():
                        col_result_img, col_result_info = st.columns([1, 2])
                        
                        with col_result_img:
                            # 尝试显示匹配的图像
                            try:
                                matched_img_path = matched_img_info['image_path']
                                if os.path.exists(matched_img_path):
                                    matched_img = Image.open(matched_img_path)
                                    st.image(matched_img, caption=f"匹配图像 {i+1}")
                                else:
                                    st.warning(f"图像文件不存在: {matched_img_path}")
                            except Exception as e:
                                st.warning(f"无法加载匹配图像: {e}")
                        
                        with col_result_info:
                            st.markdown(f"**排名 {i+1}**")
                            st.markdown(f"L2距离: **{distance:.4f}**")
                            st.markdown(f"产品ID: `{matched_img_info['product_id']}`")
                            st.markdown(f"类别: `{matched_img_info['category']}`")
                            st.text(f"路径: {matched_img_info['image_path']}")
                        
                        st.markdown("---")
            else:
                st.error("图像特征提取失败，请尝试其他图像")
        else:
            st.error("无法加载图像检索数据库")
    else:
        st.info("请在左侧上传一张图像以开始检索")

# 底部信息
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>基于 ResNet50 + FAISS 的图像相似度检索系统</p>
    <p>使用深度学习提取图像特征，通过L2距离计算图像相似度</p>
</div>
""", unsafe_allow_html=True)