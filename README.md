# easy-vectordb-practice

`easy-vectordb-practice` 是一个基于 FAISS (Facebook AI Similarity Search) 的向量数据库实践项目，包含文本搜索和图像搜索两个演示模块。该项目旨在帮助开发者理解和实践向量数据库的核心概念和技术实现。

## 项目概述

本项目提供了两个主要的向量搜索演示：

1. **文本语义搜索** (`faiss-libai-demo`) - 基于 Sentence Transformers 和 LibAI 模型的文本相似度搜索
2. **图像相似度搜索** (`faiss-pic-demo`) - 基于 ResNet50 模型的图像特征提取与相似度匹配

## 目录结构

```
easy-vectordb-practice/
├── faiss-libai-demo/      # 文本语义搜索演示
│   ├── ReadMe.md          # 详细使用说明
│   ├── app.py             # FastAPI 主应用
│   ├── embedding-notebook.ipynb  # 嵌入向量生成示例
│   ├── search_frontend.html      # HTML 前端界面
│   └── test_client.py     # 客户端测试脚本
└── faiss-pic-demo/        # 图像相似度搜索演示
    ├── ReadMe.md          # 详细使用说明
    ├── quickstart.ipynb   # 快速入门示例
    ├── streamlit_image_search.py  # Streamlit 界面
    └── requirements.txt   # 项目依赖
```

## 功能特点

### 文本搜索模块
- 使用 Sentence Transformers 或 LibAI 模型生成文本嵌入向量
- 构建基于 FAISS 的高效文本相似度检索系统
- 提供 FastAPI 接口和 HTML 前端界面
- 支持语义搜索而非关键词匹配

### 图像搜索模块
- 使用预训练的 ResNet50 模型提取图像特征向量
- 构建高效的图像相似度检索系统
- 提供 Streamlit 前端界面进行可视化展示
- 支持多种索引策略以适应不同规模的数据集

## 安装与使用

### 环境准备

推荐使用 Conda 创建虚拟环境：

```bash
conda create -n faiss-env python=3.10
conda activate faiss-env
```

### 文本搜索模块

1. 安装依赖：
   ```bash
   cd faiss-libai-demo
   pip install -r requirements.txt
   ```

2. 启动服务：
   ```bash
   python app.py
   ```
   
   或者使用 Uvicorn：
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

3. 测试服务：
   ```bash
   python test_client.py
   ```

### 图像搜索模块

1. 安装依赖：
   ```bash
   cd faiss-pic-demo
   pip install -r requirements.txt
   ```

2. 启动 Streamlit 界面：
   ```bash
   streamlit run streamlit_image_search.py
   ```

3. 或运行快速入门示例：
   ```bash
   jupyter notebook quickstart.ipynb
   ```

## 技术栈

- **FAISS**: Facebook AI Similarity Search，高效的相似性搜索和聚类库
- **FastAPI**: 现代、快速（高性能）的 Python Web 框架
- **Streamlit**: 用于创建数据应用的开源框架
- **Sentence Transformers**: 用于生成句子和文本嵌入的库
- **PyTorch/Torchvision**: 机器学习框架
- **ResNet50**: 预训练卷积神经网络模型
- **Jupyter Notebook**: 交互式开发环境

## 适用场景

- 文本语义搜索与匹配
- 图像相似度检索
- 推荐系统开发
- 内容去重与查重
- 知识库问答系统

## 学习目标

通过本项目，您可以学习到：

- 如何使用 FAISS 构建向量索引
- 文本和图像的特征提取方法
- 向量数据库的基本原理与实践
- 相似度搜索的实现方式
- 前后端接口的设计与实现

## 参考

感谢Datawhale开源的向量数据库项目，本项目基于Datawhale开源项目[easy-vecdb](https://github.com/datawhalechina/easy-vecdb)进行的实践和扩展。

## 其他

如果您有任何问题或建议，欢迎在 GitHub 仓库中提交 issue。引用请注明来源。

[GitHub 仓库](https://github.com/westenwang/easy-vectordb-practice/)

祝您使用愉快！

