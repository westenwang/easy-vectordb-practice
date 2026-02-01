## Faiss核心功能进阶



### 向量归一化和相似度匹配

#### 核心原理：COSINE相似度与归一化关系

- COSINE相似度用于衡量两个向量在方向上的一致性
- 归一化后的向量之间的COSINE相似度直接等于他们的内积
- **结论**
  - 计算COSINE相似度时，必须先对向量进行L2归一化处理（即将向量的长度规范为1）
  - 使用COSINE相似度进行检索时，可以选择**内积**或**L2距离**作为度量标准，前提是已对向量进行归一化。

#### 核心API： faiss.normalize_L2

```python
### 对矩阵的每行（向量）做L2归一化
faiss.normalize_L2(x)

# 示例
x = np.array([[1,2,3], [4,5,6]], dtype=np.float32)
faiss.noramlize_L2(x)
```



#### 实战：归一化前后检索结果对比

![fa75b57b-0526-4275-b918-a9e08776edc4](/Users/wangshiquan/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_axczh66ejzyt21_e21e/temp/InputTemp/fa75b57b-0526-4275-b918-a9e08776edc4.png)



### 索引的保存与加载：适配工程化部署

索引训练耗时长（百万级别需分钟级），需将训练好的索引“持久化”保存，在线服务时直接加载使用，避免重复训练。

#### 核心API与保存格式

- `faiss.write_index(index, 'index.path')` ,  将索引序列化保存至本地文件。注意”文件格式为二进制，不可编辑“
- `faiss.read_index('index.path')`,  从本地文件加载索引，返回索引对象。注意”加载后可直接调用search方法，无需重新训练“。

**注意**

- 索引需完成训练和数据添加后保存
- 保存路径需有写入权限，建议用`.index`作为后缀标识



#### 实战：离线训练索引，在线加载检索

案例代码使用的是本地运行离线数据集方式，如果要在ModelScope提供的CPU环境上执行，下载并上上传数据集方式引用。



#### 在魔搭社区创建数据集（使用此数据集有问题，待处理）

- 访问 [魔搭数据集中心](https://modelscope.cn/datasets)
- 点击 「创建数据集」
  填写信息：
  数据集名称：例如 sift1m-fvecs
  命名空间：你的用户名（如 yourname）
  可见性：私有（默认）或公开（需审核）
  数据格式：选择「其他」或「向量检索」
  描述：说明包含 .fvecs/.ivecs，用于 ANN benchmark
- 点击 「上传文件」，将上述 3 个文件上传
- 点击 「提交」→ 「发布版本」（如 v1.0）



#### 替换其他数据集，修改代码逻辑尝试

- 在魔搭里找到一些开源数据集，参考使用方式 + 数据集文件，进行修改处理。

```python
from modelscope.msdatasets import MsDataset
import numpy as np
import faiss
import os

# 1. 加载数据集
dataset =  MsDataset.load('MiniMax/OctoCodingBench', subset_name='default', split='train')


# 2. 提取向量数据
# 假设数据集中的向量数据存储在 "vector" 字段中，维度存储在 "dimension" 字段中
# 你可以根据实际数据结构调整字段名
base_vectors = []
query_vectors = []
groundtruth = []

for entry in dataset:
    if entry["category"] == "Skill" and entry["scaffold"]["name"] == "claudecode":
        # 提取数据库向量
        if "vector" in entry and "dimension" in entry:
            d = entry["dimension"]
            vector = np.array(entry["vector"], dtype=np.float32)
            base_vectors.append(vector)
        
        # 提取查询向量和真实近邻
        if "query_vector" in entry and "groundtruth" in entry:
            query_vector = np.array(entry["query_vector"], dtype=np.float32)
            query_vectors.append(query_vector)
            groundtruth.append(entry["groundtruth"])

# 转换为 NumPy 数组
xb = np.array(base_vectors)
xq = np.array(query_vectors)
gt = np.array(groundtruth, dtype=np.int32)

print("向量维度d:", xb.shape[1])
print("原始数据形状:", xb.shape, xq.shape, gt.shape)

# 3. 创建并训练IVF-PQ索引
d = xb.shape[1]  # 获取向量维度
nlist = 1024
m = 16
nbits = 8
nprobe = 50

# 索引格式：IVF{nlist},PQ{m}，距离度量用L2（欧式距离）
index = faiss.index_factory(d, f"IVF{nlist},PQ{m}", faiss.METRIC_L2)

# IVF-PQ必须先训练（聚类中心+PQ量化器），训练数据需足够多
print("开始训练IVF-PQ索引...")
index.train(xb)
print("训练完成，开始添加数据库向量...")

# 添加向量到索引（IVF会将向量分配到对应的聚类中心）
index.add(xb)
print(f"向量添加完成：共添加 {index.ntotal} 个向量")

# 设置查询参数nprobe（影响查询性能）
index.nprobe = nprobe
print(f"查询参数nprobe设置为：{nprobe}")

# 4. 保存索引
index_path = "IVF_PQ.index"
faiss.write_index(index, index_path)

# 计算索引文件大小
index_size = os.path.getsize(index_path) / 1024 / 1024  # 转为MB
print(f"\nIVF-PQ索引保存成功！")
print(f"索引路径：{index_path}")
print(f"索引大小：{np.round(index_size, 2)} MB")
```

### 参考文档

[Faiss核心功能进阶](https://github.com/datawhalechina/easy-vecdb/blob/main/docs/Faiss/chapter3/FAISS%E6%A0%B8%E5%BF%83%E5%8A%9F%E8%83%BD%E8%BF%9B%E9%98%B6.md)

