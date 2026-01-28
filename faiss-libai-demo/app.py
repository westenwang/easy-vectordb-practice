from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import faiss
import json
import numpy as np
import re  # 用于句子切分的正则表达式
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os
import uvicorn
from fastapi.middleware.cors import CORSMiddleware  # 添加CORS支持

# ===================== 1. 配置路径与参数 =====================
app = FastAPI(title="文本语义检索API", version="1.0")

# 添加CORS中间件以允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 获取当前脚本所在目录作为根目录
root_dir = Path(__file__).parent

# 路径配置
txt_corpus_path = root_dir / "text_corpus.txt"  # 待处理的TXT文本文件路径
data_dir = root_dir / "text_search_db"  # 索引和元数据保存目录
index_path = data_dir / "faiss_index.index"  # FAISS索引路径
metadata_path = data_dir / "metadata.json"  # 元数据路径

# 模型与检索配置（换回指定的gte中文模型）
model_path = root_dir / "model" / "iic" / "nlp_gte_sentence-embedding_chinese-base"  # 使用绝对路径
prefix = ""  # gte中文模型无需前缀，可根据需求自定义
# 中文句子分隔符（正则表达式：匹配。！？；中的任意一个，后面可能跟换行/空格）
sentence_sep_pattern = re.compile(r'[。！？；]+')


# ===================== 2. 定义数据模型（请求/响应） =====================
class SearchRequest(BaseModel):
    query: str
    top_k: int = 3  # 默认返回3条结果

class SearchResult(BaseModel):
    rank: int
    similarity: float
    text: str
    metadata: Dict[str, str]

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

# ===================== 3. 工具函数 =====================
def read_and_split_sentences(txt_path: Path) -> List[str]:
    """
    读取TXT文件并按中文句子切分，返回非空句子列表
    """
    # 检查文件是否存在
    if not txt_path.exists():
        # 生成示例文本（首次运行时自动创建）
        sample_text = """ 
        "登金陵凤凰台：凤凰台上凤凰游，凤去台空江自流。吴宫花草埋幽径，晋代衣冠成古丘。三山半落青天外，二水中分白鹭洲。总为浮云能蔽日，长安不见使人愁。",
        "赠郭将军：将军少年出武威，入掌银台护紫微。平明拂剑朝天去，薄暮垂鞭醉酒归。爱子临风吹玉笛，美人向月舞罗衣。畴昔雄豪如梦里，相逢且欲醉春晖。",
        "题雍丘崔明府丹灶：美人为尔别开池，莲花倾城出自姿。素影空中飘素月，清光水上照清辉。神仙多被宏慈误，宰执常遭白简非。借问欲栖珠树鹤，何年却向帝城飞？",
        "鹦鹉洲：鹦鹉来过吴江水，江上洲传鹦鹉名。鹦鹉西飞陇山去，芳洲之树何青青。烟开兰叶香风暖，岸夹桃花锦浪生。迁客此时徒极目，长洲孤月向谁明。",
        "别中都明府兄：吾兄诗酒继陶君，试宰中都天下闻。东楼喜奉连枝会，南陌愁为落叶分。城隅渌水明秋日，海上青山隔暮云。取醉不辞留夜月，雁行中断惜离群。",
        "将进酒：君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。",
        "宣州谢朓楼饯别校书叔云：弃我去者昨日之日不可留，乱我心者今日之日多烦忧。长风万里送秋雁，对此可以酣高楼。蓬莱文章建安骨，中间小谢又清发。俱怀逸兴壮思飞，欲上青天览明月。",
        "梦游天姥吟留别（节选）：海客谈瀛洲，烟涛微茫信难求；越人语天姥，云霞明灭或可睹。天姥连天向天横，势拔五岳掩赤城。天台四万八千丈，对此欲倒东南倾。",
        "早发白帝城：朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
        "黄鹤楼送孟浩然之广陵：故人西辞黄鹤楼，烟花三月下扬州。孤帆远影碧空尽，唯见长江天际流。"
        """
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(sample_text)
        print(f"未找到{txt_path}，已自动生成示例文本！")
        text_content = sample_text
    else:
        # 读取TXT文件内容
        with open(txt_path, "r", encoding="utf-8") as f:
            text_content = f.read()
    
    # 按句子分隔符切分文本
    sentences = sentence_sep_pattern.split(text_content)
    # 过滤空句子和仅含空白字符的句子
    sentences = [sent.strip() for sent in sentences if sent.strip()]
    if not sentences:
        raise ValueError("TXT文件中无有效句子可切分！")
    return sentences

def build_faiss_index():
    """
    构建FAISS索引：读取TXT→切分句子→生成嵌入→创建并保存索引/元数据
    """
    # 1. 读取并切分句子
    sentences = read_and_split_sentences(txt_corpus_path)
    
    # 2. 加载指定的gte中文模型（本地路径，若不存在会自动从HuggingFace下载）
    print(f"尝试加载模型: {model_path}")
    if not model_path.exists():
        raise RuntimeError(f"资源初始化失败：模型路径 {model_path} 不存在")
    
    model = SentenceTransformer(str(model_path))  # 确保传入字符串而非Path对象
    
    # 3. 生成句子嵌入（归一化，适配内积索引）
    print("正在生成句子嵌入向量...")
    embeddings = model.encode(
        [prefix + sent for sent in sentences],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)  # FAISS要求float32类型
    # 4. 构建元数据（包含句子ID、来源、文本内容）
    metadata = []
    for idx, sent in enumerate(sentences):
        metadata.append({
            "text": sent,
            "metadata": {
                "sentence_id": str(idx + 1),
                "source": str(txt_corpus_path),
                "total_sentences": str(len(sentences))
            }
        })
    # 5. 创建FAISS索引（内积索引，归一化后等价于余弦相似度）
    d = embeddings.shape[1]  # 向量维度
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    # 6. 保存索引和元数据
    data_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"FAISS索引构建完成，共添加{index.ntotal}个句子向量")
    return index, model, metadata

# ===================== 4. 初始化资源（启动时仅加载一次） =====================
print("正在初始化资源...")
try:
    # 尝试加载已存在的索引和元数据
    if index_path.exists() and metadata_path.exists():
        model = SentenceTransformer(str(model_path))  # 加载指定模型，确保传入字符串
        index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"成功加载已有索引，包含{index.ntotal}个句子向量")
    else:
        # 索引不存在时，重新构建
        index, model, metadata = build_faiss_index()
except Exception as e:
    raise RuntimeError(f"资源初始化失败：{str(e)}")

# ===================== 5. 接口定义 =====================
@app.post("/text-search", response_model=SearchResponse, summary="文本语义检索")
def text_search(request: SearchRequest):
    try:
        # 生成查询向量（与句子嵌入使用相同的前缀和归一化）
        query_embedding = model.encode(
            prefix + request.query,
            normalize_embeddings=True
        ).astype(np.float32).reshape(1, -1)
        
        # 执行检索（IndexFlatIP：内积值越大，相似度越高）
        distances, indices = index.search(query_embedding, request.top_k)
        results = []
        for i in range(request.top_k):
            idx = indices[0][i]
            if idx == -1:  # 无匹配结果时的处理
                continue
            # 内积索引的相似度直接使用距离值（归一化后为余弦相似度，范围[-1,1]）
            similarity = round(distances[0][i], 4)
            results.append(SearchResult(
                rank=i+1,
                similarity=similarity,
                text=metadata[idx]["text"],
                metadata=metadata[idx]["metadata"]
            ))
        return SearchResponse(query=request.query, results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败：{str(e)}")

@app.get("/health", summary="服务健康检查")
def health_check():
    return {
        "status": "healthy",
        "index_vector_count": index.ntotal,
        "corpus_source": str(txt_corpus_path),
        "total_sentences": len(metadata),
        "model_path": model_path
    }

# ===================== 6. 初始化服务 =====================
print("正在初始化资源...")
try:
    # 检查索引和元数据文件是否存在，如果不存在则构建
    if not index_path.exists() or not metadata_path.exists():
        print("检测到首次运行，正在构建FAISS索引...")
        index, model, metadata = build_faiss_index()
        print("FAISS索引构建完成！")
    else:
        # 加载已有的索引和元数据
        print("正在加载现有FAISS索引...")
        dimension = 768  # GTE模型输出维度
        index = faiss.read_index(str(index_path))  # 确保传入字符串
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # 重新加载模型
        if not model_path.exists():
            raise RuntimeError(f"资源初始化失败：模型路径 {model_path} 不存在")
        model = SentenceTransformer(str(model_path))  # 确保传入字符串
        
        print("现有FAISS索引加载完成！")
    
    print("服务初始化成功！")
except Exception as e:
    raise RuntimeError(f"资源初始化失败：{str(e)}")

if __name__ == "__main__":
    # 配置启动参数（可根据需求调整）
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # 生产环境应关闭热重载
    )