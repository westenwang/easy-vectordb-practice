import requests
import time

# 配置项
API_URL = "http://localhost:8000/text-search"
HEALTH_CHECK_URL = "http://localhost:8000/health"
# 可修改查询文本测试不同场景
QUERY_TEXT = "两岸猿声啼不住？"
TOP_K = 3

def test_search():
    # 构造请求体
    data = {
        "query": QUERY_TEXT,
        "top_k": TOP_K
    }
    try:
        response = requests.post(API_URL, json=data, timeout=10)
        response.raise_for_status()  # 主动触发HTTP状态码异常（如500/400）
        result = response.json()
        
        # 格式化打印结果
        print("===== 检索结果 =====")
        print(f"查询文本：{result['query']}\n")
        for item in result['results']:
            print(f"排名：{item['rank']}")
            print(f"相似度：{item['similarity']}")
            print(f"文本：{item['text']}")
            print(f"元数据：{item['metadata']}\n")
    except requests.exceptions.HTTPError as e:
        print(f"请求失败，状态码：{response.status_code}")
        print(f"错误详情：{response.text}")
        # 针对500错误给出提示
        if response.status_code == 500 and "'metadata'" in response.text:
            print("\n【解决方案】：删除text_search_db目录下的faiss_index.index和metadata.json，重启FastAPI服务后重试！")
    except requests.exceptions.ConnectionError:
        print("连接失败：请确认FastAPI服务已启动，且地址/端口正确！")
    except Exception as e:
        print(f"未知异常：{str(e)}")

def check_health():
    try:
        response = requests.get(HEALTH_CHECK_URL, timeout=10)
        response.raise_for_status()
        health_info = response.json()
        print("服务状态正常:")
        print(f"- 状态: {health_info['status']}")
        print(f"- 索引向量数: {health_info['index_vector_count']}")
        print(f"- 文档总数: {health_info['total_sentences']}")
        print(f"- 模型路径: {health_info['model_path']}")
        return True
    except Exception as e:
        print(f"服务健康检查失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("开始测试FastAPI服务...")
    
    # 等待一段时间确保服务器启动
    print("等待服务器启动...")
    time.sleep(2)
    
    if check_health():
        print("\n开始检索测试...")
        test_search()
    else:
        print("服务未启动，请先运行 'uvicorn app:app --host 0.0.0.0 --port 8000' 启动服务")