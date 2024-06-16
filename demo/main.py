import asyncio

from dotenv import dotenv_values
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.legacy.llms import OpenAILike as OpenAI
from qdrant_client import models
from tqdm.asyncio import tqdm

from pipeline.ingestion import build_pipeline, build_vector_store, read_data
from pipeline.qa import read_jsonl, save_answers
from pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval
from llama_index.core.postprocessor.types import BaseNodePostprocessor
import argparse
from custom.template import QA_TEMPLATES

all_emds = {
    'BAAI': 'BAAI/bge-small-zh-v1.5',
    'BAAI-L': 'BAAI/bge-large-zh-v1.5',
    'GTE-L': 'thenlper/gte-large-zh',
    'GTE-B': 'thenlper/gte-base-zh',
    # 'SENSE-L-v2': {'name': 'sensenova/piccolo-large-zh-v2', 'dim': 1792},
    # 'SENSE-L': {'name': 'sensenova/piccolo-large-zh', 'dim': 1024},
    # 'SENSE-B': {'name': 'sensenova/piccolo-base-zh', 'dim': 768},
}
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-emd', type=str, default='BAAI')
    parser.add_argument('-r_top_k', type=int, default=3)
    parser.add_argument('-qat_idx', type=int, default=0)
    parser.add_argument('--use_reranker', action='store_true')
    args = parser.parse_args()
    config = dotenv_values(".env")
    config["COLLECTION_NAME"] += '_' + args.emd
    qa_template = QA_TEMPLATES[args.qat_idx]
    # 初始化 LLM 嵌入模型 和 Reranker
    llm = OpenAI(
        api_key=config["GLM_KEY"],
        model="glm-4",
        api_base="https://open.bigmodel.cn/api/paas/v4/",
        is_chat_model=True,
    )
    embeding = HuggingFaceEmbedding(
        model_name=all_emds[args.emd],
        cache_folder="./",
        embed_batch_size=128,
    )
    Settings.embed_model = embeding

    # 初始化 数据ingestion pipeline 和 vector store
    client, vector_store = await build_vector_store(config, reindex=False)

    collection_info = await client.get_collection(
        config["COLLECTION_NAME"] or "aiops24"
    )

    if collection_info.points_count == 0:
        data = read_data("data")
        pipeline = build_pipeline(llm, embeding, vector_store=vector_store)
        # 暂时停止实时索引
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )
        await pipeline.arun(documents=data, show_progress=True, num_workers=1)
        # 恢复实时索引
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        )
        print(len(data))

    retriever = QdrantRetriever(vector_store, embeding, similarity_top_k=args.r_top_k)

    queries = read_jsonl("question.jsonl")

    # 生成答案
    print("Start generating answers...")

    results = []
    for query in tqdm(queries, total=len(queries)):
        result = await generation_with_knowledge_retrieval(
            query["query"], retriever, llm, qa_template=qa_template, reranker=BaseNodePostprocessor if args.use_reranker else None
        )
        results.append(result)

    # 处理结果
    save_answers(queries, results, f"{args.qat_idx}_{args.emd}_{args.r_top_k}_submit_result.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
