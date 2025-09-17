import argparse
import os
from typing import Optional, List

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


QA_PROMPT = PromptTemplate.from_template(
    """You are an AI assistant for answering questions about the Strikingly Help Center.
Use the provided context (from help articles) to answer concisely and accurately.
If the answer is not in the context, say "Hmm, I'm not sure." Do not fabricate.

Question: {question}
=========
{context}
=========
Answer in Markdown:"""
)


def load_vectorstore(path: str, base_url: Optional[str]):
    embeddings = OpenAIEmbeddings(base_url=base_url)
    # allow_dangerous_deserialization=True 兼容早期索引
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def build_retriever(vs: FAISS, k: int, fetch_k: int, use_hybrid: bool) -> object:
    # 向量检索 + MMR
    vec = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.5},
    )
    if not use_hybrid:
        return vec

    # 简易 BM25（关键词兜底），与向量做 Ensemble
    # 从 FAISS 的内置 docstore 取出原文和元数据
    texts: List[str] = []
    metas: List[dict] = []
    for doc in vs.docstore._dict.values():
        texts.append(doc.page_content)
        metas.append(doc.metadata)

    bm25 = BM25Retriever.from_texts(texts=texts, metadatas=metas)
    bm25.k = max(k, 10)

    return EnsembleRetriever(
        retrievers=[vec, bm25],
        weights=[0.6, 0.4],
    )


def build_chain(vs: FAISS, model: str, temperature: float, k: int, fetch_k: int, base_url: Optional[str], use_hybrid: bool):
    retriever = build_retriever(vs, k=k, fetch_k=fetch_k, use_hybrid=use_hybrid)
    llm = ChatOpenAI(model=model, temperature=temperature, base_url=base_url)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )


def main():
    parser = argparse.ArgumentParser(description="Strikingly Help Center Q&A (single turn)")
    parser.add_argument("question", type=str, help="Your question")
    parser.add_argument("--k", type=int, default=8, help="Top-K documents to return")
    parser.add_argument("--fetch-k", type=int, default=50, help="Candidate pool size for MMR")
    parser.add_argument("--hybrid", action="store_true", help="Enable hybrid retrieval (vector + BM25)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Chat model")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--store", type=str, default="faiss_store", help="FAISS local dir (folder)")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible API base url, e.g. https://aizex.top/v1")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY in environment.")

    vs = load_vectorstore(args.store, args.base_url)
    chain = build_chain(vs, args.model, args.temperature, args.k, args.fetch_k, args.base_url, args.hybrid)

    # 使用 .invoke（避免弃用警告）
    result = chain.invoke({"query": args.question})

    answer = result["result"]
    sources = []
    for doc in result.get("source_documents", []):
        src = doc.metadata.get("source") or "unknown"
        sources.append(src)

    # 去重并保持顺序
    seen, unique = set(), []
    for s in sources:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    print(f"Answer:\n{answer}\n")
    print("Sources:")
    for s in unique:
        print(f"- {s}")


if __name__ == "__main__":
    main()
