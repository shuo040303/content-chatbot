import argparse
import os
from typing import Optional, List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
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

CONDENSE_Q_PROMPT = PromptTemplate.from_template(
    """Rewrite the user input into a focused, standalone search query for retrieving Help Center articles.
Include likely synonyms if helpful (e.g., cancel/termination/downgrade/unsubscribe/stop auto-renew).

Chat History:
{chat_history}
User input: {question}
Searchable standalone query:"""
)


def load_vectorstore(path: str, base_url: Optional[str]) -> FAISS:
    embeddings = OpenAIEmbeddings(base_url=base_url)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def build_hybrid_retriever(vs: FAISS, k: int, fetch_k: int):
    # 向量 + MMR
    vec = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.5},
    )

    # 关键词 BM25：用原始文本 + metadata（包括我们新加的 title）
    texts: List[str] = []
    metas: List[dict] = []
    for doc in vs.docstore._dict.values():
        texts.append(doc.page_content)
        metas.append(doc.metadata)

    bm25 = BM25Retriever.from_texts(texts=texts, metadatas=metas)
    bm25.k = max(k, 10)

    # 融合：向量 0.6 + BM25 0.4（可按需调整）
    return EnsembleRetriever(retrievers=[vec, bm25], weights=[0.6, 0.4])


def build_chain(
    vectorstore: FAISS,
    model: str,
    temperature: float,
    k: int,
    fetch_k: int,
    base_url: Optional[str],
) -> ConversationalRetrievalChain:
    retriever = build_hybrid_retriever(vectorstore, k=k, fetch_k=fetch_k)
    llm = ChatOpenAI(model=model, temperature=temperature, base_url=base_url)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=CONDENSE_Q_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Strikingly Help Center Q&A (single turn, ConversationalRetrievalChain + Hybrid)")
    parser.add_argument("question", type=str, help="Your question")
    parser.add_argument("--store", type=str, default="faiss_store", help="FAISS local dir (folder)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Chat model")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--k", type=int, default=10, help="Top-K documents to return")
    parser.add_argument("--fetch-k", type=int, default=60, help="Candidate pool size for MMR")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible API base url, e.g. https://aizex.top/v1")
    parser.add_argument("--debug", action="store_true", help="Print retrieved URLs and titles")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY in environment.")

    vs = load_vectorstore(args.store, args.base_url)
    chain = build_chain(vs, args.model, args.temperature, args.k, args.fetch_k, args.base_url)

    # 单轮调用，满足链的输入要求：chat_history 传空列表
    result = chain.invoke({"question": args.question, "chat_history": []})

    answer = result.get("answer", "").strip()
    source_docs = result.get("source_documents", []) or []

    # 提取来源 URL（以及可选标题，用于 debug）
    urls: List[str] = []
    debug_lines: List[str] = []
    for i, doc in enumerate(source_docs, 1):
        src = doc.metadata.get("source") or "unknown"
        title = doc.metadata.get("title") or ""
        urls.append(src)
        if args.debug:
            debug_lines.append(f"{i}. {title} — {src}")

    # 去重并保持顺序
    seen, unique = set(), []
    for s in urls:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    print(f"Answer:\n{answer}\n")
    print("Sources:")
    for s in unique:
        print(f"- {s}")

    if args.debug and debug_lines:
        print("\n[DEBUG] Retrieved (title — url):")
        for line in debug_lines:
            print(line)


if __name__ == "__main__":
    main()
