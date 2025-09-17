import os
import argparse
from typing import Optional, List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


CONDENSE_Q_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow-up question,
rephrase the follow-up to a standalone question in the same language.

Chat History:
{chat_history}
Follow-up: {question}
Standalone question:"""
)

QA_PROMPT = PromptTemplate.from_template(
    """You are an AI assistant for answering questions about the Strikingly Help Center.
Use the provided context (from help articles) to answer conversationally and accurately.
If the answer is not in the context, say "Hmm, I'm not sure." Do not fabricate.

Question: {question}
=========
{context}
=========
Answer in Markdown:"""
)


def build_retriever(vs: FAISS, k: int, fetch_k: int, use_hybrid: bool):
    vec = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.5},
    )
    if not use_hybrid:
        return vec

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


def get_chain(vectorstore: FAISS, model: str, temperature: float, k: int, fetch_k: int, base_url: Optional[str], use_hybrid: bool):
    retriever = build_retriever(vectorstore, k=k, fetch_k=fetch_k, use_hybrid=use_hybrid)
    llm = ChatOpenAI(model=model, temperature=temperature, base_url=base_url)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=CONDENSE_Q_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Chat with the Strikingly Help Center bot")
    parser.add_argument("--store", type=str, default="faiss_store", help="FAISS local dir (folder)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Chat model")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--k", type=int, default=8, help="Top-K documents to return")
    parser.add_argument("--fetch-k", type=int, default=50, help="Candidate pool size for MMR")
    parser.add_argument("--hybrid", action="store_true", help="Enable hybrid retrieval (vector + BM25)")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible API base url, e.g. https://aizex.top/v1")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY in environment.")

    embeddings = OpenAIEmbeddings(base_url=args.base_url)
    vectorstore = FAISS.load_local(args.store, embeddings, allow_dangerous_deserialization=True)
    qa_chain = get_chain(vectorstore, args.model, args.temperature, args.k, args.fetch_k, args.base_url, args.hybrid)

    chat_history = []
    print("Chat with the Strikingly Help Center bot:")
    while True:
        try:
            question = input("Your question:\n").strip()
            if not question:
                continue
            result = qa_chain.invoke({"question": question, "chat_history": chat_history})
            answer = result["answer"]
            chat_history.append((question, answer))
            print(f"\nAI: {answer}\n")

            if result.get("source_documents"):
                print("Sources:")
                seen = set()
                for d in result["source_documents"]:
                    src = d.metadata.get("source")
                    if src and src not in seen:
                        print(f"- {src}")
                        seen.add(src)
                print()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break


if __name__ == "__main__":
    main()
