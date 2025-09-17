import argparse
import json
import time
from typing import List, Dict

import requests
from bs4 import BeautifulSoup
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

HEADERS = {"User-Agent": "strikingly-helpcenter-embeddings/1.2"}
TIMEOUT = 20
RETRIES = 3


def get_with_retry(url: str) -> requests.Response:
    last_err = None
    for _ in range(RETRIES):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise last_err


def clean_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content or "", "html.parser")
    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def fetch_zendesk_articles_en_us(api_url: str) -> List[Dict[str, str]]:
    """
    抓取 Zendesk Help Center (en-us) 的所有文章（分页），返回页面列表：
    [{"text": <title + body>, "source": <html_url>}]
    """
    pages: List[Dict[str, str]] = []
    next_page = api_url
    while next_page:
        r = get_with_retry(next_page)
        data = r.json()
        articles = data.get("articles", [])
        for a in articles:
            title = (a.get("title") or "").strip()
            body_html = a.get("body") or ""
            html_url = a.get("html_url") or ""
            text = clean_html(body_html)
            full_text = (title + "\n\n" + text).strip()
            pages.append({"text": full_text, "source": html_url})
        next_page = data.get("next_page")
    return pages


def main():
    parser = argparse.ArgumentParser(description="Create FAISS store from Zendesk Help Center (en-us)")
    parser.add_argument(
        "--zendesk",
        type=str,
        default="https://support.strikingly.com/api/v2/help_center/en-us/articles.json",
        help="Zendesk Help Center articles API URL for en-us",
    )
    parser.add_argument("--store", type=str, default="faiss_store", help="FAISS local dir (folder)")
    parser.add_argument("--pages-json", type=str, default="pages_en_us.json", help="Output JSON for pages list")
    parser.add_argument("--chunk-size", type=int, default=600, help="Splitter chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=80, help="Splitter chunk overlap")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible API base url, e.g. https://aizex.top/v1")
    parser.add_argument("--embed-model", type=str, default="text-embedding-3-small", help="Embeddings model name")
    args = parser.parse_args()

    # 1) 抓取 & 生成页面列表（title + body）
    pages = fetch_zendesk_articles_en_us(args.zendesk)
    print(f"Fetched {len(pages)} en-us articles")

    # 2) 落盘页面列表（便于验证）
    with open(args.pages_json, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"Saved pages to ./{args.pages_json}")

    # 3) 切分文本并构建向量库
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    docs, metadatas = [], []
    for p in pages:
        chunks = splitter.split_text(p["text"])
        docs.extend(chunks)
        metadatas.extend([{"source": p["source"]}] * len(chunks))

    print(f"Total chunks: {len(docs)}")
    embeddings = OpenAIEmbeddings(model=args.embed_model, base_url=args.base_url)
    vs = FAISS.from_texts(docs, embeddings, metadatas=metadatas)
    vs.save_local(args.store)
    print(f"Saved FAISS index to ./{args.store}")


if __name__ == "__main__":
    main()
