import os
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import requests
from bs4 import BeautifulSoup

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

def get_article_text(url: str) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        # 느린 사이트 대기 시간을 2초로 단축
        response = requests.get(url, headers=headers, timeout=2)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.decompose()

        article = soup.find('article', class_='story-news article')
        if article: return article.get_text(strip=True)
        
        if soup.find('article'):
            return soup.find('article').get_text(strip=True)
        
        cm_ad = soup.find('div', id="CmAdContent")
        if cm_ad:
            return cm_ad.get_text(strip=True)
            
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if len(p.get_text(strip=True)) > 30]
        if paragraphs:
            return " ".join(paragraphs)
            
        return ""
    except Exception:
        return ""

def process_single_doc(doc: str) -> str:
    if "link:" not in doc:
        return ""
    raw_link = doc.split("link:")[1].strip()
    link = raw_link.rstrip("]").split(",")[0].strip()
    snippet_info = doc.split("link:")[0].strip().lstrip("[")
    
    body_text = get_article_text(link)
    if body_text:
        return f"[출처: {link}]\n[요약: {snippet_info}]\n[본문]: {body_text[:1000]}"
    return f"[출처: {link}]\n[뉴스 요약]: {snippet_info}"

def get_search_context(query: str) -> str:
    wrapper = DuckDuckGoSearchAPIWrapper(region="kr-kr", time="w")
    search = DuckDuckGoSearchResults(
        api_wrapper=wrapper, 
        source="news", 
        results_separator=';\n'
    )
    
    try:
        docs = search.invoke(query)
    except Exception as e:
        return f"검색 중 오류가 발생했습니다: {e}"
    
    if not docs:
        return "관련 기사를 찾을 수 없습니다."

    doc_list = docs.split(";\n")[:2]
    
    # 2개 기사를 동시에 병렬 스크래핑
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(process_single_doc, doc_list))
        
    articles = [r for r in results if r]
    return "\n\n".join(articles) if articles else docs

@app.post("/api/chat")
def chat_with_bot(req: ChatRequest):
    try:
        # 모델을 gemini-3.5-flash-lite로 변경!
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite", 
            temperature=0.2, 
            api_key=os.getenv("GEMINI_API_KEY")
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "사용자의 질문에 대해 아래 context에 기반하여 답변하라.:\n\n{context}"),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{question}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        context = get_search_context(req.message)
        
        formatted_history = []
        for h in req.history:
            formatted_history.append((h["role"], h["content"]))
            
        result = chain.invoke({
            "context": context,
            "messages": formatted_history,
            "question": req.message
        })
        
        return {"reply": result}
        
    except Exception as e:
        return {"reply": f"⚠️ 서버 처리 중 오류가 발생했습니다: {str(e)}"}
