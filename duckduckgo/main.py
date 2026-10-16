import os
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
        # 1. 언론사 봇 차단을 피하기 위해 브라우저 헤더 추가
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 불필요한 스크립트/스타일 태그 제거
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.decompose()

        # 콜랩 예제에 명시된 특정 영역 추출 로직
        article = soup.find('article', class_='story-news article')
        if article: return article.get_text(strip=True)
        
        if soup.find('article'):
            return soup.find('article').get_text(strip=True)
        
        cm_ad = soup.find('div', id="CmAdContent")
        if cm_ad:
            return cm_ad.get_text(strip=True)
            
        # 2. 위 태그가 없는 일반 언론사들을 위해 본문(p 태그들) 추출 추가
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if len(p.get_text(strip=True)) > 30]
        if paragraphs:
            return " ".join(paragraphs)
            
        return ""
    except Exception as e:
        return ""

def get_search_context(query: str) -> str:
    # 1. DuckDuckGo API wrapper (한국지역, 최근 일주일)
    wrapper = DuckDuckGoSearchAPIWrapper(region="kr-kr", time="w")
    
    # 2. DuckDuckGoSearchResults (뉴스 소스)
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

    # 3. 링크 추출 및 스크래핑 (대괄호 제거 + 검색 요약본 보존)
    articles = []
    for doc in docs.split(";\n")[:3]:  # 상위 3개 뉴스 활용
        if "link:" in doc:
            # link: 뒤의 주소를 가져오되 끝에 붙는 ']'나 공백, 콤마를 깔끔하게 제거
            raw_link = doc.split("link:")[1].strip()
            link = raw_link.rstrip("]").split(",")[0].strip()
            
            # 기사를 못 긁어올 때를 대비해 덕덕고 자체 요약본(snippet/title)도 함께 보관
            snippet_info = doc.split("link:")[0].strip().lstrip("[")
            
            body_text = get_article_text(link)
            if body_text:
                articles.append(f"[출처: {link}]\n[요약: {snippet_info}]\n[본문]: {body_text[:1500]}")
            else:
                # 스크래핑이 막혀도 요약본으로 답변할 수 있도록 처리
                articles.append(f"[출처: {link}]\n[뉴스 요약]: {snippet_info}")
    
    return "\n\n".join(articles) if articles else docs

@app.post("/api/chat")
def chat_with_bot(req: ChatRequest):
    try:
        # LLM 초기화
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash", 
            temperature=0.2, 
            api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # LCEL 프롬프트 설정
        prompt = ChatPromptTemplate.from_messages([
            ("system", "사용자의 질문에 대해 아래 context에 기반하여 답변하라.:\n\n{context}"),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{question}")
        ])
        
        # 체인 구성
        chain = prompt | llm | StrOutputParser()
        
        # 웹 스크래핑을 통한 컨텍스트 수집
        context = get_search_context(req.message)
        
        # 메모리(히스토리) 포맷팅
        formatted_history = []
        for h in req.history:
            formatted_history.append((h["role"], h["content"]))
            
        # 결과 생성
        result = chain.invoke({
            "context": context,
            "messages": formatted_history,
            "question": req.message
        })
        
        return {"reply": result}
        
    except Exception as e:
        return {"reply": f"⚠️ 서버 처리 중 오류가 발생했습니다: {str(e)}"}
