import os
import io
import json
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pdfplumber

# LangChain 및 RAG 구성요소
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

app = FastAPI(title="Google Cloud Run RAG Server")

# CORS 설정: GAS 클라이언트 등에서 접근 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 환경변수 로드 ---
# 구글 클라우드 런 환경변수 또는 로컬 .env에서 가져옵니다.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")

# Supabase 클라이언트 초기화
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
except Exception as e:
    print(f"Supabase 초기화 오류: {str(e)}")

# 모델 초기화
# 오픈 AI 임베딩
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
# 구글 LLM (질의 확장 및 답변 생성용)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite", 
    google_api_key=GOOGLE_API_KEY, 
    temperature=0.3
)

# 클라이언트 요청 데이터 모델
class ChatRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]] # [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """PDF를 텍스트로 변환하고 청크 단위로 쪼개어 Supabase에 저장합니다."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 허용됩니다.")
    
    try:
        # 1. PDF 텍스트 추출 (pdfplumber 활용)
        contents = await file.read()
        extracted_text = ""
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n\n"
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없습니다.")

        # 2. 텍스트를 청크 단위로 쪼개기
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(extracted_text)
        
        # 3. Document 객체로 변환
        documents = [
            Document(page_content=chunk, metadata={"source": file.filename, "chunk_index": i}) 
            for i, chunk in enumerate(chunks)
        ]
        
        # 4. Supabase 벡터 DB에 삽입
        # SupabaseVectorStore를 사용해 OpenAI 임베딩 후 pgvector에 저장
        SupabaseVectorStore.from_documents(
            documents,
            embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents"
        )
        
        return {
            "message": "RAG 벡터 DB 생성 성공", 
            "filename": file.filename,
            "extracted_content": extracted_text[:1000] + "... (생략)" # 미리보기 용도로 일부만 반환
        }

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF 처리 중 오류 발생: {str(e)}")


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    1) 대화 기록을 바탕으로 질의 확장 (Query Augmentation)
    2) 확장된 질의로 Supabase에서 검색 (Retriever)
    3) 검색 결과와 시스템 메시지를 기반으로 LLM 답변 생성 후 스트리밍 반환
    """
    
    # 1. 채팅 히스토리 객체 변환
    formatted_chat_history = []
    for msg in request.chat_history:
        if msg.get("role") == "user":
            formatted_chat_history.append(HumanMessage(content=msg.get("content")))
        elif msg.get("role") == "assistant":
            formatted_chat_history.append(AIMessage(content=msg.get("content")))

    # 2. Query Augmentation Chain 구성
    # 대명사 해소 및 의도를 파악하여 하나의 명료한 질의로 변환
    condense_question_system = """기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라."""
    condense_question_prompt = ChatPromptTemplate.from_messages([
        ("system", condense_question_system),
        MessagesPlaceholder(variable_name="chat_history"), # 과거 대화 내역이 통째로 들어가는 동적 빈칸
        ("human", "{question}")
    ])
    
    query_augmentation_chain = condense_question_prompt | llm | StrOutputParser()
    
    try:
        # 체인 실행 (데이터 주입)
        if formatted_chat_history:
            augmented_query = query_augmentation_chain.invoke({
                "chat_history": formatted_chat_history,
                "question": request.query
            })
        else:
            augmented_query = request.query
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질의 확장 중 오류 발생: {str(e)}")

    # 3. 증강된 질의로 벡터 DB 검색 (Retriever)
    try:
        vectorstore = SupabaseVectorStore(
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents"
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        retrieved_docs = retriever.invoke(augmented_query)
        
        # 검색된 컨텍스트 문자열 결합
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        # 출처 목록 추출
        sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in retrieved_docs]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase 검색 중 오류 발생: {str(e)}")

    # 4. LLM 답변 생성 프롬프트 설계 (도시 정책 전문가)
    qa_system_message = """너는 문서에 기반해 답변하는 도시 정책 전문가야.
아래 제공된 [Context]를 바탕으로 사용자의 질문에 전문적이고 명확하게 답변해.
만약 [Context]에 질문에 대한 정보가 없다면, 모른다고 솔직하게 답변해.

[Context]
{context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    # 5. 스트리밍 응답 제너레이터 구현
    async def stream_generator():
        # 첫 번째 SSE 이벤트: 출처(Sources) 및 증강된 질의 정보 전송
        meta_data = {
            "type": "metadata",
            "augmented_query": augmented_query,
            "sources": sources
        }
        yield f"data: {json.dumps(meta_data, ensure_ascii=False)}\n\n"
        
        # LLM 스트리밍 생성
        qa_chain = qa_prompt | llm | StrOutputParser()
        
        try:
            # astream을 사용하여 스트리밍 핸들링
            async for chunk in qa_chain.astream({
                "context": context,
                "chat_history": formatted_chat_history,
                "question": augmented_query
            }):
                # 청크 데이터를 JSON 형태로 SSE 포맷에 맞춰 전송
                chunk_data = {"type": "chunk", "text": chunk}
                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                
        except Exception as e:
            error_data = {"type": "error", "message": f"답변 생성 중 오류: {str(e)}"}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            
        finally:
            # 스트림 종료 시그널
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    # Server-Sent Events (SSE)로 반환
    return StreamingResponse(stream_generator(), media_type="text/event-stream")
