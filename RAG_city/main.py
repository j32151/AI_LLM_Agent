import os
import io
import json
import asyncio
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pdfplumber

# LangChain 및 RAG 구성요소
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 수정: 패키지명 오타 (splitter -> splitters)
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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
# 수정: 모델명을 환경변수로 바꿀 수 있게 함 (재배포 없이 교체 가능)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# 수정: 필요한 환경변수를 한 번에 검사해서, 무엇이 빠졌는지 로그에 명확히 남김
_required_env = {
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_API_KEY": SUPABASE_API_KEY,
}
_missing = [name for name, value in _required_env.items() if not value]
if _missing:
    raise ValueError(f"필수 환경변수가 설정되지 않았습니다: {', '.join(_missing)}")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

# 모델 초기화
# 오픈 AI 임베딩
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

# 구글 LLM (질의 확장 및 답변 생성용)
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0.3,
)

# 벡터스토어는 요청마다 만들 필요가 없으므로 한 번만 생성
vectorstore = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

# 클라이언트 요청 데이터 모델
class ChatRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]] = []  # [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]


@app.get("/")
async def health():
    """헬스 체크용 엔드포인트"""
    return {"status": "ok"}


def _extract_pdf_text(contents: bytes) -> str:
    """PDF 바이트에서 텍스트 추출 (동기 함수, 스레드에서 실행)"""
    extracted_text = ""
    with pdfplumber.open(io.BytesIO(contents)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n\n"
    return extracted_text


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """PDF를 텍스트로 변환하고 청크 단위로 쪼개어 Supabase에 저장합니다."""
    # 수정: 대소문자 구분 없이 확장자 검사 (.PDF도 허용)
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 허용됩니다.")

    try:
        # 1. PDF 텍스트 추출 (수정: 무거운 동기 작업은 스레드로 넘겨 이벤트 루프 블로킹 방지)
        contents = await file.read()
        extracted_text = await asyncio.to_thread(_extract_pdf_text, contents)

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없습니다.")

        # 2. 텍스트를 청크 단위로 쪼개기
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(extracted_text)

        # 3. Document 객체로 변환
        documents = [
            Document(page_content=chunk, metadata={"source": file.filename, "chunk_index": i})
            for i, chunk in enumerate(chunks)
        ]

        # 4. Supabase 벡터 DB에 삽입 (임베딩 + 업로드도 스레드에서 실행)
        await asyncio.to_thread(vectorstore.add_documents, documents)

        preview = extracted_text[:1000]
        if len(extracted_text) > 1000:
            preview += "... (생략)"

        return {
            "message": "RAG 벡터 DB 생성 성공",
            "filename": file.filename,
            "chunk_count": len(documents),
            "extracted_content": preview,  # 미리보기 용도로 일부만 반환
        }

    # 수정: 위에서 의도적으로 던진 HTTPException(400)이 500으로 바뀌지 않도록 그대로 전달
    except HTTPException:
        raise
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
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "user":
            formatted_chat_history.append(HumanMessage(content=content))
        elif role == "assistant":
            formatted_chat_history.append(AIMessage(content=content))

    # 2. Query Augmentation Chain 구성
    condense_question_system = """기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라."""
    condense_question_prompt = ChatPromptTemplate.from_messages([
        ("system", condense_question_system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    query_augmentation_chain = condense_question_prompt | llm | StrOutputParser()

    try:
        if formatted_chat_history:
            # 수정: async 함수 안에서는 ainvoke 사용 (이벤트 루프 블로킹 방지)
            augmented_query = await query_augmentation_chain.ainvoke({
                "chat_history": formatted_chat_history,
                "question": request.query,
            })
        else:
            augmented_query = request.query
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질의 확장 중 오류 발생: {str(e)}")

    # 3. 증강된 질의로 벡터 DB 검색 (Retriever)
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        retrieved_docs = await retriever.ainvoke(augmented_query)

        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in retrieved_docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase 검색 중 오류 발생: {str(e)}")

    # 4. LLM 답변 생성 프롬프트 설계
    qa_system_message = """너는 문서에 기반해 답변하는 도시 정책 전문가야.
아래 제공된 [Context]를 바탕으로 사용자의 질문에 전문적이고 명확하게 답변해.
만약 [Context]에 질문에 대한 정보가 없다면, 모른다고 솔직하게 답변해.

[Context]
{context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    qa_chain = qa_prompt | llm | StrOutputParser()

    # 5. 스트리밍 응답 제너레이터 구현
    async def stream_generator():
        meta_data = {
            "type": "metadata",
            "augmented_query": augmented_query,
            "sources": sources,
        }
        yield f"data: {json.dumps(meta_data, ensure_ascii=False)}\n\n"

        try:
            async for chunk in qa_chain.astream({
                "context": context,
                "chat_history": formatted_chat_history,
                "question": augmented_query,
            }):
                chunk_data = {"type": "chunk", "text": chunk}
                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

        except Exception as e:
            error_data = {"type": "error", "message": f"답변 생성 중 오류: {str(e)}"}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

        finally:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
