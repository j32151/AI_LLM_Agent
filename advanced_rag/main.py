"""
================================================================================
[Advanced RAG Server - main.py]
FAISS Index + Multi-Query (5x) + Hybrid Search (FAISS + BM25) + RAG Fusion (RRF)
================================================================================
본 백엔드 서버는 Google Apps Script(Code.gs / index.html) 클라이언트와 연동되며,
다음 4단계 핵심 파이프라인과 실시간 StreamHandler 이벤트를 제공합니다.

1. FAISS Index 구축 계층 (FAISS Index Layer)
   - IndexFlatL2 (정확한 L2 거리 계산, O(N*d)) 또는 IndexHNSWFlat (근사 그래프 탐색, O(d*log N)) 지원
   - PDF 페이지별 텍스트 추출, 슬라이딩 윈도우 청킹(Chunking), 메타데이터 적재

2. 다중 쿼리 확장 계층 (Multi-Query Layer)
   - 사용자의 단일 질의를 LLM을 통해 서로 다른 5가지 관점(정의, 원리, 활용, 비교/한계, 핵심키워드)의
     파생 쿼리로 확장하여 단일 쿼리 검색의 어휘 불일치(Vocabulary Mismatch) 및 누락 방지

3. 하이브리드 검색 계층 (Hybrid Search & FAISS Layer)
   - 확장된 5개 쿼리 각각에 대해:
     (a) FAISS Dense Vector Search: 문맥/의미적 유사도 검색
     (b) BM25 Sparse Keyword Search: 정확한 키워드 및 전문 용어 매칭
   - 두 검색 경로를 병행 수행하여 의미적 재현율(Recall)과 키워드 정밀도(Precision)를 동시 확보

4. RRF 순위 융합 계층 (RAG Fusion & RRF Layer)
   - Reciprocal Rank Fusion 공식: RRF_Score(d) = Σ [ w_d / (k + rank_dense(q, d)) + w_s / (k + rank_bm25(q, d)) ]
   - 다중 검색 경로의 순위를 상호 융합하여 상위 K개의 최적 문서를 재정렬(Re-ranking)

5. 실시간 스트리밍 핸들러 (StreamHandler)
   - 각 파이프라인 단계별 진행률, 소요 시간, 중간 산출물을 NDJSON 스트림 및 이벤트 로그로 전달
================================================================================
"""

import base64
import hashlib
import io
import json
import math
import os
import re
import time
import traceback
from dataclasses import dataclass, asdict, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import faiss
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

# Google Gemini SDK 로드 (설치 및 환경변수 유무에 따라 안전하게 초기화)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ==============================================================================
# 0. 알고리즘 복잡도 상수 및 시스템 설정
# ==============================================================================
EMBEDDING_DIM: int = 768  # 고차원 임베딩 벡터 차원 (text-embedding-004 호환 768차원)
RRF_K: int = 60           # RAG Fusion RRF 표준 평활화 상수 (k=60)
MULTI_QUERY_COUNT: int = 5  # 다중 쿼리 확장 개수 (요구사항: 5가지 관점으로 확장)

ALGORITHM_COMPLEXITY_INFO: Dict[str, Any] = {
    "faiss_flat_l2": {
        "name": "FAISS IndexFlatL2 (Exhaustive Dense Search)",
        "build_time": "O(N * d)",
        "search_time": "O(N * d)",
        "space": "O(N * d)",
        "description": "모든 N개 문서 벡터(차원 d)와의 유클리드 거리(L2)를 전수 계산하여 100% 정확한 최근접 이웃 탐색 보장"
    },
    "faiss_hnsw_flat": {
        "name": "FAISS IndexHNSWFlat (Hierarchical Navigable Small World)",
        "build_time": "O(N * log(N) * d)",
        "search_time": "O(log(N) * d)",
        "space": "O(N * (d + M_graph))",
        "description": "다계층 소세계 그래프(HNSW)를 통해 대규모 벡터 공간에서 로그 시간 복잡도로 초고속 근사 최근접 탐색 수행"
    },
    "bm25_sparse": {
        "name": "BM25Okapi (Sparse Keyword Inverted Matching)",
        "build_time": "O(N * L_avg)",
        "search_time": "O(|Q| * N)",
        "space": "O(V + N * L_avg)",
        "description": "TF-IDF 포화도 및 문서 길이 정규화를 반영한 키워드 매칭 (|Q|: 질의 토큰 수, L_avg: 평균 문서 길이)"
    },
    "multi_query_hybrid": {
        "name": "5-Way Multi-Query + Hybrid Search",
        "search_time": "O(M * (N * d + |Q| * N))  [M = 5 queries]",
        "description": "단일 질의를 M=5개의 다각도 파생 쿼리로 확장 후 각 쿼리당 Dense(FAISS) + Sparse(BM25) 병행 검색"
    },
    "rag_fusion_rrf": {
        "name": "RAG Fusion - Reciprocal Rank Fusion (RRF Re-ranking)",
        "formula": "RRF(d) = Σ_{q=1..5} [ w_dense / (k + rank_FAISS(q, d)) + w_bm25 / (k + rank_BM25(q, d)) ], k=60",
        "time_complexity": "O(M * K * log(M * K))",
        "space_complexity": "O(U)  [U: 고유 후보 문서 수 <= 2 * M * K]",
        "description": "서로 다른 스케일의 거리 점수(L2)와 확률 점수(BM25)를 별도의 정규화 없이 순위 역수 합산으로 안정적 융합"
    }
}


# ==============================================================================
# 1. 실시간 스트리밍 핸들러 (StreamHandler)
# ==============================================================================
class StreamHandler:
    """
    서버의 파이프라인 진행 상태(PDF 파싱, FAISS 인덱싱, 5-Way Multi-Query 생성,
    Hybrid Search, RRF 재정렬, LLM 답변 생성)를 실시간 수집 및 스트리밍 포맷(NDJSON)으로
    직렬화하는 핸들러 클래스입니다.
    - GAS(UrlFetchApp)의 동기/배치 응답과 스트리밍 엔드포인트(/query_stream) 양쪽을 모두 지원합니다.
    """

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.start_time = time.perf_counter()
        self.events: List[Dict[str, Any]] = []

    def emit(
        self,
        stage: str,
        progress: int,
        message: str,
        detail: Optional[Dict[str, Any]] = None,
        status: str = "running"
    ) -> Dict[str, Any]:
        """단일 스트림 이벤트를 기록하고 직렬화 가능한 딕셔너리로 반환합니다."""
        elapsed_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        event = {
            "pipeline": self.pipeline_name,
            "stage": stage,
            "progress": max(0, min(100, progress)),
            "message": message,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "timestamp": time.strftime("%H:%M:%S"),
            "detail": detail or {}
        }
        self.events.append(event)
        return event

    def to_ndjson_line(self, event: Dict[str, Any]) -> str:
        """실시간 스트리밍 전송을 위해 이벤트를 NDJSON(Newline Delimited JSON) 라인으로 변환합니다."""
        return json.dumps(event, ensure_ascii=False) + "\n"

    def get_summary(self) -> Dict[str, Any]:
        """전체 파이프라인 실행 이벤트 목록 및 총 소요 시간을 반환합니다."""
        total_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        return {
            "pipeline": self.pipeline_name,
            "total_elapsed_ms": total_ms,
            "event_count": len(self.events),
            "events": self.events
        }


# ==============================================================================
# 2. 데이터 모델 및 문서 메타데이터 구조체
# ==============================================================================
@dataclass
class DocumentChunk:
    """FAISS 및 BM25 인덱스에 적재되는 개별 문서 청크 및 메타데이터"""
    chunk_id: int
    doc_id: str
    source: str
    page: int
    text: str
    char_length: int
    token_count: int


class IngestRequest(BaseModel):
    """PDF 또는 텍스트 기반 벡터 DB 생성 요청 스키마"""
    filename: str = Field(default="uploaded_document.pdf", description="원본 PDF 파일명")
    pdf_base64: Optional[str] = Field(default=None, description="Base64 인코딩된 PDF 바이너리 데이터")
    raw_text: Optional[str] = Field(default=None, description="직접 입력 텍스트 (PDF 대안)")
    index_type: str = Field(default="IndexFlatL2", description="FAISS 인덱스 종류: IndexFlatL2 또는 IndexHNSWFlat")
    chunk_size: int = Field(default=500, ge=100, le=2000, description="청크당 최대 글자 수")
    chunk_overlap: int = Field(default=100, ge=0, le=500, description="인접 청크 간 중첩 글자 수")
    api_key: Optional[str] = Field(default=None, description="선택적 Gemini API 키")


class QueryRequest(BaseModel):
    """Multi-Query + Hybrid Search + RAG Fusion 질의 요청 스키마"""
    query: str = Field(..., min_length=1, description="사용자의 단일 입력 질의")
    top_k: int = Field(default=5, ge=1, le=15, description="RRF 재정렬 후 최종 반환할 상위 문서 수")
    dense_weight: float = Field(default=1.0, ge=0.0, le=5.0, description="FAISS Dense 검색 RRF 가중치")
    sparse_weight: float = Field(default=1.0, ge=0.0, le=5.0, description="BM25 Sparse 검색 RRF 가중치")
    rrf_k: int = Field(default=60, ge=1, le=200, description="Reciprocal Rank Fusion 상수 k")
    api_key: Optional[str] = Field(default=None, description="선택적 Gemini API 키")


# ==============================================================================
# 3. 텍스트 토크나이저 & 임베딩 엔진 (Gemini API + 결정론적 고차원 폴백)
# ==============================================================================
def tokenize_korean_and_english(text: str) -> List[str]:
    """
    한국어 및 영어 하이브리드 토크나이저 (BM25 및 키워드 매칭용)
    - 어절 단위 토큰과 한국어 조사 제거/2-gram 및 3-gram 서브워드를 함께 추출하여
      한국어 교착어 특성에 따른 BM25 키워드 매칭 재현율을 높입니다.
    - 시간 복잡도: O(L) (L: 텍스트 길이)
    """
    cleaned = re.sub(r"[^\w\s가-힣]", " ", text.lower())
    raw_words = [w for w in cleaned.split() if len(w) >= 1]
    tokens: List[str] = []

    # 한국어 주요 어미/조사 제거 패턴
    josa_pattern = re.compile(r"(은|는|이|가|을|를|의|에|에서|으로|로|과|와|도|만|까지|부터|이란|란)$")

    for word in raw_words:
        tokens.append(word)
        stemmed = josa_pattern.sub("", word)
        if stemmed and stemmed != word and len(stemmed) >= 2:
            tokens.append(stemmed)
        # 복합명사 대응을 위한 문자 바이그램(2-gram) 추가 (한글 3글자 이상 단어)
        if len(word) >= 3 and re.search(r"[가-힣]", word):
            for i in range(len(word) - 1):
                tokens.append(word[i:i + 2])

    return tokens


def compute_deterministic_embedding(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """
    API 키가 없거나 네트워크 오프라인 환경일 때 동작하는 고차원 의미/문자 엔그램 해시 임베딩 생성기.
    - 동일 단어/어근을 공유하는 문장들이 벡터 공간에서 높은 코사인/L2 유사도를 갖도록
      Feature Hashing (Weinberger et al.) 방식으로 768차원 실수 벡터를 구성하고 L2 정규화합니다.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = tokenize_korean_and_english(text)
    if not tokens:
        tokens = [text]

    for idx, token in enumerate(tokens):
        # 토큰별 해시 버킷 및 부호 결정
        digest = hashlib.md5(token.encode("utf-8")).digest()
        h1 = int.from_bytes(digest[0:4], "little") % dim
        h2 = int.from_bytes(digest[4:8], "little") % dim
        sign1 = 1.0 if (digest[8] & 1) == 0 else -1.0
        sign2 = 1.0 if (digest[9] & 1) == 0 else -1.0

        # 앞부분 핵심 토큰에 미세 가중치 부여
        weight = 1.0 + (0.2 if idx < 10 else 0.0)
        vec[h1] += sign1 * weight
        vec[h2] += sign2 * (weight * 0.5)

    # L2 정규화 (단위 초구면 투영 -> IndexFlatL2 거리가 코사인 유사도와 단조 대응됨)
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec = vec / norm
    return vec.astype(np.float32)


def get_embeddings_batch(texts: List[str], api_key: Optional[str] = None) -> Tuple[np.ndarray, str]:
    """
    텍스트 리스트를 고차원 벡터 행렬 (N, d)로 변환합니다.
    - Gemini API 키가 유효하면 `models/text-embedding-004` 모델을 호출합니다.
    - API 키가 없거나 호출 실패 시 Feature Hashing 기반 768차원 결정론적 임베딩으로 자동 폴백합니다.
    """
    effective_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if GEMINI_AVAILABLE and effective_key:
        try:
            genai.configure(api_key=effective_key)
            embeddings_list = []
            # 배치 처리
            for t in texts:
                res = genai.embed_content(
                    model="models/text-embedding-004",
                    content=t,
                    task_type="retrieval_document"
                )
                emb = np.array(res["embedding"], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 1e-8:
                    emb = emb / norm
                embeddings_list.append(emb)
            matrix = np.vstack(embeddings_list).astype(np.float32)
            return matrix, "gemini-text-embedding-004"
        except Exception:
            # 폴백 처리
            pass

    matrix = np.vstack([compute_deterministic_embedding(t, EMBEDDING_DIM) for t in texts]).astype(np.float32)
    return matrix, "feature-hash-768d-l2norm"


# ==============================================================================
# 4. 다중 쿼리 확장 계층 (Multi-Query Layer - 5가지 관점 확장)
# ==============================================================================
class MultiQueryExpander:
    """
    사용자의 단일 입력 쿼리를 서로 다른 5가지 관점의 파생 쿼리들로 확장하는 계층입니다.
    [확장 관점 5가지]
      1. 개념/정의 관점 (Conceptual & Definitional)
      2. 작동 원리/구조 관점 (Mechanism & Structural)
      3. 실무 적용/절차 관점 (Practical & Procedural)
      4. 비교/한계/주의사항 관점 (Analytical & Edge-case)
      5. 핵심 키워드/동의어 조합 관점 (Keyword & Synonym Dense)
    """

    PERSPECTIVE_LABELS = [
        "관점 1 [개념·정의]",
        "관점 2 [원리·구조]",
        "관점 3 [적용·활용]",
        "관점 4 [특징·비교]",
        "관점 5 [핵심 키워드]"
    ]

    @classmethod
    def expand_into_five_queries(
        cls,
        original_query: str,
        corpus_keywords: List[str],
        api_key: Optional[str] = None
    ) -> Tuple[List[Dict[str, str]], str]:
        """
        원본 질의를 5가지 다각도 쿼리로 확장하여 반환합니다.
        반환 형식: [{"id": 1, "perspective": "...", "query": "..."}, ... (총 5개)], 사용된 모델명
        """
        q_clean = original_query.strip()
        effective_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        # 1) Gemini LLM 기반 5가지 다중 쿼리 생성 시도
        if GEMINI_AVAILABLE and effective_key:
            try:
                genai.configure(api_key=effective_key)
                model = genai.GenerativeModel("gemini-2.0-flash")
                prompt = (
                    "당신은 Advanced RAG 시스템의 Multi-Query 확장 전문가입니다.\n"
                    f"사용자의 원본 질문: \"{q_clean}\"\n\n"
                    "단일 쿼리 검색의 어휘 불일치와 정보 누락을 방지하기 위해, 위 질문을 서로 다른 5가지 관점의 "
                    "검색 쿼리로 확장하세요. 반드시 JSON 배열(문자열 5개)만 출력하세요.\n"
                    "관점 1: 핵심 개념 및 정의 중심\n"
                    "관점 2: 세부 작동 원리 및 내부 프로세스 중심\n"
                    "관점 3: 실제 적용 방법 및 구체적 사례 중심\n"
                    "관점 4: 주요 특징, 장단점 및 제약사항 중심\n"
                    "관점 5: 핵심 전문 용어 및 동의어 키워드 나열 중심\n"
                    "출력 예시: [\"쿼리1\", \"쿼리2\", \"쿼리3\", \"쿼리4\", \"쿼리5\"]"
                )
                response = model.generate_content(prompt)
                raw_text = response.text.strip()
                match = re.search(r"\[.*\]", raw_text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, list) and len(parsed) >= 5:
                        expanded = []
                        for i in range(5):
                            expanded.append({
                                "id": i + 1,
                                "perspective": cls.PERSPECTIVE_LABELS[i],
                                "query": str(parsed[i]).strip()
                            })
                        return expanded, "gemini-2.0-flash"
            except Exception:
                pass

        # 2) 규칙 및 코퍼스 문맥 기반 5가지 관점 쿼리 확장 (오프라인/폴백 보장)
        query_tokens = [t for t in tokenize_korean_and_english(q_clean) if len(t) >= 2]
        core_terms = " ".join(dict.fromkeys(query_tokens[:6])) if query_tokens else q_clean
        related_terms = " ".join(corpus_keywords[:4]) if corpus_keywords else ""

        expanded_fallback = [
            {
                "id": 1,
                "perspective": cls.PERSPECTIVE_LABELS[0],
                "query": f"{q_clean} 핵심 개념 정의 및 기본 의미 ({core_terms})"
            },
            {
                "id": 2,
                "perspective": cls.PERSPECTIVE_LABELS[1],
                "query": f"{core_terms} 세부 구조 작동 원리 프로세스 및 구성 요소"
            },
            {
                "id": 3,
                "perspective": cls.PERSPECTIVE_LABELS[2],
                "query": f"{q_clean} 실제 적용 방법 주요 활용 사례 및 실행 절차"
            },
            {
                "id": 4,
                "perspective": cls.PERSPECTIVE_LABELS[3],
                "query": f"{core_terms} 주요 특징 장단점 비교 분석 및 핵심 요약 사항"
            },
            {
                "id": 5,
                "perspective": cls.PERSPECTIVE_LABELS[4],
                "query": f"{core_terms} {related_terms} 핵심 키워드 관련 상세 내용".strip()
            }
        ]
        return expanded_fallback, "rule-perspective-expander-5x"


# ==============================================================================
# 5. FAISS 인덱스 + BM25 하이브리드 검색 & RAG Fusion (RRF) 엔진
# ==============================================================================
class AdvancedRAGEngine:
    """
    FAISS Index (IndexFlatL2 / IndexHNSWFlat) + BM25 Sparse Index +
    5-Way Multi-Query + Reciprocal Rank Fusion (RRF) 통합 관리 엔진
    """

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.index_type: str = "IndexFlatL2"
        self.embedding_dim: int = EMBEDDING_DIM
        self.embedding_model_used: str = "none"
        self.current_filename: str = "샘플_AI_RAG_아키텍처_가이드.pdf (기본 내장 문서)"
        self.last_updated: str = ""
        self.corpus_top_keywords: List[str] = []

        # 서버 기동 시 즉시 질의 테스트가 가능하도록 기본 지식 문서 적재
        self._initialize_default_knowledge_base()

    def _initialize_default_knowledge_base(self) -> None:
        """초기 상태에서 바로 질의 테스트가 가능하도록 기본 RAG 기술 문서를 인덱싱합니다."""
        sample_pages = [
            (
                1,
                "FAISS(Facebook AI Similarity Search)는 고차원 밀집 벡터(Dense Vector)의 효율적인 유사도 검색과 "
                "클러스터링을 위한 라이브러리입니다. IndexFlatL2는 모든 벡터 간의 유클리드 거리(L2 Distance)를 전수 계산하여 "
                "100% 정확한 검색 정확도(Recall@K)를 보장하며 시간 복잡도는 O(N*d)입니다. 반면 IndexHNSWFlat은 "
                "계층적 탐색 가능한 소세계 그래프(Hierarchical Navigable Small World) 구조를 구축하여 O(d*log N)의 "
                "로그 시간 복잡도로 대규모 벡터 데이터베이스에서 초고속 근사 최근접 이웃(ANN) 검색을 수행합니다."
            ),
            (
                2,
                "다중 쿼리 확장(Multi-Query Expansion)은 사용자의 단일 질문이 가진 모호성과 어휘 불일치(Vocabulary Mismatch) "
                "문제를 해결하는 기법입니다. LLM을 활용하여 원본 질문을 개념 정의, 작동 원리, 실무 활용, 한계점 비교, "
                "핵심 키워드 등 5가지 서로 다른 관점의 파생 쿼리로 확장합니다. 이를 통해 단일 쿼리로는 검색되지 않던 "
                "잠재적 관련 문서 청크들을 폭넓게 회수(High Recall)할 수 있습니다."
            ),
            (
                3,
                "하이브리드 검색(Hybrid Search)은 FAISS 기반의 밀집 벡터 검색(Dense Search)과 BM25 알고리즘 기반의 "
                "희소 키워드 검색(Sparse Search)을 결합하는 아키텍처입니다. Dense 검색은 문맥과 의미적 유사성을 파악하는 데 "
                "강점이 있으나 고유명사, 법령 조항, 모델 번호, 약어 매칭에 취약할 수 있습니다. BM25는 단어 빈도(TF) 포화와 "
                "문서 길이 정규화를 통해 정확한 키워드 일치를 찾아내어 두 방식이 상호 보완적으로 동작합니다."
            ),
            (
                4,
                "RAG Fusion과 상호 순위 융합(Reciprocal Rank Fusion, RRF)은 다중 쿼리 및 하이브리드 검색 경로에서 수집된 "
                "여러 검색 결과 리스트를 하나의 최적 순위로 통합하는 재정렬(Re-ranking) 알고리즘입니다. "
                "RRF 공식은 RRF_Score(d) = Σ 1 / (k + rank(q, d)) (일반적으로 k=60)을 사용합니다. "
                "FAISS의 L2 거리 점수와 BM25의 확률 점수는 스케일 분포가 전혀 다르지만, RRF는 원시 점수 대신 '순위(Rank)'만을 "
                "사용하므로 별도의 점수 정규화 없이도 여러 검색기에서 공통으로 상위에 랭크된 문서를 최상위로 부스트합니다."
            ),
            (
                5,
                "Google Apps Script(GAS)와 Google Cloud Run(FastAPI) 연동 아키텍처에서는 클라이언트가 UrlFetchApp을 통해 "
                "PDF 문서 인제스트(/ingest) 및 고급 RAG 질의(/query) API를 호출합니다. StreamHandler는 각 파이프라인 단계의 "
                "진행률(Progress %), 소요 시간(ms), 알고리즘 복잡도 및 중간 결과(5개 확장 쿼리, FAISS/BM25 개별 순위, "
                "RRF 통합 점수)를 실시간 기록하여 프론트엔드 UI에서 RAG 검색 근거와 LLM 최종 답변을 명확히 분리해 시각화합니다."
            )
        ]
        self.build_index_from_pages(
            pages=sample_pages,
            filename="샘플_AI_RAG_아키텍처_가이드.pdf",
            index_type="IndexFlatL2",
            chunk_size=500,
            chunk_overlap=80,
            api_key=None
        )

    @staticmethod
    def extract_pages_from_pdf_bytes(pdf_bytes: bytes) -> List[Tuple[int, str]]:
        """
        PDF 바이너리에서 페이지 번호와 텍스트를 추출합니다.
        - 시간 복잡도: O(P) (P: PDF 전체 페이지 수)
        """
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages: List[Tuple[int, str]] = []
        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_text = re.sub(r"\s+", " ", page_text).strip()
            if page_text:
                pages.append((idx + 1, page_text))
        return pages

    @staticmethod
    def chunk_pages(
        pages: List[Tuple[int, str]],
        filename: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ) -> List[DocumentChunk]:
        """
        페이지별 텍스트를 슬라이딩 윈도우 방식으로 분할하여 DocumentChunk 리스트를 생성합니다.
        - 문장 경계(., ?, !, 다.)를 우선 고려하여 문맥 단절을 최소화합니다.
        """
        chunks: List[DocumentChunk] = []
        chunk_id = 1
        step = max(50, chunk_size - chunk_overlap)
        doc_id = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:8]

        for page_num, text in pages:
            if len(text) <= chunk_size:
                tokens = tokenize_korean_and_english(text)
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        source=filename,
                        page=page_num,
                        text=text,
                        char_length=len(text),
                        token_count=len(tokens)
                    )
                )
                chunk_id += 1
            else:
                start = 0
                while start < len(text):
                    end = min(len(text), start + chunk_size)
                    # 문장 끝 경계 보정
                    if end < len(text):
                        last_period = max(
                            text.rfind(". ", start, end),
                            text.rfind("다.", start, end),
                            text.rfind("? ", start, end)
                        )
                        if last_period != -1 and (last_period - start) >= int(chunk_size * 0.5):
                            end = last_period + 2

                    sub_text = text[start:end].strip()
                    if sub_text:
                        tokens = tokenize_korean_and_english(sub_text)
                        chunks.append(
                            DocumentChunk(
                                chunk_id=chunk_id,
                                doc_id=doc_id,
                                source=filename,
                                page=page_num,
                                text=sub_text,
                                char_length=len(sub_text),
                                token_count=len(tokens)
                            )
                        )
                        chunk_id += 1
                    if end >= len(text):
                        break
                    start += step

        return chunks

    def build_index_from_pages(
        self,
        pages: List[Tuple[int, str]],
        filename: str,
        index_type: str = "IndexFlatL2",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        api_key: Optional[str] = None,
        stream_handler: Optional[StreamHandler] = None
    ) -> Dict[str, Any]:
        """
        페이지 리스트로부터 청크 분할 -> 임베딩 생성 -> FAISS 인덱스(IndexFlatL2 / IndexHNSWFlat) 및
        BM25 희소 인덱스를 동시에 구축합니다.
        """
        if stream_handler:
            stream_handler.emit(
                stage="CHUNKING",
                progress=25,
                message=f"총 {len(pages)}페이지 텍스트를 슬라이딩 윈도우(size={chunk_size}, overlap={chunk_overlap})로 분할 중..."
            )

        new_chunks = self.chunk_pages(pages, filename, chunk_size, chunk_overlap)
        if not new_chunks:
            raise ValueError("추출된 텍스트에서 유효한 문서 청크를 생성할 수 없습니다. (스캔 이미지 PDF 여부 확인 필요)")

        if stream_handler:
            stream_handler.emit(
                stage="EMBEDDING",
                progress=55,
                message=f"생성된 {len(new_chunks)}개 청크에 대해 {self.embedding_dim}차원 Dense 임베딩 벡터 계산 중..."
            )

        texts = [c.text for c in new_chunks]
        embeddings, emb_model_name = get_embeddings_batch(texts, api_key=api_key)
        num_vectors, dim = embeddings.shape
        self.embedding_dim = dim
        self.embedding_model_used = emb_model_name

        # FAISS 인덱스 생성 (IndexFlatL2 또는 IndexHNSWFlat 선택)
        normalized_type = "IndexHNSWFlat" if "hnsw" in index_type.lower() else "IndexFlatL2"
        if normalized_type == "IndexHNSWFlat":
            # HNSW 그래프 이웃 수 M=32, 탐색 깊이 efSearch=64, 구축 깊이 efConstruction=100
            index = faiss.IndexHNSWFlat(dim, 32)
            index.hnsw.efConstruction = 100
            index.hnsw.efSearch = 64
        else:
            index = faiss.IndexFlatL2(dim)

        index.add(embeddings)

        if stream_handler:
            stream_handler.emit(
                stage="FAISS_BM25_INDEXING",
                progress=85,
                message=f"FAISS({normalized_type}) 벡터 적재 및 BM25Okapi 역색인 구축 중..."
            )

        # BM25 코퍼스 토크나이징 및 인덱스 구축
        tokenized_corpus = [tokenize_korean_and_english(c.text) for c in new_chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        # 빈도 상위 키워드 추출 (Multi-Query 폴백 보강용)
        freq: Dict[str, int] = {}
        for tok_list in tokenized_corpus:
            for tok in tok_list:
                if len(tok) >= 2:
                    freq[tok] = freq.get(tok, 0) + 1
        sorted_kw = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        self.corpus_top_keywords = [k for k, _ in sorted_kw[:15]]

        # 상태 업데이트
        self.chunks = new_chunks
        self.tokenized_corpus = tokenized_corpus
        self.bm25 = bm25
        self.faiss_index = index
        self.index_type = normalized_type
        self.current_filename = filename
        self.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")

        complexity_key = "faiss_hnsw_flat" if normalized_type == "IndexHNSWFlat" else "faiss_flat_l2"
        build_stats = {
            "filename": filename,
            "page_count": len(pages),
            "chunk_count": num_vectors,
            "embedding_dim": dim,
            "embedding_model": emb_model_name,
            "faiss_index_type": normalized_type,
            "faiss_ntotal": int(index.ntotal),
            "bm25_corpus_size": len(tokenized_corpus),
            "last_updated": self.last_updated,
            "complexity": {
                "faiss": ALGORITHM_COMPLEXITY_INFO[complexity_key],
                "bm25": ALGORITHM_COMPLEXITY_INFO["bm25_sparse"]
            }
        }

        if stream_handler:
            stream_handler.emit(
                stage="COMPLETED",
                progress=100,
                message=f"벡터 DB 구축 완료! ({num_vectors}개 청크, FAISS {normalized_type} + BM25 하이브리드 준비 완료)",
                detail=build_stats,
                status="completed"
            )

        return build_stats

    def execute_rag_fusion_pipeline(
        self,
        user_query: str,
        top_k: int = 5,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        rrf_k: int = RRF_K,
        api_key: Optional[str] = None,
        stream_handler: Optional[StreamHandler] = None
    ) -> Dict[str, Any]:
        """
        [핵심 파이프라인 실행]
        Step 1: Multi-Query Layer (단일 질의 -> 5가지 관점 파생 쿼리 확장)
        Step 2: Hybrid Search Layer (5개 쿼리 각각에 대해 FAISS Dense + BM25 Sparse 병행 검색)
        Step 3: RAG Fusion & RRF Layer (Reciprocal Rank Fusion 수식으로 순위 통합 및 상위 K개 재정렬)
        Step 4: LLM Answer Synthesis (검색된 근거 스니펫과 별도로 출처가 명시된 최종 답변 생성)
        """
        if not self.chunks or self.faiss_index is None or self.bm25 is None:
            raise RuntimeError("벡터 데이터베이스가 비어 있습니다. 먼저 PDF 벡터 DB를 생성해 주세요.")

        t0 = time.perf_counter()

        # ----------------------------------------------------------------------
        # [Step 1] 다중 쿼리 확장 계층 (Multi-Query Layer: 5가지 관점으로 확장)
        # ----------------------------------------------------------------------
        if stream_handler:
            stream_handler.emit(
                stage="MULTI_QUERY_EXPANSION",
                progress=20,
                message="단일 질의를 5가지 다각도 관점(개념/원리/적용/비교/키워드)의 파생 쿼리로 확장 중..."
            )

        expanded_queries, mq_model = MultiQueryExpander.expand_into_five_queries(
            original_query=user_query,
            corpus_keywords=self.corpus_top_keywords,
            api_key=api_key
        )
        t_mq = round((time.perf_counter() - t0) * 1000, 2)

        if stream_handler:
            stream_handler.emit(
                stage="MULTI_QUERY_DONE",
                progress=40,
                message=f"5가지 파생 쿼리 생성 완료 ({t_mq}ms)",
                detail={"expanded_queries": expanded_queries, "model": mq_model}
            )

        # ----------------------------------------------------------------------
        # [Step 2] 하이브리드 검색 계층 (FAISS Dense + BM25 Sparse 병행 검색)
        # ----------------------------------------------------------------------
        t1 = time.perf_counter()
        num_docs = len(self.chunks)
        candidate_depth = min(num_docs, max(top_k * 3, 10))

        # 5개 확장 쿼리의 임베딩 일괄 생성
        query_Strings = [item["query"] for item in expanded_queries]
        query_embeddings, _ = get_embeddings_batch(query_Strings, api_key=api_key)

        # FAISS 일괄 검색: distances (5, candidate_depth), indices (5, candidate_depth)
        faiss_distances, faiss_indices = self.faiss_index.search(query_embeddings, candidate_depth)

        # 각 쿼리별 Dense & Sparse 검색 결과 기록 및 RRF 누적 테이블 초기화
        rrf_scores: Dict[int, float] = {}
        doc_provenance: Dict[int, Dict[str, Any]] = {}
        per_query_search_breakdown: List[Dict[str, Any]] = []

        for q_idx, q_item in enumerate(expanded_queries):
            q_text = q_item["query"]
            q_perspective = q_item["perspective"]

            # (A) FAISS Dense 검색 순위 처리
            dense_hits: List[Dict[str, Any]] = []
            for rank_zero, doc_idx in enumerate(faiss_indices[q_idx]):
                if doc_idx < 0 or doc_idx >= num_docs:
                    continue
                rank = rank_zero + 1
                l2_dist = float(faiss_distances[q_idx][rank_zero])
                dense_sim = round(1.0 / (1.0 + max(0.0, l2_dist)), 4)

                # RRF 점수 가산 (Dense 경로): w_dense / (k + rank_dense)
                rrf_contrib = dense_weight / (rrf_k + rank)
                rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + rrf_contrib

                if doc_idx not in doc_provenance:
                    doc_provenance[doc_idx] = {
                        "best_dense_rank": rank,
                        "best_bm25_rank": None,
                        "max_dense_similarity": dense_sim,
                        "min_l2_distance": round(l2_dist, 4),
                        "max_bm25_score": 0.0,
                        "dense_rrf_sum": 0.0,
                        "sparse_rrf_sum": 0.0,
                        "matched_queries": []
                    }
                prov = doc_provenance[doc_idx]
                prov["dense_rrf_sum"] += rrf_contrib
                if prov["best_dense_rank"] is None or rank < prov["best_dense_rank"]:
                    prov["best_dense_rank"] = rank
                if dense_sim > prov["max_dense_similarity"]:
                    prov["max_dense_similarity"] = dense_sim
                    prov["min_l2_distance"] = round(l2_dist, 4)

                prov["matched_queries"].append({
                    "query_id": q_item["id"],
                    "perspective": q_perspective,
                    "channel": "FAISS_DENSE",
                    "rank": rank,
                    "rrf_contribution": round(rrf_contrib, 6)
                })
                if rank <= 3:
                    dense_hits.append({
                        "chunk_id": self.chunks[doc_idx].chunk_id,
                        "page": self.chunks[doc_idx].page,
                        "rank": rank,
                        "similarity": dense_sim
                    })

            # (B) BM25 Sparse 키워드 검색 순위 처리
            q_tokens = tokenize_korean_and_english(q_text)
            bm25_raw_scores = self.bm25.get_scores(q_tokens)
            bm25_sorted_indices = np.argsort(bm25_raw_scores)[::-1][:candidate_depth]

            sparse_hits: List[Dict[str, Any]] = []
            for rank_zero, doc_idx_np in enumerate(bm25_sorted_indices):
                doc_idx = int(doc_idx_np)
                bm25_score = float(bm25_raw_scores[doc_idx])
                rank = rank_zero + 1

                # 토큰 매칭 점수가 0보다 큰 경우 또는 상위 후보에 대해 RRF 가산
                effective_weight = sparse_weight if bm25_score > 0 else (sparse_weight * 0.25)
                rrf_contrib = effective_weight / (rrf_k + rank)
                rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + rrf_contrib

                if doc_idx not in doc_provenance:
                    doc_provenance[doc_idx] = {
                        "best_dense_rank": None,
                        "best_bm25_rank": rank,
                        "max_dense_similarity": 0.0,
                        "min_l2_distance": None,
                        "max_bm25_score": round(bm25_score, 4),
                        "dense_rrf_sum": 0.0,
                        "sparse_rrf_sum": 0.0,
                        "matched_queries": []
                    }
                prov = doc_provenance[doc_idx]
                prov["sparse_rrf_sum"] += rrf_contrib
                if prov["best_bm25_rank"] is None or rank < prov["best_bm25_rank"]:
                    prov["best_bm25_rank"] = rank
                if bm25_score > prov["max_bm25_score"]:
                    prov["max_bm25_score"] = round(bm25_score, 4)

                prov["matched_queries"].append({
                    "query_id": q_item["id"],
                    "perspective": q_perspective,
                    "channel": "BM25_SPARSE",
                    "rank": rank,
                    "rrf_contribution": round(rrf_contrib, 6)
                })
                if rank <= 3:
                    sparse_hits.append({
                        "chunk_id": self.chunks[doc_idx].chunk_id,
                        "page": self.chunks[doc_idx].page,
                        "rank": rank,
                        "bm25_score": round(bm25_score, 4)
                    })

            per_query_search_breakdown.append({
                "query_id": q_item["id"],
                "perspective": q_perspective,
                "expanded_query": q_text,
                "top_dense_hits": dense_hits,
                "top_sparse_hits": sparse_hits
            })

        t_hybrid = round((time.perf_counter() - t1) * 1000, 2)
        if stream_handler:
            stream_handler.emit(
                stage="HYBRID_SEARCH_DONE",
                progress=65,
                message=f"5개 쿼리 x (FAISS {self.index_type} + BM25) 총 10개 검색 경로 탐색 완료 ({t_hybrid}ms)"
            )

        # ----------------------------------------------------------------------
        # [Step 3] RRF 순위 융합 계층 (RAG Fusion & RRF Re-ranking)
        # ----------------------------------------------------------------------
        t2 = time.perf_counter()
        sorted_doc_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)[:top_k]

        # 이론적 최대 RRF 점수 (5개 쿼리 모두 Dense 1위 + Sparse 1위일 때) 대비 백분율 계산
        max_possible_rrf = len(expanded_queries) * ((dense_weight / (rrf_k + 1)) + (sparse_weight / (rrf_k + 1)))

        reranked_documents: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []

        for final_rank, doc_idx in enumerate(sorted_doc_indices, start=1):
            chunk = self.chunks[doc_idx]
            prov = doc_provenance[doc_idx]
            raw_rrf = rrf_scores[doc_idx]
            normalized_rrf_pct = round((raw_rrf / max_possible_rrf) * 100, 1) if max_possible_rrf > 0 else 0.0

            citation_label = f"[{final_rank}] {chunk.source} (p.{chunk.page}, Chunk #{chunk.chunk_id})"
            doc_entry = {
                "final_rank": final_rank,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "source": chunk.source,
                "page": chunk.page,
                "snippet": chunk.text,
                "char_length": chunk.char_length,
                "token_count": chunk.token_count,
                "rrf_score": round(raw_rrf, 6),
                "rrf_normalized_percent": normalized_rrf_pct,
                "dense_rrf_component": round(prov["dense_rrf_sum"], 6),
                "sparse_rrf_component": round(prov["sparse_rrf_sum"], 6),
                "best_dense_rank": prov["best_dense_rank"],
                "best_bm25_rank": prov["best_bm25_rank"],
                "dense_similarity": prov["max_dense_similarity"],
                "l2_distance": prov["min_l2_distance"],
                "bm25_score": prov["max_bm25_score"],
                "citation_tag": citation_label,
                "matched_paths_count": len(prov["matched_queries"])
            }
            reranked_documents.append(doc_entry)
            citations.append({
                "ref_num": final_rank,
                "source": chunk.source,
                "page": chunk.page,
                "chunk_id": chunk.chunk_id,
                "rrf_score": round(raw_rrf, 6),
                "label": citation_label
            })

        t_rrf = round((time.perf_counter() - t2) * 1000, 2)
        if stream_handler:
            stream_handler.emit(
                stage="RRF_FUSION_DONE",
                progress=82,
                message=f"Reciprocal Rank Fusion(k={rrf_k}) 재정렬 완료: 상위 {len(reranked_documents)}개 문서 선별 ({t_rrf}ms)",
                detail={"top_rrf_score": reranked_documents[0]["rrf_score"] if reranked_documents else 0}
            )

        # ----------------------------------------------------------------------
        # [Step 4] LLM 최종 답변 생성 (검색 근거 스니펫과 분리된 종합 답변 + 출처 표기)
        # ----------------------------------------------------------------------
        t3 = time.perf_counter()
        llm_answer, answer_model = self._synthesize_answer_with_citations(
            user_query=user_query,
            reranked_docs=reranked_documents,
            api_key=api_key
        )
        t_llm = round((time.perf_counter() - t3) * 1000, 2)
        total_ms = round((time.perf_counter() - t0) * 1000, 2)

        if stream_handler:
            stream_handler.emit(
                stage="ANSWER_SYNTHESIS_DONE",
                progress=100,
                message=f"RAG Fusion 기반 최종 답변 및 출처 생성 완료 (총 {total_ms}ms)",
                status="completed"
            )

        complexity_key = "faiss_hnsw_flat" if self.index_type == "IndexHNSWFlat" else "faiss_flat_l2"
        return {
            "status": "success",
            "original_query": user_query,
            "expanded_queries": expanded_queries,
            "per_query_breakdown": per_query_search_breakdown,
            "rag_retrieved_documents": reranked_documents,
            "llm_answer": llm_answer,
            "citations": citations,
            "metadata": {
                "active_document": self.current_filename,
                "total_indexed_chunks": num_docs,
                "faiss_index_type": self.index_type,
                "embedding_dim": self.embedding_dim,
                "embedding_model": self.embedding_model_used,
                "multi_query_model": mq_model,
                "answer_model": answer_model,
                "rrf_k": rrf_k,
                "dense_weight": dense_weight,
                "sparse_weight": sparse_weight,
                "top_k": top_k,
                "timings_ms": {
                    "multi_query_ms": t_mq,
                    "hybrid_search_ms": t_hybrid,
                    "rrf_fusion_ms": t_rrf,
                    "llm_synthesis_ms": t_llm,
                    "total_ms": total_ms
                }
            },
            "complexity": {
                "active_faiss": ALGORITHM_COMPLEXITY_INFO[complexity_key],
                "bm25": ALGORITHM_COMPLEXITY_INFO["bm25_sparse"],
                "multi_query_hybrid": ALGORITHM_COMPLEXITY_INFO["multi_query_hybrid"],
                "rag_fusion_rrf": ALGORITHM_COMPLEXITY_INFO["rag_fusion_rrf"],
                "runtime_metrics": {
                    "N_chunks": num_docs,
                    "d_dimension": self.embedding_dim,
                    "M_expanded_queries": len(expanded_queries),
                    "K_candidates_per_channel": candidate_depth,
                    "total_search_channels": len(expanded_queries) * 2
                }
            }
        }

    @staticmethod
    def _synthesize_answer_with_citations(
        user_query: str,
        reranked_docs: List[Dict[str, Any]],
        api_key: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        RRF로 재정렬된 상위 문서 스니펫만을 근거로 사용하여 출처([1], [2] 등)가 명확히 표기된
        LLM 최종 답변을 생성합니다.
        """
        if not reranked_docs:
            return "검색된 관련 문서가 없습니다. 먼저 관련 PDF를 업로드하여 벡터 DB를 생성해 주세요.", "none"

        context_blocks = []
        for doc in reranked_docs:
            context_blocks.append(
                f"[{doc['final_rank']}] 출처: {doc['source']} (p.{doc['page']}, Chunk #{doc['chunk_id']}, "
                f"RRF={doc['rrf_score']}, FAISS순위={doc['best_dense_rank']}, BM25순위={doc['best_bm25_rank']})\n"
                f"내용: {doc['snippet']}"
            )
        joined_context = "\n\n".join(context_blocks)

        effective_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if GEMINI_AVAILABLE and effective_key:
            try:
                genai.configure(api_key=effective_key)
                model = genai.GenerativeModel("gemini-2.0-flash")
                prompt = (
                    "당신은 FAISS + Multi-Query + Hybrid Search + RAG Fusion 파이프라인의 최종 답변 생성 AI입니다.\n"
                    "아래 [RRF 상위 검색 문서]만을 근거로 사용자의 질문에 대해 체계적이고 명확한 한국어로 답변하세요.\n"
                    "반드시 답변 문장 끝에 인용한 문서 번호와 페이지 출처(예: [1, p.2], [2, p.4])를 명시하세요.\n\n"
                    f"[사용자 질문]\n{user_query}\n\n"
                    f"[RRF 상위 검색 문서]\n{joined_context}\n"
                )
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text.strip(), "gemini-2.0-flash"
            except Exception:
                pass

        # 오프라인 또는 API 키 미설정 시: RRF 상위 문서 기반 구조화 종합 답변 자동 생성
        lines = [
            f"질문하신 **\"{user_query}\"**에 대해 5개 다중 확장 쿼리와 하이브리드 검색(FAISS + BM25) 및 "
            f"RRF(Reciprocal Rank Fusion) 재정렬을 통해 도출된 핵심 답변입니다.\n"
        ]
        for doc in reranked_docs[:3]:
            rank = doc["final_rank"]
            src = doc["source"]
            page = doc["page"]
            cid = doc["chunk_id"]
            rrf_val = doc["rrf_score"]
            lines.append(
                f"{rank}. **핵심 근거 요약 [{rank}]**: {doc['snippet']} "
                f"*(출처: {src}, p.{page}, Chunk #{cid} | RRF 점수: {rrf_val})*"
            )

        top_doc = reranked_docs[0]
        lines.append(
            f"\n**종합 결론**: RRF 상호 순위 융합 결과, **{top_doc['source']} (p.{top_doc['page']}, Chunk #{top_doc['chunk_id']})** "
            f"문서가 Dense 유사도(FAISS {top_doc['best_dense_rank']}위)와 키워드 매칭(BM25 {top_doc['best_bm25_rank']}위) "
            f"양쪽에서 가장 높은 연관성(RRF={top_doc['rrf_score']})을 기록하여 최우선 근거로 채택되었습니다 [1]."
        )
        return "\n\n".join(lines), "rag-fusion-extractive-synthesizer"


# ==============================================================================
# 6. FastAPI 애플리케이션 인스턴스 및 전역 에러 핸들러
# ==============================================================================
app = FastAPI(
    title="Advanced RAG Server (FAISS + Multi-Query + Hybrid Search + RAG Fusion)",
    description="FAISS Dense Vector + BM25 Sparse Keyword + 5x Multi-Query + RRF Re-ranking Backend for GAS",
    version="2.0.0"
)

# GAS 및 웹 브라우저 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 싱글톤 RAG 엔진 인스턴스 생성
rag_engine = AdvancedRAGEngine()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    모든 예외를 정확한 에러 코드, 발생 경로, 상세 메시지, 해결 가이드가 포함된
    표준 JSON 구조로 반환합니다.
    """
    status_code = exc.status_code if isinstance(exc, HTTPException) else 500
    detail_msg = exc.detail if isinstance(exc, HTTPException) else str(exc)
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "error": {
                "code": f"ERR_RAG_{status_code}",
                "http_status": status_code,
                "endpoint": str(request.url.path),
                "type": exc.__class__.__name__,
                "message": detail_msg,
                "hint": "입력 파라미터(PDF 인코딩, 질의 문자열, API URL 설정)를 확인하거나 서버 상태를 점검하세요.",
                "trace_summary": traceback.format_exc().splitlines()[-3:]
            }
        }
    )


# ==============================================================================
# 7. API 엔드포인트 정의 (/health, /ingest, /query, /query_stream)
# ==============================================================================
@app.get("/")
@app.get("/health")
async def health_check():
    """
    서버 동작 상태, 현재 적재된 FAISS/BM25 인덱스 현황 및 알고리즘 복잡도 정보를 반환합니다.
    """
    complexity_key = "faiss_hnsw_flat" if rag_engine.index_type == "IndexHNSWFlat" else "faiss_flat_l2"
    return {
        "status": "online",
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine_state": {
            "active_filename": rag_engine.current_filename,
            "indexed_chunks": len(rag_engine.chunks),
            "faiss_index_type": rag_engine.index_type,
            "embedding_dim": rag_engine.embedding_dim,
            "embedding_model": rag_engine.embedding_model_used,
            "last_updated": rag_engine.last_updated
        },
        "complexity": {
            "active_faiss": ALGORITHM_COMPLEXITY_INFO[complexity_key],
            "bm25_sparse": ALGORITHM_COMPLEXITY_INFO["bm25_sparse"],
            "multi_query_hybrid": ALGORITHM_COMPLEXITY_INFO["multi_query_hybrid"],
            "rag_fusion_rrf": ALGORITHM_COMPLEXITY_INFO["rag_fusion_rrf"]
        }
    }


@app.post("/ingest")
async def ingest_document(req: IngestRequest):
    """
    PDF(Base64) 또는 텍스트를 전달받아 FAISS(IndexFlatL2 / IndexHNSWFlat) 및 BM25 인덱스를 생성하고,
    StreamHandler의 단계별 진행 현황(events)을 함께 반환합니다.
    """
    stream = StreamHandler(pipeline_name="RAG_VECTOR_DB_INGEST")
    stream.emit(
        stage="INIT_INGEST",
        progress=10,
        message=f"'{req.filename}' 파일 수신 완료. PDF 텍스트 추출을 시작합니다..."
    )

    pages: List[Tuple[int, str]] = []
    if req.pdf_base64:
        try:
            # Data URL prefix(data:application/pdf;base64,...) 제거 처리
            b64_clean = req.pdf_base64.split(",")[-1].strip()
            pdf_bytes = base64.b64decode(b64_clean)
            pages = rag_engine.extract_pages_from_pdf_bytes(pdf_bytes)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"PDF 파일 디코딩 또는 텍스트 추출 실패: {str(e)}"
            )
    elif req.raw_text and req.raw_text.strip():
        pages = [(1, req.raw_text.strip())]
    else:
        raise HTTPException(
            status_code=400,
            detail="pdf_base64 또는 raw_text 중 하나는 반드시 제공되어야 합니다."
        )

    if not pages:
        raise HTTPException(
            status_code=422,
            detail="PDF에서 추출된 텍스트가 없습니다. 텍스트 레이어가 포함된 PDF 파일인지 확인해 주세요."
        )

    try:
        build_stats = rag_engine.build_index_from_pages(
            pages=pages,
            filename=req.filename,
            index_type=req.index_type,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            api_key=req.api_key,
            stream_handler=stream
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"벡터 DB 인덱스 구축 중 오류 발생: {str(e)}")

    return {
        "status": "success",
        "message": f"'{req.filename}' 기반 FAISS({build_stats['faiss_index_type']}) + BM25 벡터 DB 구축 완료",
        "build_stats": build_stats,
        "stream_handler": stream.get_summary()
    }


@app.post("/query")
async def query_rag_fusion(req: QueryRequest):
    """
    사용자의 질의를 받아 5-Way Multi-Query 확장 -> FAISS + BM25 하이브리드 검색 ->
    RRF 순위 융합 -> LLM 답변 생성을 수행하고, 정제된 상위 문서 스니펫과 메타데이터,
    LLM 답변, 출처, StreamHandler 이벤트를 JSON으로 반환합니다.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="질의 내용(query)이 비어 있습니다.")

    stream = StreamHandler(pipeline_name="ADVANCED_RAG_FUSION_QUERY")
    stream.emit(
        stage="QUERY_RECEIVED",
        progress=5,
        message=f"질의 수신: '{req.query.strip()}' (Top-K={req.top_k}, RRF k={req.rrf_k})"
    )

    result = rag_engine.execute_rag_fusion_pipeline(
        user_query=req.query.strip(),
        top_k=req.top_k,
        dense_weight=req.dense_weight,
        sparse_weight=req.sparse_weight,
        rrf_k=req.rrf_k,
        api_key=req.api_key,
        stream_handler=stream
    )
    result["stream_handler"] = stream.get_summary()
    return result


@app.post("/query_stream")
async def query_rag_fusion_stream(req: QueryRequest):
    """
    StreamHandler를 통해 파이프라인의 각 계층 진행 상태와 최종 RAG Fusion 결과를
    실시간 NDJSON 스트림(application/x-ndjson)으로 전송하는 스트리밍 엔드포인트입니다.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        stream = StreamHandler(pipeline_name="ADVANCED_RAG_FUSION_STREAM")
        init_ev = stream.emit(
            stage="STREAM_START",
            progress=5,
            message=f"실시간 스트리밍 파이프라인 시작: '{req.query.strip()}'"
        )
        yield stream.to_ndjson_line(init_ev)

        result = rag_engine.execute_rag_fusion_pipeline(
            user_query=req.query.strip(),
            top_k=req.top_k,
            dense_weight=req.dense_weight,
            sparse_weight=req.sparse_weight,
            rrf_k=req.rrf_k,
            api_key=req.api_key,
            stream_handler=stream
        )

        # 누적된 중간 단계 이벤트 순차 송출
        for ev in stream.events[1:]:
            yield stream.to_ndjson_line(ev)

        # 최종 결과 페이로드 송출
        final_payload = {
            "pipeline": "ADVANCED_RAG_FUSION_STREAM",
            "stage": "FINAL_RESULT",
            "progress": 100,
            "status": "completed",
            "result": result
        }
        yield stream.to_ndjson_line(final_payload)

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
