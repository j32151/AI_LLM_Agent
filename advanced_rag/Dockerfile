# ==============================================================================
# Dockerfile for Google Cloud Run Deployment
# Project: Advanced RAG (FAISS + Multi-Query + Hybrid Search + RAG Fusion)
# ==============================================================================

FROM python:3.11-slim

# 시스템 패키지 설치 (FAISS 구동에 필요한 OpenMP 런타임 라이브러리 포함)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 파이썬 출력 버퍼링 비활성화 (실시간 로그 및 StreamHandler 스트리밍 지원)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    FAISS_INDEX_TYPE=IndexFlatL2

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 서버 애플리케이션 코드 복사
COPY main.py .

# Cloud Run 기본 포트 노출
EXPOSE 8080

# Uvicorn 서버 실행 (Cloud Run의 $PORT 환경변수 자동 바인딩)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]

