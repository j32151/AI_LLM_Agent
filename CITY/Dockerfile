FROM python:3.11-slim

# Python 로그 출력 설정
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
# Chroma 텔레메트리 끄기
ENV ANONYMIZED_TELEMETRY=False

# 작업 폴더
WORKDIR /app

# chromadb 일부 의존성 빌드용
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# 빌드 기준 폴더가 저장소 루트(Cloud Run GitHub 연결 기본값)여도,
# RAG_city 폴더여도 동작하도록 두 위치 중 있는 쪽에서 파일을 가져옴
COPY . /src
RUN set -e; \
    if [ -f /src/RAG_city/main.py ]; then SRC=/src/RAG_city; else SRC=/src; fi; \
    cp "$SRC/requirements.txt" "$SRC/main.py" /app/; \
    rm -rf /src; \
    pip install --no-cache-dir -r /app/requirements.txt

# Cloud Run 포트
EXPOSE 8080

# FastAPI 서버 실행 (워커 1개: Chroma DB 공유)
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
