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

# 필요한 Python 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 서버 파일 복사
COPY main.py .

# Cloud Run 포트
EXPOSE 8080

# FastAPI 서버 실행 (워커 1개: Chroma 인메모리 DB 공유)
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
