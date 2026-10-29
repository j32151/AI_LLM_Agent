
FROM python:3.11-slim

# Python 로그 출력 설정
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# 작업 폴더
WORKDIR /app

# 필요한 Python 라이브러리 설치
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 서버 파일 복사
COPY main.py .

# Cloud Run 포트
EXPOSE 8080

# FastAPI 서버 실행
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
