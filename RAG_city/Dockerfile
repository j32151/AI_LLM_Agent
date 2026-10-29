
FROM python:3.11-slim

# Python 실행 환경
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# 라이브러리 설치
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --shell /usr/sbin/nologin appuser

# 메인 서버 코드 복사
COPY main.py ./

# 일반 사용자로 실행
USER appuser

EXPOSE 8080

# Cloud Run 서버 실행
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers"]
