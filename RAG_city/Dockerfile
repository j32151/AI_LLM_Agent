# Python 3.10 slim 이미지 사용
FROM python:3.10-slim
# 작업 디렉토리 설정
WORKDIR /app
# 시스템 패키지 업데이트 및 필요한 유틸리티 설치
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*
# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# 애플리케이션 소스 복사
COPY . .
# Cloud Run에서 사용할 포트 노출
EXPOSE 8080
# uvicorn으로 FastAPI 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
