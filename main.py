import os
import re
import yaml
import numpy as np
import shutil
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import zipfile
from io import BytesIO

from worker import worker_process_receipt
from user_log import get_usage_data

# Concurrent Processing
from concurrent.futures import ProcessPoolExecutor
import functools

# --- 1. 환경 및 설정 로드 ---
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

UPLOAD_DIR = config['ocr']['upload_dir']
RESULT_DIR = config['ocr']['result_dir'] # 이름 변경된 원본 저장
OCR_VIS_DIR = os.path.join(RESULT_DIR, "vis") # OCR 결과 이미지 저장

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(OCR_VIS_DIR, exist_ok=True)

# --- 4. 좌표 오류 해결된 이미지 그리기 ---

# 1. 글로벌 프로세스 풀 생성 (CPU 코어 수에 맞춰 설정)
# 서버 시작 시 한 번만 생성됩니다.
executor = ProcessPoolExecutor(max_workers=os.cpu_count() // 2)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/ocr_result", StaticFiles(directory=RESULT_DIR), name="ocr_result")

# --- 5. API 엔드포인트 ---
@app.get("/api/usage")
async def get_today_usage():
    print(datetime.today())
    print(datetime.now())
    today = datetime.now().strftime("%Y-%m-%d")
    data = get_usage_data().get(today, {"total": 0})
    return {"total": data["total"]}

# 현재 접속 IP 확인 API
@app.get("/api/my_ip")
async def get_my_ip(request: Request):
    return {"ip": request.client.host}

# FastAPI 엔드포인트 수정
@app.post("/api/upload")
async def upload_receipts(request: Request, files: List[UploadFile] = File(...), user_key: Optional[str] = Form(None)):
    client_ip = request.client.host
    # 유저별 격리된 경로 설정
    u_dir = get_user_path(UPLOAD_DIR, request)
    r_dir = get_user_path(RESULT_DIR, request)
    v_dir = get_user_path(OCR_VIS_DIR, request)
    
    active_key = user_key if user_key else config['ocr']['default_service_key']

    file_tasks = []
    for file in files:
        temp_path = os.path.join(u_dir, file.filename)
        content = await file.read()
        with open(temp_path, "wb") as f: f.write(content)
        file_tasks.append((temp_path, file.filename))

    import asyncio
    loop = asyncio.get_event_loop()
    # 워커에 격리된 r_dir, v_dir 전달 (worker.py의 인자 순서 유지)
    tasks = [
        loop.run_in_executor(executor,
                             worker_process_receipt,
                             task,
                             client_ip,
                             active_key,
                             u_dir, r_dir, v_dir)
        for task in file_tasks
    ]
    
    raw_results = await asyncio.gather(*tasks)
    # status가 success인 데이터만 필터링
    final_data = [r["data"] for r in raw_results]
    
    return {"status": "success", "data": final_data}

# 전체 다운로드 (Zip) 기능 추가
# Zip 다운로드 시에도 유저 폴더만 압축하도록 수정
from fastapi.responses import StreamingResponse
@app.get("/api/download_all/{type}")
async def download_all(type: str, request: Request):
    client_ip = request.client.host.replace(":", "_")
    base_folder = RESULT_DIR if type == "origin" else OCR_VIS_DIR
    user_folder = os.path.join(base_folder, client_ip) # 유저 폴더 타겟팅
    
    if not os.path.exists(user_folder):
        return JSONResponse({"status": "error", "message": "No files found"}, status_code=404)

    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in os.listdir(user_folder):
            file_path = os.path.join(user_folder, filename)
            if os.path.isfile(file_path):
                zf.write(file_path, filename)
    memory_file.seek(0)
    return StreamingResponse(memory_file, media_type="application/zip", 
                             headers={"Content-Disposition": f"attachment; filename=receipts_{client_ip}.zip"})

# 헬퍼 함수: IP별 독립 경로 생성
def get_user_path(base_dir, request: Request):
    client_ip = request.client.host.replace(":", "_") # IPv6 대응
    path = os.path.join(base_dir, client_ip)
    os.makedirs(path, exist_ok=True)
    return path

# 접속(페이지 로드) 시 기존 파일 삭제
@app.get("/")
async def read_index(request: Request):
    client_ip = request.client.host.replace(":", "_")
    # 해당 유저의 업로드/결과 폴더가 있다면 삭제 후 재생성
    for base in [UPLOAD_DIR, RESULT_DIR, OCR_VIS_DIR]:
        user_path = os.path.join(base, client_ip)
        if os.path.exists(user_path):
            shutil.rmtree(user_path)
        os.makedirs(user_path, exist_ok=True)
    return FileResponse('static/index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config['server']['host'], port=config['server']['port'])