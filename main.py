import os
import re
import yaml
import numpy as np
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


def extract_payment_amount(lines):
    AMOUNT_REGEX = re.compile(r"(\d{1,3}(?:,\d{3})+|\d+)원")
    candidates = []
    for line in lines:
        nums = AMOUNT_REGEX.findall(line)
        for n in nums: candidates.append(int(n.replace(",", "")))
    return max(candidates) if candidates else 0

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

# 3. FastAPI 엔드포인트 수정
@app.post("/api/upload")
async def upload_receipts(request: Request, files: List[UploadFile] = File(...), user_key: Optional[str] = Form(None)):
    active_key = user_key if user_key else config['ocr']['default_service_key']
    client_ip = request.client.host # IP만 추출

    # 1. 파일 임시 저장 (I/O)
    file_tasks = []
    for file in files:
        temp_path = os.path.join(UPLOAD_DIR, file.filename)
        content = await file.read()
        with open(temp_path, "wb") as f: f.write(content)
        file_tasks.append((temp_path, file.filename))

    # 2. 병렬 처리 요청 (CPU 작업은 워커 함수로 전달)
    import asyncio
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(executor, worker_process_receipt, task, client_ip, active_key, UPLOAD_DIR, RESULT_DIR, OCR_VIS_DIR)
        for task in file_tasks
    ]
    
    raw_results = await asyncio.gather(*tasks)
    final_data = [r["data"] for r in raw_results]
    
    return {"status": "success", "data": final_data}

# 전체 다운로드 (Zip) 기능 추가
from fastapi.responses import StreamingResponse
@app.get("/api/download_all/{type}")
async def download_all(type: str):
    # 다운로드할 폴더 결정
    folder = RESULT_DIR if type == "origin" else OCR_VIS_DIR
    zip_filename = f"receipts_{type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    
    # 메모리 버퍼 생성
    memory_file = BytesIO()
    
    # Zip 파일 생성
    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    
                    # 폴더가 아닌 '파일'인 경우에만 Zip에 추가
                    if os.path.isfile(file_path):
                        zf.write(file_path, filename)
    
    # 버퍼의 포인터를 처음으로 이동
    memory_file.seek(0)
    
    # StreamingResponse를 통해 전달
    return StreamingResponse(
        memory_file,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
    )

@app.get("/")
async def read_index(): return FileResponse('static/index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config['server']['host'], port=config['server']['port'])