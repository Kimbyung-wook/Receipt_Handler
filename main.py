import os
import re
import cv2
import yaml
import json
import shutil
import requests
import numpy as np
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path

# --- 1. 환경 및 설정 로드 ---
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

app = FastAPI()
USAGE_LOG = "usage_log.json"

# 디렉토리 설정
UPLOAD_DIR = config['ocr']['upload_dir']
RESULT_DIR = config['ocr']['result_dir']
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/ocr_result", StaticFiles(directory=RESULT_DIR), name="ocr_result")

ocr_engine = PaddleOCR(lang="korean", use_angle_cls=True, show_log=False)

# --- 2. 호출 횟수 관리 함수 ---
def get_usage_data():
    if not os.path.exists(USAGE_LOG):
        return {}
    with open(USAGE_LOG, "r", encoding="utf-8") as f:
        return json.load(f)

def log_api_call(ip):
    today = datetime.now().strftime("%Y-%m-%d")
    data = get_usage_data()
    
    if today not in data:
        data[today] = {"total": 0, "ips": {}}
    
    data[today]["total"] += 1
    data[today]["ips"][ip] = data[today]["ips"].get(ip, 0) + 1
    
    with open(USAGE_LOG, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return data[today]["total"]

# --- 3. 좌표 오류 해결된 이미지 그리기 함수 ---
def draw_bb_on_img(img_arr, result, save_path):
    image_pil = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(image_pil)
    try: font = ImageFont.truetype("malgun.ttf", 15)
    except: font = ImageFont.load_default()

    if result and result[0]:
        for line in result[0]:
            # 좌표를 정수형 리스트의 튜플로 변환하여 오류 방지
            bbox = [tuple(map(int, p)) for p in line[0]]
            text = line[1][0]
            if len(bbox) >= 3:
                draw.polygon(bbox, outline="red", width=3)
                draw.text((bbox[0][0], bbox[0][1] - 15), text, font=font, fill=(0, 255, 0))
    image_pil.save(save_path)

# --- 4. API 엔드포인트 ---

@app.get("/api/usage")
async def get_today_usage():
    today = datetime.now().strftime("%Y-%m-%d")
    data = get_usage_data().get(today, {"total": 0})
    return {"total": data["total"]}

@app.post("/api/upload")
async def upload_receipts(
    request: Request,
    files: List[UploadFile] = File(...),
    user_key: Optional[str] = Form(None)
):
    client_ip = request.client.host
    active_key = user_key if user_key and user_key.strip() != "" else config['ocr']['default_service_key']
    
    results = []
    for file in files:
        temp_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            # 이미지 변환
            if file.filename.lower().endswith(".pdf"):
                pages = convert_from_path(temp_path, dpi=200)
                img_arr = np.array(pages[0])
            else:
                img_arr = cv2.imdecode(np.fromfile(temp_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

            ocr_res = ocr_engine.ocr(img_arr, cls=True)
            all_text = "\n".join([line[1][0] for line in ocr_res[0]]) if ocr_res[0] else ""
            
            # 사업자번호가 있을 때만 국세청 API 호출 및 카운트
            biz_no = re.search(r'\d{3}-\d{2}-\d{5}', all_text)
            tax_type = "미확인"
            if biz_no and active_key:
                log_api_call(client_ip) # API 호출 기록
                # (기존 국세청 API 호출 로직 위치...)
                tax_type = "일반" # 예시 결과

            renamed_name = f"{datetime.now().strftime('%y%m%d')}_{tax_type}_{file.filename}"
            save_path = os.path.join(RESULT_DIR, renamed_name)
            draw_bb_on_img(img_arr, ocr_res, save_path)

            results.append({
                "original_name": file.filename,
                "renamed_name": renamed_name,
                "tax_type": tax_type,
                "ip": client_ip
            })
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            continue

    return {"status": "success", "data": results}

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

if __name__ == "__main__":
    uvicorn.run(app, host=config['server']['host'], port=config['server']['port'])