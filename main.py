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
import zipfile
from io import BytesIO
import platform

# 작성하신 파서 파일에서 함수 임포트
from receipt_parser_paddle_multi_thread import (
    extract_text_from_paddle, get_ocr_lines, extract_biz_number, 
    extract_merchant_name, extract_payment_date_with_keyword, 
    extract_payment_date_without_keyword, extract_payment_amount, 
    resize_for_ocr, get_tax_type_from_nts, normalize_tax_type
)

# --- 1. 환경 및 설정 로드 ---
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

app = FastAPI()
USAGE_LOG = "usage_log.json"
UPLOAD_DIR = config['ocr']['upload_dir']
RESULT_DIR = config['ocr']['result_dir'] # 이름 변경된 원본 저장
OCR_VIS_DIR = os.path.join(RESULT_DIR, "vis") # OCR 결과 이미지 저장

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(OCR_VIS_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/ocr_result", StaticFiles(directory=RESULT_DIR), name="ocr_result")

# OCR 엔진 초기화 (기존 설정 유지)
ocr_engine = PaddleOCR(
    lang="korean",
    use_doc_orientation_classify=False,
    use_textline_orientation=False,
    # use_angle_cls=False,
    use_doc_unwarping=False,
)

# --- 2. 호출 횟수 관리 (IP별 로그) ---
def get_usage_data():
    if not os.path.exists(USAGE_LOG): return {}
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

# --- 3. 기존 소스코드의 정교한 추출 함수들 ---
def extract_biz_number(text):
    m = re.search(r'\d{3}-\d{2}-\d{5}', text)
    return m.group() if m else None

def extract_payment_date(text, lines):
    # 키워드 기반 우선 (거래일시)
    DATE_PATTERN = re.compile(r"(20\d{2})[./-](\d{1,2})[./-](\d{1,2})")
    for i, line in enumerate(lines):
        if "거래일시" in line and i + 1 < len(lines):
            m = DATE_PATTERN.search(lines[i+1])
            if m: return f"{m.group(1)[2:]}{m.group(2).zfill(2)}{m.group(3).zfill(2)}"
    # 전체 텍스트 검색
    m = DATE_PATTERN.search(text)
    if m: return f"{m.group(1)[2:]}{m.group(2).zfill(2)}{m.group(3).zfill(2)}"
    return datetime.now().strftime("%y%m%d") # 기본값

def extract_merchant_name(lines):
    for i, line in enumerate(lines):
        if "가맹점명" in line:
            same_line = re.sub(r".*가맹점명[:\s]*", "", line).strip()
            if same_line: return re.sub(r"[^\w가-힣\s]", "", same_line)
            if i + 1 < len(lines): return re.sub(r"[^\w가-힣\s]", "", lines[i+1])
    return "가맹점미상"

def extract_payment_amount(lines):
    AMOUNT_REGEX = re.compile(r"(\d{1,3}(?:,\d{3})+|\d+)원")
    candidates = []
    for line in lines:
        nums = AMOUNT_REGEX.findall(line)
        for n in nums: candidates.append(int(n.replace(",", "")))
    return max(candidates) if candidates else 0

# --- 4. 좌표 오류 해결된 이미지 그리기 ---
def get_system_font(font_size=20):
    os_name = platform.system()
    
    # OS별 기본 폰트 경로 후보
    if os_name == "Windows":
        # 윈도우: 맑은 고딕
        font_path = "C:/Windows/Fonts/malgun.ttf"
    elif os_name == "Linux":
        # 리눅스(Ubuntu 등): 나눔고딕 또는 백묵 폰트
        # 경로 예시: /usr/share/fonts/truetype/nanum/NanumGothic.ttf
        candidates = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf" # 최후의 수단
        ]
        font_path = next((p for p in candidates if os.path.exists(p)), None)
    elif os_name == "Darwin":
        # macOS: 애플 고딕
        font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
    else:
        font_path = None

    # 폰트 로드 시도
    try:
        if font_path and os.path.exists(font_path):
            return ImageFont.truetype(font_path, font_size)
        else:
            # 폰트 파일이 없으면 기본 폰트 반환
            return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()

def draw_bb_on_img(img_arr, result, save_path):
    image_pil = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(image_pil)

    font = get_system_font(20)

    # Bounding Box 및 텍스트 표시
    for i in range(np.shape(result['rec_boxes'])[0]):
        bbox  = result['dt_polys'][i]
        text  = result['rec_texts'][i]
        score = result['rec_scores'][i]

        # OpenCV로 Bounding Box 그리기
        draw.polygon([tuple(point) for point in bbox], outline="red", width=3)

        # PIL을 사용한 한글 텍스트 출력
        x, y = bbox[0]
        draw.text((x, y - 10), text, font=font, fill=(0, 255, 0))  # 초록색 텍스트

    image_pil.save(save_path)

# --- 5. API 엔드포인트 ---
@app.get("/api/usage")
async def get_today_usage():
    today = datetime.now().strftime("%Y-%m-%d")
    data = get_usage_data().get(today, {"total": 0})
    return {"total": data["total"]}

@app.post("/api/upload")
async def upload_receipts(request: Request, files: List[UploadFile] = File(...), user_key: Optional[str] = Form(None)):
    client_ip = request.client.host
    active_key = user_key if user_key and user_key.strip() != "" else config['ocr']['default_service_key']
    
    results = []
    # main.py 내 /api/upload 엔드포인트 수정 제안
    for file in files:
        # 안전한 파일명 사용 (한글 포함 시 오류 방지)
        safe_filename = file.filename.replace(" ", "_") 
        temp_path = os.path.join(UPLOAD_DIR, safe_filename)
        
        with open(temp_path, "wb") as buffer:
            content = await file.read() # 비동기 파일 읽기 권장
            buffer.write(content)

        # try:
        if True:
            ext = os.path.splitext(temp_path)[1].lower()
            if (ext.lower() == ".pdf"):
                pages = convert_from_path(temp_path, dpi=300)
                img_pil = pages[0] if pages else None
                if img_pil is None:
                    raise ValueError("PDF 변환 실패")
                img_arr = np.array(img_pil)
                img_arr = resize_for_ocr(img_arr)

            elif (ext.lower() == ".jpg"  or
                ext.lower() == ".png"  or
                ext.lower() == ".jpeg"   ):
                img_ori = Image.open(temp_path).convert("RGB")
                img_arr = np.array(img_ori)
            else:
                raise ValueError("확장자 오류")

            ocr_res = ocr_engine.ocr(img_arr)
            ocr_res = ocr_res[0]
            
            text = extract_text_from_paddle(ocr_res)
            textline = get_ocr_lines(ocr_res)

            biz_no = extract_biz_number(text)
            merchant = extract_merchant_name(textline)
            if "거래일시" in textline:
                pay_date = extract_payment_date_with_keyword(textline)
            else:
                pay_date = extract_payment_date_without_keyword(text)

            biz_no = extract_biz_number(text)
            merchant = extract_merchant_name(textline)
            amount = extract_payment_amount(textline)
            
            # API 호출 및 로그 기록
            tax_type = "미확인"
            if biz_no and active_key:
                log_api_call(client_ip)
                tax_type = get_tax_type_from_nts(biz_no, active_key)
                tax_type = normalize_tax_type(tax_type)

            # 파일명 변경 규칙 적용
            # [YYMMDD]_[TaxType]_[Amount]_[Merchant]
            renamed_name     = f"{pay_date}_{tax_type}_{amount}_{merchant}.png"
            visualized_name  = f"{pay_date}_{tax_type}_{amount}_{merchant}_vis.png"
            
            # Old
            # save_path = os.path.join(RESULT_DIR, renamed_name)
            # draw_bb_on_img(img_arr, ocr_res, save_path)

            # New
            # (A) 이름만 바뀐 원본 이미지 저장
            final_origin_path = os.path.join(RESULT_DIR, renamed_name)
            Image.fromarray(img_arr).save(final_origin_path)
            
            # (B) OCR 결과(BB)가 포함된 이미지 별도 저장
            draw_bb_on_img(img_arr, ocr_res, os.path.join(OCR_VIS_DIR, visualized_name))

            results.append({
                "original_name": file.filename,
                "renamed_name": renamed_name,
                "vis_name": visualized_name,
                "merchant": merchant,
                "biz_no": biz_no,
                "pay_date": pay_date,
                "amount": amount,
                "tax_type": tax_type
            })
        # except Exception as e:
        #    print(f"Error processing {file.filename}: {e}")

    return {"status": "success", "data": results}

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