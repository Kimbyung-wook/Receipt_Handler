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

# 작성하신 파서 파일에서 함수 임포트
from receipt_parser_paddle_multi_thread import (
    extract_text_from_paddle, get_ocr_lines, extract_biz_number, 
    extract_merchant_name, extract_payment_date_with_keyword, 
    extract_payment_date_without_keyword, extract_payment_amount, 
    resize_for_ocr, get_tax_type_from_nts
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

# def get_tax_type_from_nts(biz_no, service_key):
#     if not biz_no or not service_key: return "미확인"
#     url = "https://api.odcloud.kr/api/nts-businessman/v1/status"
#     try:
#         r = requests.post(url, json={"b_no": [biz_no.replace("-", "")]}, 
#                           params={"serviceKey": service_key}, timeout=5)
#         res = r.json()["data"][0].get("tax_type", "미확인")
#         return "일반" if "일반" in res else "간이" if "간이" in res else "면세" if "면세" in res else "미확인"
#     except: return "미확인"

# --- 4. 좌표 오류 해결된 이미지 그리기 ---
def draw_bb_on_img(img_arr, result, save_path):
    image_pil = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(image_pil)

    try: font = ImageFont.truetype("malgun.ttf", 15)
    except: font = ImageFont.load_default()

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
            pay_date = extract_payment_date(text, textline)
            merchant = extract_merchant_name(textline)
            amount = extract_payment_amount(textline)
            
            # # 이미지 변환 로직
            # if file.filename.lower().endswith(".pdf"):
            #     pages = convert_from_path(temp_path, dpi=200)
            #     img_arr = np.array(pages[0])
            # else:
            #     img_arr = cv2.imdecode(np.fromfile(temp_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            #     img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

            # ocr_res = ocr_engine.ocr(img_arr)
            # lines = [line[1][0] for line in ocr_res[0]] if ocr_res[0] else []
            # all_text = "\n".join(lines)
            # print(lines)
            # print(all_text)
            
            # # 정보 추출
            # biz_no = extract_biz_number(all_text)
            # pay_date = extract_payment_date(all_text, lines)
            # merchant = extract_merchant_name(lines)
            # amount = extract_payment_amount(lines)
            
            # API 호출 및 로그 기록
            tax_type = "미확인"
            if biz_no and active_key:
                log_api_call(client_ip)
                tax_type = get_tax_type_from_nts(biz_no, active_key)

            # 파일명 변경 규칙 적용
            # [YYMMDD]_[TaxType]_[Amount]_[Merchant]
            renamed_name = f"{pay_date}_{tax_type}_{amount}_{merchant}.jpg"
            save_path = os.path.join(RESULT_DIR, renamed_name)
            
            draw_bb_on_img(img_arr, ocr_res, save_path)

            results.append({
                "original_name": file.filename,
                "renamed_name": renamed_name,
                "merchant": merchant,
                "biz_no": biz_no,
                "pay_date": pay_date,
                "amount": amount,
                "tax_type": tax_type
            })
        # except Exception as e:
        #     print(f"Error: {e}")
        #     continue

    return {"status": "success", "data": results}

@app.get("/")
async def read_index(): return FileResponse('static/index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config['server']['host'], port=config['server']['port'])