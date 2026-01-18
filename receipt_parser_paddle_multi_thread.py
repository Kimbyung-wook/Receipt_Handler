import os
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["GLOG_minloglevel"] = "3"
os.environ["PADDLE_LOG_LEVEL"] = "ERROR"
import re
import cv2
import csv
import shutil
import requests
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ProcessPoolExecutor, as_completed # 병렬처리
from pdf2image import convert_from_path

import pprint

# ===============================
# 0. 환경 설정 (중요)
# ===============================

# 속도 개선
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# INPUT_DIR = "./test1"
# OUTPUT_DIR = "./renamed_1"
# OCR_RESULT_DIR = "./ocr_result_1"

INPUT_DIR = "./test_images"
OUTPUT_DIR = "./renamed"
OCR_RESULT_DIR = "./ocr_result"
CSV_PATH = "./receipt_result.csv"

SERVICE_KEY = f"0WTiyd8+EajIBrN1jHRNSo+gjYGCWi29o2ccl51EH6Fy1lFX7yCkx1XvtM8L+cWj8SE6bGymOFRRuDUhcj/kdw=="

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OCR_RESULT_DIR, exist_ok=True)

# ===============================
# 1. OCR 엔진 초기화
# ===============================
# ocr_engine = PaddleOCR(
#     lang="korean",
#     use_doc_orientation_classify=False,
#     use_textline_orientation=False,
#     # use_angle_cls=False,
#     use_doc_unwarping=False,
#     show_log=False,
# )

import platform
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
    
def draw_bb_on_img(img_arr, result):
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

    return image_pil

# OCR 결과를 줄 대로 받는다.
def get_ocr_lines(result):
    if isinstance(result, dict) and "rec_texts" in result:
        return [t.strip() for t in result["rec_texts"] if t.strip()]
    return []

# ===============================
# 2. OCR 결과 텍스트 추출
# ===============================

def pdf_to_images(pdf_path, dpi=300):
    """
    PDF 파일을 이미지 리스트(PIL Image)로 변환
    """
    images = convert_from_path(pdf_path, dpi=dpi)
    return images

def extract_text_from_paddle(result):
    """
    PaddleOCR (최신 PaddleX pipeline) 전용
    """
    # rec_texts 라는 key 으로 찾아온다.
    # 대부분 여기서 빠진다.
    if isinstance(result, dict) and "rec_texts" in result:
        return "\n".join(result["rec_texts"])

    # 혹시 모를 fallback
    texts = []

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "rec_texts" and isinstance(v, str):
                    texts.append(v)
                else:
                    walk(v)
        elif isinstance(obj, list):
            for i in obj:
                walk(i)

    walk(result)
    return "\n".join(texts)

def resize_for_ocr(img, max_width=1000):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def ocr_image(image_path):
    img_ori = Image.open(image_path).convert("RGB")
    img_arr = np.array(img_ori)
    img_arr = resize_for_ocr(img_arr)
    return img_arr

    # ocr_engine = PaddleOCR(
    #     lang="korean",
    #     use_doc_orientation_classify=False,
    #     use_textline_orientation=False,
    #     # use_angle_cls=False,
    #     use_doc_unwarping=False,
    # )
    # result = ocr_engine.ocr(img_arr)
    # result = result[0]

    # return result

def ocr_image_from_pdf(pdf_path):
    pages = convert_from_path(pdf_path, dpi=300)
    img_pil = pages[0] if pages else None
    if img_pil is None:
        raise ValueError("PDF 변환 실패")
    img_arr = np.array(img_pil)
    img_arr = resize_for_ocr(img_arr)
    return img_arr

    # ocr_engine = PaddleOCR(
    #     lang="korean",
    #     use_doc_orientation_classify=False,
    #     use_textline_orientation=False,
    #     # use_angle_cls=False,
    #     use_doc_unwarping=False,
    # )
    # result = ocr_engine.ocr(img_arr)
    # result = result[0]

    # return result, img_arr

# ===============================
# 3. 정보 추출 함수들
# ===============================
def extract_biz_number(text):
    m = re.search(r'\d{3}-\d{2}-\d{5}', text)
    return m.group() if m else None

def extract_payment_date_without_keyword(text):
    patterns = [
        r'(20\d{2})[./-](\d{1,2})[./-](\d{1,2})',
          r'(\d{2})[./-](\d{1,2})[./-](\d{1,2})'
    ]
    # print("text at extract_payment_date : ", text)
    for p in patterns:
        m = re.search(p, text)
        if m:
            yy = 0; mm = 0; dd = 0
            if (len(m.group(1)) == 4):
                yy = m.group(1)[2:]
                mm = m.group(2).zfill(2)
                dd = m.group(3).zfill(2)
                # if(yy < 2000): # Exception
                #     continue
            elif (len(m.group(1)) == 2):
                yy, mm, dd = m.groups()
            else:
                continue
            return f"{yy}{mm}{dd}"

    return "UNKNOWN"

def extract_payment_date_with_keyword(lines):
    
    DATE_PATTERN = re.compile(r"(20\d{2})[./-](\d{2})[./-](\d{2})")

    for i, line in enumerate(lines):
        if "거래일시" in line:
            if i + 1 < len(lines):
                # print("거래일시 -> ", lines[i + 1])
                m = re.search(DATE_PATTERN, lines[i + 1])
                if not m:
                    return "UNKNOWN"
                yy = m.group(1)[2:]
                mm = m.group(2).zfill(2)
                dd = m.group(3).zfill(2)
                # print(f"{yy}{mm}{dd}")
                return f"{yy}{mm}{dd}"

def clean_merchant_name(name):
    # 불필요한 단어 제거
    blacklist = [
        "사업자등록번호", "대표자", "전화", "주소",
        "카드", "승인", "금액", "합계"
    ]

    for b in blacklist:
        name = name.replace(b, "")

    # 특수문자 제거
    name = re.sub(r"[^\w가-힣\s]", "", name)

    return name.strip()

def extract_merchant_name(lines):
    """
    OCR 줄 리스트에서 가맹점명 추출
    """
    for i, line in enumerate(lines):
        # 1. 같은 줄에 있는 경우
        if "가맹점명" in line:
            # 예: 가맹점명: 나주곰탕
            same_line = re.sub(r".*가맹점명[:\s]*", "", line).strip()
            # print("가맹점명 키워드 찾음 @ extract_merchant_name")
            if same_line:
                # print("같은 줄에 있음 @ extract_merchant_name")
                return clean_merchant_name(same_line)

            # 2. 다음 줄에 있는 경우
            if i + 1 < len(lines):
                # print("다음 줄에 있음 @ extract_merchant_name")
                return clean_merchant_name(lines[i + 1])
            
        # 1. 같은 줄에 있는 경우
        if "가맹점정보" in line:
            # 예: 가맹점정보: 나주곰탕
            same_line = re.sub(r".*가맹점정보[:\s]*", "", line).strip()
            # print("가맹점정보 키워드 찾음 @ extract_merchant_name")
            if same_line:
                # print("같은 줄에 있음 @ extract_merchant_name")
                return clean_merchant_name(same_line)

            # 2. 다음 줄에 있는 경우
            if i + 1 < len(lines):
                # print("다음 줄에 있음 @ extract_merchant_name")
                return clean_merchant_name(lines[i + 1])

    return "UNKNOWN"

def normalize_amount(text):
    """
    '12,000원' → 12000
    """
    return int(text.replace(",", ""))

AMOUNT_REGEX = re.compile(r"(\d{1,3}(?:,\d{3})+|\d+)원")
IGNORE_KEYWORDS = ["부가세", "봉사료", "면세"]
AMOUNT_KEYWORDS = [
    "결제금액", "거래금액", "합계", "총액", "청구금액", "승인금액"
]

def extract_payment_amount(lines):
    """
    OCR 줄 리스트에서 결제금액 추출
    """
    candidates = []

    for i, line in enumerate(lines):
        # 1. 무시 키워드 넘기기
        if any(x in line for x in IGNORE_KEYWORDS):
            continue

        # 2. 키워드 포함 라인
        if any(k in line for k in AMOUNT_KEYWORDS):
            # 같은 줄에서 숫자
            nums = AMOUNT_REGEX.findall(line)
            if nums:
                return normalize_amount(nums[-1])

            # 다음 줄에서 숫자
            if i + 1 < len(lines):
                nums = AMOUNT_REGEX.findall(lines[i + 1])
                if nums:
                    return normalize_amount(nums[-1])

        # 3. 모든 금액 후보 수집 (fallback용)
        nums = AMOUNT_REGEX.findall(line)
        for n in nums:
            candidates.append(normalize_amount(n))

    # 3. fallback: 가장 큰 금액
    if candidates:
        return max(candidates)

    return None

# ===============================
# 4. 국세청 과세유형 조회
# ===============================
def get_tax_type_from_nts(biz_no, service_key):
    if not biz_no:
        print("Biz_no is not exist")
        return "오류"

    url = "https://api.odcloud.kr/api/nts-businessman/v1/status"
    payload = {"b_no": [biz_no.replace("-", "")]}
    headers = {"Content-Type" : "application/json",
               "accept" : "application/json"}

    params = {"serviceKey": service_key}

    for i in range(3):
        # if True:
        try:
            r = requests.post(url, json=payload, headers=headers, params=params, timeout=10)
            data = r.json()

            info = data["data"][0]
            return info.get("tax_type", "UNKNOWN")
    
        except Exception as e:
            print("Failed to get TaxType from Biz_no : ", biz_no)
            print(e)
            return "오류"
        


def normalize_tax_type(tax_type):
    
    if tax_type is None:
        return "오류"
    if "일반" in tax_type:
        return "일반"
    if "간이" in tax_type:
        return "간이"
    if "면세" in tax_type:
        return "면세"
    return "오류"

# ===============================
# 5. 파일명 정리
# ===============================
def sanitize_filename(text):
    text = re.sub(r'[\\/:*?"<>|]', "", text)
    return text.strip()[:30]

def copy_and_rename(src, date, tax_type, merchant, payment_amount):
    # ext = os.path.splitext(src)[1]
    # new_name = f"{date}_{normalize_tax_type(tax_type)}_{sanitize_filename(merchant)}_{payment_amount}{ext}}"
    new_name = f"{date}_{normalize_tax_type(tax_type)}_{payment_amount}_{sanitize_filename(merchant)}.png"
    dst = os.path.join(OUTPUT_DIR, new_name)
    shutil.copy2(src, dst)
    return new_name

def get_img_arr_from_file_name(file_full_path):
    ext = os.path.splitext(file_full_path)[1].lower()
    if (ext.lower() == ".pdf"):
        pages = convert_from_path(file_full_path, dpi=300)
        img_pil = pages[0] if pages else None
        if img_pil is None:
            raise ValueError("PDF 변환 실패")
        img_arr = np.array(img_pil)

    elif (ext.lower() == ".jpg"  or
        ext.lower() == ".png"  or
        ext.lower() == ".jpeg"   ):
        img_ori = Image.open(file_full_path).convert("RGB")
        img_arr = np.array(img_ori)
    else:
        raise ValueError("확장자 오류")
    
    img_arr = resize_for_ocr(img_arr)
    return img_arr

def process_image(path):
    print("▶ process_file start:", path)
    ext = os.path.splitext(path)[1].lower()
    if True:
    # try:
        if (ext.lower() == ".pdf"):
            img_arr = ocr_image_from_pdf(path)

        elif (ext.lower() == ".jpg"  or
              ext.lower() == ".png"  or
              ext.lower() == ".jpeg"   ):
            img_arr = ocr_image(path)
        else:
            raise ValueError("확장자 오류")
        
        ocr_engine = PaddleOCR(
            lang="korean",
            use_doc_orientation_classify=False,
            use_textline_orientation=False,
            # use_angle_cls=False,
            use_doc_unwarping=False,
        )
        result = ocr_engine.ocr(img_arr)
        result = result[0]

        text = extract_text_from_paddle(result)
        textline = get_ocr_lines(result)

        biz_no = extract_biz_number(text)
        merchant = extract_merchant_name(textline)
        if "거래일시" in textline:
            pay_date = extract_payment_date_with_keyword(textline)
        else:
            pay_date = extract_payment_date_without_keyword(text)
        payment_amount = extract_payment_amount(textline)

        tax_type = get_tax_type_from_nts(biz_no, SERVICE_KEY)
        print(pay_date, "/", biz_no, "/", merchant, "/", tax_type, "/", payment_amount)

        # BB 이미지 저장
        new_file = copy_and_rename(
            path,
            pay_date,
            tax_type,
            merchant,
            payment_amount
        )

        image_pil = draw_bb_on_img(img_arr, result)
        image_pil.save(os.path.join(OCR_RESULT_DIR, new_file))

        return {
            "path": path,
            "fname": os.path.basename(path),
            "merchant": merchant,
            "biz_no": biz_no,
            "pay_date": pay_date,
            "payment_amount": payment_amount,
            "tax_type": tax_type,
        }


# ===============================
# 6. 메인 처리
# ===============================
def main():
    file_paths = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".pdf"))
    ]

    results = []

    # 🔥 병렬 OCR & 파싱
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_image, p) for p in file_paths]

        for future in as_completed(futures):
            results.append(future.result())

    rows = []

    # 결제일에 맞추어 정렬
    results

    # 🔽 여기부터는 단일 스레드
    print("[완료된 리스트 - 시작]")
    for r in results:
        if "error" in r:
            print("❌ 오류:", r["fname"], r["error"])
            continue

        new_file = copy_and_rename(
            r["path"],
            r["pay_date"],
            r["tax_type"],
            r["merchant"],
            r["payment_amount"]
        )

        rows.append([
            r["fname"],
            r["merchant"],
            r["biz_no"],
            r["pay_date"],
            r["payment_amount"],
            r["tax_type"],
            new_file
        ])

        print(
            r["pay_date"], "/", r["biz_no"], "/", r["merchant"],
            "/", r["tax_type"], "/", r["payment_amount"]
        )
    print("[완료된 리스트 - 종료]")

    # CSV 저장
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "original_file",
            "merchant_name",
            "business_number",
            "payment_date",
            "payment_amount",
            "tax_type",
            "renamed_file"
        ])
        writer.writerows(rows)

    print("✅ 병렬 처리 완료")

if __name__ == "__main__":
    main()
