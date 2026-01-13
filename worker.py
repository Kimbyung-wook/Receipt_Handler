import os
from pdf2image import convert_from_path
# 작성하신 파서 파일에서 함수 임포트
from receipt_parser_paddle_multi_thread import (
    extract_text_from_paddle, get_ocr_lines, extract_biz_number, 
    extract_merchant_name, extract_payment_date_with_keyword, 
    extract_payment_date_without_keyword, extract_payment_amount, 
    resize_for_ocr, normalize_tax_type,
    draw_bb_on_img, get_img_arr_from_file_name
)



# ===============================
# 4. 국세청 과세유형 조회
# ===============================
import requests
from user_log import log_api_call
def get_tax_type_from_nts_with_api_call_counter(client_ip, biz_no, service_key):
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
            log_api_call(client_ip)
            return info.get("tax_type", "UNKNOWN")
    
        except Exception as e:
            print("Failed to get TaxType from Biz_no : ", biz_no)
            print("getTaxType : ", client_ip, biz_no)
            print(e)
            return "오류"

# 2. 개별 파일을 처리할 독립적인 워커 함수
# 이 함수는 별도의 프로세스에서 실행되므로 전역 변수에 접근이 어렵습니다.
def worker_process_receipt(file_info, client_ip, active_key, upload_dir, result_dir, ocr_vis_dir):
    from paddleocr import PaddleOCR
    import numpy as np
    from PIL import Image
    results = []

    # 각 프로세스 내에서 OCR 엔진을 처음 한 번만 초기화 (Lazy Initialization)
    # 함수 밖에 있으면 직렬화 에러가 날 수 있으므로 내부에서 로드하는 것이 안전합니다.
    global local_ocr
    if 'local_ocr' not in globals():
        local_ocr = PaddleOCR(
            lang="korean",
            use_doc_orientation_classify=False,
            use_textline_orientation=False,
            # use_angle_cls=False,
            use_doc_unwarping=False,
        )
            
    temp_path, original_filename = file_info
    print("Start parsing: ", original_filename)
    
    if True:
    # try:
        # 이미지 로드 및 OCR 처리 (기존 로직 동일)
        # ... (생략: 이미지 변환 및 ocr 실행) ...
        img_arr = get_img_arr_from_file_name(temp_path)
        ocr_res = local_ocr.ocr(img_arr)
        ocr_res = ocr_res[0]
        
        text = extract_text_from_paddle(ocr_res)
        textline = get_ocr_lines(ocr_res)

        if "거래일시" in textline:
            pay_date = extract_payment_date_with_keyword(textline)
        else:
            pay_date = extract_payment_date_without_keyword(text)

        biz_no = extract_biz_number(text)
        merchant = extract_merchant_name(textline)
        amount = extract_payment_amount(textline)
        
        # API 호출 및 로그 기록
        tax_type = "오류"
        if biz_no and active_key:
            tax_type = get_tax_type_from_nts_with_api_call_counter(client_ip, biz_no, active_key)
            tax_type = normalize_tax_type(tax_type)

        # 파일명 변경 규칙 적용
        # [YYMMDD]_[TaxType]_[Amount]_[Merchant]
        renamed_name     = f"{pay_date}_{tax_type}_{amount}_{merchant}.png"
        visualized_name  = f"{pay_date}_{tax_type}_{amount}_{merchant}_vis.png"

        # New
        # (A) 이름만 바뀐 원본 이미지 저장
        final_origin_path = os.path.join(result_dir, renamed_name)
        Image.fromarray(img_arr).save(final_origin_path)
        
        # (B) OCR 결과(BB)가 포함된 이미지 별도 저장
        image_pil = draw_bb_on_img(img_arr, ocr_res)
        image_pil.save(os.path.join(ocr_vis_dir, visualized_name))

        print("End parsing: ", original_filename)

        return {"status": "success",
                    "data": {
                        "original_name": original_filename,
                        "renamed_name": renamed_name,
                        "vis_name": visualized_name,
                        "merchant": merchant,
                        "biz_no": biz_no,
                        "pay_date": pay_date,
                        "amount": amount,
                        "tax_type": tax_type
                    }
                }
    # except Exception as e:
    #     print("Failed to parse : ", file_info.filename)
    #     return {"status": "error",
    #             "message": str(e),
    #                 "data": {
    #                     "original_name"   : original_filename,
    #                     "renamed_name"    : original_filename,
    #                     "vis_name"        : original_filename,
    #                     "merchant"        : "UNKNOWN",
    #                     "biz_no"          : "000-00-00000",
    #                     "pay_date"        : "000000",
    #                     "amount"          : "00000",
    #                     "tax_type"        : "불명"
    #                 }
    #             }
