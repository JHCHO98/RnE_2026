import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer
from mecab import MeCab
import random
import re
from tqdm import tqdm

# ==========================================
# 1. 초기 설정 및 CUDA 확인
# ==========================================
MODEL_NAME = "klue/bert-base"
INPUT_FILE = "data_raw/data_channel_extended.csv"   # 실제 파일명으로 수정하세요
OUTPUT_FILE = "augmented_cuda_final.csv"

# CUDA 사용 가능 여부 체크
device_id = 0 if torch.cuda.is_available() else -1
if device_id == 0:
    print(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name(0)}에서 실행합니다.")
else:
    print("⚠️ CUDA를 찾을 수 없어 CPU로 실행합니다.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
fill_mask = pipeline("fill-mask", model=MODEL_NAME, device=device_id)
mecab = MeCab()

# ==========================================
# 2. 전처리 및 증강 핵심 함수
# ==========================================

def clean_text(text):
    """이모티콘 제거 및 공백 정규화"""
    if not isinstance(text, str): return ""
    # 유니코드 이모지 범위 제거
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    # [UNK], [MASK] 등 혹시 원본에 있을지 모를 특수 토큰 제거
    text = text.replace("[UNK]", "").replace("[MASK]", "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_proper_nouns(text):
    """Mecab NNP(고유명사) 추출"""
    try:
        return [word for word, pos in mecab.pos(text) if pos == 'NNP']
    except:
        return []

def augment_unit(unit_text, ratio=0.15):
    """개별 문장 증강 (실패 시 원본 단어로 복구)"""
    unit_text = clean_text(unit_text)
    if len(unit_text) < 5:
        return unit_text

    proper_nouns = get_proper_nouns(unit_text)
    words = unit_text.split()
    
    # 고유명사 제외 후보 인덱스
    candidates = [i for i, word in enumerate(words) if not any(pn in word for pn in proper_nouns)]
    
    if not candidates:
        return unit_text

    n_masks = min(max(1, int(len(candidates) * ratio)), 3)
    target_indices = random.sample(candidates, n_masks)
    
    current_words = words.copy()
    backup_words = words.copy() # 원본 보존용

    for idx in target_indices:
        original_word = backup_words[idx]
        current_words[idx] = tokenizer.mask_token
        masked_text = " ".join(current_words)
        
        success = False
        try:
            results = fill_mask(masked_text)
            for res in results:
                new_word = res['token_str'].replace("##", "").strip()
                
                # 특수 토큰 제외 및 유효성 검사
                if new_word.startswith("[") and new_word.endswith("]"):
                    continue
                
                if new_word != original_word and len(new_word) > 0:
                    current_words[idx] = new_word
                    success = True
                    break
        except:
            pass
        
        # 실패하거나 [UNK] 등이 들어가면 원본 기반으로 수정(복구)
        if not success or current_words[idx] in [tokenizer.unk_token, tokenizer.mask_token]:
            current_words[idx] = original_word
            
    return " ".join(current_words)

def process_row(text, is_comment=False):
    """행 단위 처리 (||| 구분자 대응)"""
    if pd.isna(text) or str(text).strip() == "":
        return text

    if is_comment:
        # 1. ||| 구분자로 분리
        units = text.split("|||")
        # 2. 각 유닛별 증강 (실패 시 원본 복구가 augment_unit 내부에 있음)
        augmented_units = [augment_unit(u, ratio=0.15) for u in units]
        return " ||| ".join(augmented_units)
    else:
        return augment_unit(text, ratio=0.2)

# ==========================================
# 3. 데이터 로드 및 실행
# ==========================================

def main():
    try:
        # 원본 데이터 로드
        df = pd.read_csv(INPUT_FILE)
        print(f"📂 파일 로드 완료: {len(df)} 행")
    except Exception as e:
        print(f"❌ 파일 로드 오류: {e}")
        return

    tqdm.pandas()

    print("\n[1/2] Title 증강 (CUDA 가속)...")
    df['aug_title'] = df['title'].progress_apply(lambda x: process_row(x, is_comment=False))

    print("\n[2/2] Comment 증강 (||| 구분자 & CUDA 가속)...")
    df['aug_comment'] = df['comment'].progress_apply(lambda x: process_row(x, is_comment=True))

    # 결과 저장 (utf-8-sig로 엑셀 깨짐 방지)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✨ 증강 완료! 저장된 파일: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()