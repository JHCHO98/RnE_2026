import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

# --- 1. 설정 상수 (학습 시와 동일하게 설정) ---
MODEL_NAME = 'monologg/koelectra-base-v3-discriminator' # 학습에 사용한 모델 이름
NUM_CLASSES = 15                                      # 클래스 개수
MAX_LEN = 128                                         # 최대 길이
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'best_koelectra_model.bin'                  # 학습 시 저장한 모델 파일 이름

# (선택 사항) 예측 결과를 사람이 알아볼 수 있도록 클래스 레이블을 정의합니다.
# 실제 레이블 목록에 맞게 수정하세요.
CLASS_LABELS = {
    0: "게임",
    1: "과학과 기술",
    2: "노하우/스타일",
    3: "비영리/사회운동",
    4: "스포츠",
    5: "애완동물/동물",
    6: "여행/이벤트",
    7: "영화/애니메이션",
    8:"예능",
    9: "음악",
    10: "인물/블로그",
    11: "자동차/탈것",
    12: "정치",
    13: "코미디",
}

# --- 2. 모델 정의 (학습 시와 동일) ---
class KoELECTRAClassifier(torch.nn.Module):
    def __init__(self, electra, num_classes):
        super(KoELECTRAClassifier, self).__init__()
        self.electra = electra
        self.classifier = torch.nn.Linear(768, num_classes)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_output = outputs[0][:, 0, :] 
        logits = self.classifier(cls_output)
        return logits

# --- 3. 모델 로드 및 준비 ---

# 토크나이저 로드 (학습 시와 동일하게 use_fast=False 적용)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# 모델 아키텍처 로드
electra_model = AutoModel.from_pretrained(MODEL_NAME)
model = KoELECTRAClassifier(electra_model, NUM_CLASSES)

# 학습된 가중치 로드
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"\n✅ 모델 가중치 로드 완료: {MODEL_PATH}")
except FileNotFoundError:
    print(f"\n❌ 오류: 모델 파일 '{MODEL_PATH}'을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

# --- 4. 예측 함수 정의 ---

def predict_class(title, comment):
    """
    영상 제목과 댓글을 입력받아 분류 클래스를 예측합니다.
    """
    # 1. 입력 텍스트 결합 (학습 시와 동일한 포맷)
    text_input = title + " [SEP] " + comment
    
    # 2. 텍스트 토큰화
    encoding = tokenizer.encode_plus(
        text_input,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # 3. 데이터 장치 이동
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    token_type_ids = encoding['token_type_ids'].to(DEVICE)
    
    # 4. 예측 수행
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)
    
    # 5. 결과 해석
    probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy().flatten()
    predicted_index = np.argmax(probabilities)
    predicted_label = CLASS_LABELS.get(predicted_index, f"클래스 {predicted_index}")
    confidence = probabilities[predicted_index]
    
    return predicted_label, confidence, probabilities

# --- 5. 테스트 실행 ---

print("\n--- 예측 테스트 시작 ---")

# (예시 1) 스포츠 관련 영상
title_1 ="파란색 LED 만들기가 거의 불가능했던 이유"
comment_1 = "나무위키로 노벨상 수상내역 보다가 청색LED로 받았다길래 뭐지 싶었는데 정말 고난도 기술이었네..."
label_1, conf_1, _ = predict_class(title_1, comment_1)

print("-" * 30)
print(f"제목: {title_1}")
print(f"댓글: {comment_1}")
print(f"**예측 결과:** {label_1} (확률: {conf_1:.2f})")
