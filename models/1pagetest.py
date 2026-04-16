import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# ==========================================
# 1. 설정 상수 및 경로
# ==========================================
# KcELECTRA 모델/토크나이저 이름 (학습 시 사용된 것과 동일해야 함)
MODEL_NAME = 'monologg/koelectra-base-v3-discriminator' 
LOAD_MODEL_PATH = 'best_koelectra_model_15.bin' # 로드할 모델 파일
DATA_FILE_PATH = 'crawling_data/data_processed.csv' # 테스트 데이터 파일

MAX_LEN = 128    # 입력 시퀀스 최대 길이
BATCH_SIZE = 32
# ⚠️ 핵심 수정: Size Mismatch 오류에 따라, 모델 파일에 맞춰 15로 설정
NUM_CLASSES = 15 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"사용할 장치: {DEVICE}")

# ==========================================
# 2. 데이터셋 클래스 및 모델 아키텍처 (재정의)
# ==========================================

# 2-1. 커스텀 데이터셋 클래스 정의 (데이터 로드에 필요)
class KCELectraDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.sentences = df['text_input'].tolist()
        self.labels = df['label_id'].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.sentences[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(), 
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 2-2. 모델 정의 (학습된 모델 구조와 동일해야 함)
class KCELectraClassifier(nn.Module):
    def __init__(self, electra, num_classes):
        super(KCELectraClassifier, self).__init__()
        self.electra = electra
        # NUM_CLASSES=15 에 맞춰 분류층을 정의합니다.
        self.classifier = nn.Linear(768, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight) 

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_output = outputs[0][:, 0, :] 
        
        logits = self.classifier(cls_output)
        return logits


# ==========================================
# 3. 테스트 데이터 로드 및 분할
# ==========================================
try:
    full_data = pd.read_csv(DATA_FILE_PATH)
    
    # 텍스트 전처리 (학습 시와 동일해야 함)
    full_data['text_input'] = full_data['title_clean'] + " [SEP] " + full_data['comment_clean'].fillna('')
    
    # 테스트 셋 추출 (학습 시와 동일한 80:10:10 분할 로직 재현)
    # 전체 데이터의 20%를 임시 데이터로 분리 -> 그 중 절반(전체의 10%)을 테스트 셋으로 사용
    _, temp_df = train_test_split(full_data, test_size=0.2, random_state=42) 
    test_df, _ = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_dataset = KCELectraDataset(test_df, tokenizer, MAX_LEN)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\n[테스트 데이터셋 로드 완료] 총 샘플 수: {len(test_df)}개")

except FileNotFoundError:
    print(f"\n❌ 오류: 데이터 파일 '{DATA_FILE_PATH}'을(를) 찾을 수 없습니다. 테스트를 시작할 수 없습니다.")
    exit()
except Exception as e:
    print(f"\n❌ 데이터 로드 및 전처리 중 예상치 못한 오류 발생: {e}")
    exit()


# ==========================================
# 4. 모델 로드 및 평가 수행
# ==========================================

# 모델 아키텍처 초기화 (NUM_CLASSES=15)
electra_model = AutoModel.from_pretrained(MODEL_NAME)
final_model = KCELectraClassifier(electra_model, NUM_CLASSES) 

try:
    # 저장된 가중치 로드 (Size mismatch 오류 해결)
    final_model.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location=DEVICE))
    final_model.to(DEVICE)
    final_model.eval() 
    print(f"✅ 모델 가중치 로드 및 평가 모드 설정 완료: {LOAD_MODEL_PATH}")
except FileNotFoundError:
    print(f"\n❌ 오류: 모델 파일 '{LOAD_MODEL_PATH}'을(를) 찾을 수 없습니다. 성능 평가를 진행할 수 없습니다.")
    exit()
except RuntimeError as e:
    print(f"\n❌ 치명적 오류: 모델 로드 중 예상치 못한 Runtime Error 발생. 학습된 모델과 현재 NUM_CLASSES={NUM_CLASSES} 설정이 일치하는지 확인하십시오.")
    print(f"상세 오류: {e}")
    exit()

# --- 평가 수행 함수 ---
def evaluate_test_set(model, data_loader, device):
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Final Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            _, preds = torch.max(outputs, dim=1) 

            # 라벨과 예측 값을 CPU에서 가져와 저장
            all_labels.extend(batch['labels'].cpu().numpy()) 
            all_preds.extend(preds.cpu().numpy())

    return all_labels, all_preds

# --- 최종 결과 출력 ---

y_true, y_pred = evaluate_test_set(final_model, test_dataloader, DEVICE)

# 분류 리포트 출력
print("\n" + "="*60)
print("             🏆 최종 테스트 분류 보고서 (Classification Report) 🏆")
print("="*60)

# target_names를 15개로 생성
target_names = [f'Class {i}' for i in range(NUM_CLASSES)]

# y_true/y_pred에 실제로 존재하는 클래스(예: 0~13)에 대해서만 보고서를 출력하고,
# target_names도 실제 존재하는 클래스의 이름(14개)까지만 잘라서 사용합니다.
# 이렇게 하면 "Number of classes does not match size of target_names" 오류를 우회합니다.
unique_labels = np.unique(y_true)
labels_to_report = list(unique_labels)
names_to_report = [target_names[i] for i in labels_to_report]


print(classification_report(
    y_true, 
    y_pred, 
    labels=labels_to_report,             # 실제 데이터에 존재하는 라벨만 지정
    target_names=names_to_report,         # 해당 라벨의 이름만 지정
    zero_division=0
))

print("="*60)
test_accuracy = np.mean(np.array(y_true) == np.array(y_pred))
print(f"총 샘플 수: {len(y_true)}개")
print(f"최종 정확도: {test_accuracy:.4f}")
print("==============================================")