import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import os

# KoBERT 모델/토크나이저 로드
# Option 1: skt/kobert (SentencePiece 필요)
# Option 2: monologg/kobert (SentencePiece 필요)
# Option 3: klue/bert-base (SentencePiece 불필요 - 대안)
MODEL_NAME = 'klue/bert-base'  # SentencePiece 설치가 안 되면 이것 사용

# 설정 상수
MAX_LEN = 128
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
NUM_CLASSES = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"사용할 장치: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU 장치 이름: {torch.cuda.get_device_name(0)}")

# --- 1. 데이터셋 준비 ---
DATA_FILE_PATH = "crawling_data/data_processed.csv"

try:
    if not os.path.exists(DATA_FILE_PATH):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {DATA_FILE_PATH}")
        
    data = pd.read_csv(DATA_FILE_PATH)
    
    # 데이터셋 구성
    data['text_input'] = data['title_clean'] + " [SEP] " + data['comment_clean'].fillna('')
    
    # 학습/검증 데이터 분리
    train_df, val_df = train_test_split(data, test_size=0.1, random_state=42)
    print(f"\n✅ 데이터 로드 및 분할 완료. 훈련 데이터: {len(train_df)}개, 검증 데이터: {len(val_df)}개")

except FileNotFoundError as e:
    print(f"경고: {e}. 학습을 위해 예시 더미 데이터를 사용합니다.")
    train_data = {
        'title_clean': ['코로나 주식 상황', '최신 아이폰 리뷰', '쉬운 베이킹 레시피', '고양이 영상 모음', '정치 토론 하이라이트'],
        'comment_clean': ['경제 채널 좋아요', '디자인이 별로네요', '따라하기 쉬워요 굿', '귀여워서 힐링됨', '너무 싸우는 듯'],
        'label_id': [0, 1, 2, 3, 4]
    }
    train_df = pd.DataFrame(train_data)
    val_df = train_df.copy()
    
    train_df['text_input'] = train_df['title_clean'] + " [SEP] " + train_df['comment_clean']
    val_df['text_input'] = val_df['title_clean'] + " [SEP] " + val_df['comment_clean']
    
    NUM_CLASSES = train_df['label_id'].nunique()


# 1-1. 커스텀 데이터셋 클래스 정의
class KOBERTDataset(Dataset):
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


# --- 핵심 수정: 토크나이저 로딩 ---
# SentencePiece가 설치되어 있다면: 'skt/kobert-base-v1' 또는 'monologg/kobert' 사용
# SentencePiece가 없다면: 'klue/bert-base' 사용 (현재 설정)
print("\n토크나이저 로딩 중...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    print(f"✅ 토크나이저 로드 완료: {MODEL_NAME}")
except ImportError as e:
    print(f"⚠️  오류 발생: {e}")
    print("SentencePiece 설치가 필요합니다. 다음 명령어를 실행하세요:")
    print("pip install sentencepiece")
    raise

# 데이터셋 및 데이터로더 생성
train_dataset = KOBERTDataset(train_df, tokenizer, MAX_LEN)
val_dataset = KOBERTDataset(val_df, tokenizer, MAX_LEN)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# --- 2. 모델 정의 ---
class KoBERTClassifier(torch.nn.Module):
    def __init__(self, bert, num_classes):
        super(KoBERTClassifier, self).__init__()
        self.bert = bert
        self.classifier = torch.nn.Linear(768, num_classes)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_output = outputs[0][:, 0, :]
        logits = self.classifier(cls_output)
        return logits


# 모델 로드 및 장치 이동
print("\n모델 로딩 중...")
bert_model = AutoModel.from_pretrained(MODEL_NAME)
model = KoBERTClassifier(bert_model, NUM_CLASSES)
model.to(DEVICE)
print("✅ 모델 로드 완료")


# --- 3. 학습 설정 ---
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_dataloader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# 클래스 불균형 보정
if 'label_id' in train_df.columns and len(train_df) > 0:
    full_weights = np.zeros(NUM_CLASSES)
    for idx, count in train_df['label_id'].value_counts().items():
        if count > 0:
            full_weights[idx] = 1.0 / count
    
    if (full_weights > 0).sum() > 0:
        full_weights = full_weights / full_weights.sum() * (full_weights > 0).sum()
    
    class_weights = torch.tensor(full_weights, dtype=torch.float).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights).to(DEVICE)
    print("✅ 클래스 불균형 보정 (Class Weighting)이 손실 함수에 적용되었습니다.")
else:
    loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)
    print("기본 CrossEntropyLoss가 적용되었습니다.")


# --- 4. 학습 및 평가 함수 ---
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask, token_type_ids)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)


def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)


# --- 5. 학습 루프 실행 ---
print("\n--- 학습 시작 ---")
best_accuracy = 0

for epoch in range(NUM_EPOCHS):
    print(f'\nEpoch {epoch + 1}/{NUM_EPOCHS}')
    print('-' * 10)

    # 학습
    train_acc, train_loss = train_epoch(
        model,
        train_dataloader,
        loss_fn,
        optimizer,
        DEVICE,
        scheduler
    )

    print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f}')

    # 검증
    val_acc, val_loss = eval_model(
        model,
        val_dataloader,
        loss_fn,
        DEVICE
    )

    print(f'Val   loss {val_loss:.4f} accuracy {val_acc:.4f}')

    # 모델 저장
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_kobert_model.bin')
        best_accuracy = val_acc
        print("-> Best model 저장 완료.")

print("\n--- 학습 완료 ---")
print(f"최고 검증 정확도: {best_accuracy:.4f}")