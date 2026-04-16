import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # 시각화를 위한 라이브러리 추가

# KcELECTRA 모델/토크나이저 로드
MODEL_NAME = 'monologg/koelectra-base-v3-discriminator' 

# 설정 상수
MAX_LEN = 128    # 입력 시퀀스 최대 길이
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 5e-5
NUM_CLASSES = 14 # 클래스 개수: 0부터 14까지 15개
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"사용할 장치: {DEVICE}")

# --- 1. 데이터셋 준비 (예시 데이터 로드 및 전처리) ---

# 실제 데이터 파일을 로드하는 부분입니다. (예: CSV 파일)
try:
    # 가정: data.csv 파일에 title_clean, comment_clean, label_id 컬럼이 있습니다.
    data = pd.read_csv("crawling_data/data_processed.csv") # 실제 파일 경로로 변경하세요.
    
    # --- 핵심 수정 부분 ---
    # 데이터셋 구성: 제목과 하나의 'comment_clean' 컬럼을 [SEP] 토큰으로 연결합니다.
    data['text_input'] = data['title_clean'] + " [SEP] " + data['comment_clean'].fillna('')
    # ------------------
    
    # 학습/검증 데이터 분리
    train_df, val_df = train_test_split(data, test_size=0.1, random_state=42)

except FileNotFoundError:
    print("경고: 'your_data.csv' 파일을 찾을 수 없습니다. 예시 더미 데이터를 사용합니다.")
    # 파일이 없는 경우를 위한 더미 데이터 생성 (단일 comment_clean 컬럼 사용)
    train_data = {
        'title_clean': ['안녕하세요', 'KcELECTRA', '댓글이 중요합니다', '다중 클래스', '코드 작성'],
        'comment_clean': ['영상 리뷰 좋아요', '구어체와 신조어가 많음', '이 모델로 분류합니다', '클래스 15개 충분', '최종적으로 완성'],
        'label_id': [0, 14, 7, 3, 11]
    }
    train_df = pd.DataFrame(train_data)
    val_df = train_df.copy()
    
    # --- 핵심 수정 부분 (더미 데이터) ---
    train_df['text_input'] = train_df['title_clean'] + " [SEP] " + train_df['comment_clean']
    val_df['text_input'] = val_df['title_clean'] + " [SEP] " + val_df['comment_clean']
    # ------------------


# 1-1. 커스텀 데이터셋 클래스 정의 (이전 코드와 동일)
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

# KcELECTRA 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 데이터셋 및 데이터로더 생성
train_dataset = KCELectraDataset(train_df, tokenizer, MAX_LEN)
val_dataset = KCELectraDataset(val_df, tokenizer, MAX_LEN)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# --- 2. 모델 정의 (분류층 추가) (이전 코드와 동일) ---
class KCELectraClassifier(torch.nn.Module):
    def __init__(self, electra, num_classes):
        super(KCELectraClassifier, self).__init__()
        self.electra = electra
        self.classifier = torch.nn.Linear(768, num_classes)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # last_hidden_state의 첫 번째 토큰([CLS] 토큰)을 사용
        cls_output = outputs[0][:, 0, :] 
        
        logits = self.classifier(cls_output)
        return logits

# 모델 로드 및 장치 이동
electra_model = AutoModel.from_pretrained(MODEL_NAME)
model = KCELectraClassifier(electra_model, NUM_CLASSES)
model.to(DEVICE)

# --- 3. 학습 설정 (클래스 불균형 가중치 적용) ---

# 옵티마이저 및 스케줄러
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE) # 'correct_bias=False' 제거
total_steps = len(train_dataloader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# [클래스 불균형 보정 로직]
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


# --- 4. 학습 및 평가 함수 정의 ---

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

history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
}

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

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    # Tensor 값을 Python 기본 타입으로 변환 (.item() 사용)
    history['train_acc'].append(train_acc.item()) 
    history['val_acc'].append(val_acc.item())

    # 모델 저장 (가장 좋은 성능의 모델)
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_koelectra_model_14.bin')
        best_accuracy = val_acc
        print("-> Best model 저장 완료.")

print("\n--- 학습 완료 ---")


# --- 6. 시각화 및 파일 저장 함수 정의 ---

def plot_and_save_metrics(history, num_epochs):
    epochs = range(1, num_epochs + 1)

    # 1. Loss 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.close() # 메모리 해제
    print("✅ Loss 그래프 ('loss_plot.png') 저장 완료.")

    # 2. Accuracy 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_plot.png')
    plt.close() # 메모리 해제
    print("✅ Accuracy 그래프 ('accuracy_plot.png') 저장 완료.")

# 시각화 실행 (NUM_EPOCHS는 상수 설정 값 사용)
plot_and_save_metrics(history, NUM_EPOCHS)