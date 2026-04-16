import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt # ⭐ 시각화를 위한 라이브러리 추가 ⭐

def csv_to_list_of_dicts(file_path):
    try:
        # 1. CSV 파일 읽기
        df = pd.read_csv(file_path)

        # 2. 'label' 열을 정수형(int)으로 변환
        try:
            df['label'] = df['label'].astype(int)
        except ValueError:
            print("경고: 'label' 열에 정수로 변환할 수 없는 값이 포함되어 있습니다. 문자열로 유지합니다.")
            
        # 3. DataFrame을 딕셔너리 리스트로 변환
        result_list = df.to_dict('records')
        
        return result_list
    
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"오류: 파일이 비어 있습니다: {file_path}")
        return None
    except Exception as e:
        print(f"처리 중 예상치 못한 오류 발생: {e}")
        return None


# 2. 함수 실행
# 경고: 'silver_data_fixed.csv' 파일이 실제로 존재해야 정상 동작합니다.
data_list = csv_to_list_of_dicts("data/data_pls.csv")

# ==========================================
# 1. 설정 및 하이퍼파라미터
# ==========================================
CONFIG = {
    'model_name': "beomi/KcELECTRA-base", # ⭐ KcELECTRA 모델로 변경 ⭐
    'num_classes': 3, # 예: 0:보수, 1:진보, 2:중립 (학습용 라벨)
    'hidden_dim': 768,# ⭐ KcELECTRA-base의 Hidden Size (768)로 변경 ⭐
    'nhead': 8,# Multi-head Attention 헤드 수 (768 / 8 = 96)
    'num_layers': 4,# Transformer Encoder 레이어 수
    'batch_size': 16,
    'epochs': 50,
    'learning_rate': 0.00003,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu' # CUDA 우선
}


print(f"Current Device: {CONFIG['device']}")


# ==========================================
# 2. 데이터셋 클래스 (전처리된 데이터 가정)
# ==========================================
class YouTubeBiasDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 모델 입력: [제목, 합쳐진 댓글] -> 리스트 형태
        texts = [item['title'], item['comment']]
        label = item['label']
        return texts, label

def collate_fn(batch):
    # 배치 처리를 위해 텍스트 리스트와 라벨을 분리
    texts = [item[0] for item in batch] # [[제목, 댓글], [제목, 댓글], ...]
    labels = torch.tensor([item[1] for item in batch])
    return texts, labels

# ==========================================
# 3. 모델 아키텍처 (SBERT + Self-Attention)
# ==========================================
class BiasAnalyzer(nn.Module):
    def __init__(self, config):
        super(BiasAnalyzer, self).__init__()
        self.device = config['device']
        
        # A. Backbone: KR-SBERT
        print(f"Loading SBERT model: {config['model_name']}...")
        self.bert = AutoModel.from_pretrained(config['model_name'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        
        # B. Interaction: Transformer Encoder (Self-Attention)
        # batch_first=True -> (Batch, Seq_len, Dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['hidden_dim'],
            nhead=config['nhead'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
        
        # C. Task Head: Classification (Fine-tuning용)
        h_dim = config['hidden_dim']
        self.classifier = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim), # 안정적인 학습을 위해 배치 정규화 추가
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(h_dim, h_dim // 2),
            nn.BatchNorm1d(h_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(h_dim // 2, h_dim // 8),
            nn.ReLU(),
            
            nn.Linear(h_dim // 8, config['num_classes'])
        )

        # 모델을 GPU로 이동
        self.to(self.device)

    def get_sbert_embeddings(self, flat_texts):
        # 텍스트 리스트를 토크나이징 및 임베딩
        inputs = self.tokenizer(
            flat_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512 # 댓글이 합쳐져 있으니 길이를 좀 넉넉하게
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.bert(**inputs)
        
        # Mean Pooling (문장 단위 임베딩 추출)
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask

    def forward(self, batch_texts):
        """
        batch_texts: [ ["제목1", "댓글모음1"], ["제목2", "댓글모음2"] ] (Batch Size, 2)
        """
        batch_size = len(batch_texts)
        seq_len = 2 # [Title, Comments] 고정
        
        # 1. Flatten: BERT에 넣기 위해 1차원으로 폄
        # ["제목1", "댓글1", "제목2", "댓글2"]
        flat_texts = [t for video in batch_texts for t in video]
        
        # 2. SBERT Embedding
        # Shape: (Batch * 2, Hidden_Dim)
        flat_embeddings = self.get_sbert_embeddings(flat_texts)
        
        # 3. Reshape & Self-Attention
        # 다시 (Batch, 2, Hidden)으로 묶어서 Transformer에 넣음
        # 여기서 제목과 댓글이 서로 Attention Score를 계산함
        context_input = flat_embeddings.view(batch_size, seq_len, -1)
        context_output = self.transformer(context_input)
        
        # 4. Aggregation (Final Video Vector)
        # 제목과 댓글의 맥락이 섞인 두 벡터를 평균내서 '영상 벡터'로 만듦
        video_vector = torch.mean(context_output, dim=1) # Shape: (Batch, Hidden)
        
        # 5. Classification
        logits = self.classifier(video_vector)
        
        return logits, video_vector


# ==========================================
# 4. 실행 및 테스트 (Main Loop)
# ==========================================
if __name__ == "__main__":
    
    # --- 데이터 로드 및 유효성 검사 ---
    if data_list is None or len(data_list) < 5: # 최소 샘플 수 5개로 가정 (분할 가능하도록)
        print("오류: CSV 파일 로드에 실패했거나 데이터가 너무 적어 학습을 시작할 수 없습니다.")
        exit()

    # --- A. 데이터 분할 (Split) ---
    print(f"\n[데이터 분할 시작] 전체 샘플 수: {len(data_list)}개")
    
    # 1차 분할: Train (80%) vs Temp (20%)
    train_data, temp_data = train_test_split(
        data_list,
        test_size=0.2,
        random_state=42 # 재현성을 위해 고정
    )

    # 2차 분할: Temp (20%)를 Validation (10%)과 Test (10%)로 분할
    # 0.5 * 0.2 = 0.1 (총 데이터의 10%)
    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,
        random_state=42
    )
    
    # 학습 데이터가 비었는지 재확인 (ZeroDivisionError 방지)
    if len(train_data) == 0:
        print("\n!!! 치명적 오류: 학습 데이터셋(Train Data)이 0개입니다. !!!")
        exit()

    print(f"  - 학습(Train) 데이터: {len(train_data)}개")
    print(f"  - 검증(Validation) 데이터: {len(val_data)}개")
    print(f"  - 테스트(Test) 데이터: {len(test_data)}개")
    
    # --- B. 데이터 로더 준비 ---
    train_dataset = YouTubeBiasDataset(train_data)
    val_dataset = YouTubeBiasDataset(val_data)
    test_dataset = YouTubeBiasDataset(test_data)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    # --- C. 모델 초기화 ---
    model = BiasAnalyzer(CONFIG)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    best_val_accuracy = 0.0 # Best Model 저장을 위한 변수

    # ⭐ 시각화를 위한 history 딕셔너리 초기화 ⭐
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    # ⭐ ----------------------------------- ⭐

    # --- D. 학습 (Fine-tuning) 및 검증 ---
    print("\n[Start Training & Validation]")
    
    for epoch in range(CONFIG['epochs']):
        # 1. TRAIN PHASE
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for texts, labels in train_dataloader:
            labels = labels.to(CONFIG['device'])
            
            optimizer.zero_grad()
            logits, _ = model(texts)
            loss = criterion(logits, labels)
            
            # 정확도 계산
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        epoch_loss = total_loss / len(train_dataloader)
        train_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        # 2. VALIDATION PHASE
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        
        with torch.no_grad():
            for texts, labels in val_dataloader:
                labels = labels.to(CONFIG['device'])
                logits, _ = model(texts)
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_samples += labels.size(0)

        val_accuracy = val_correct / val_samples if val_samples > 0 else 0.0
        val_loss_avg = val_loss / len(val_dataloader)

        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_accuracy:.4f}")

        # ⭐ history 딕셔너리에 값 기록 ⭐
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss_avg)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)
        # ⭐ --------------------------- ⭐

        # 3. Best Model 저장 (Early Stopping)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            MODEL_SAVE_PATH = 'please.pt'
            torch.save(model, MODEL_SAVE_PATH)
            print(f"-> [모델 저장] 검증 정확도 {best_val_accuracy:.4f} 달성, {MODEL_SAVE_PATH}에 저장.")
            
    
    print("\n--- 학습 완료 ---")

    # ⭐ --- E. 시각화 및 파일 저장 --- ⭐
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
        plt.savefig('please.png')
        plt.close()
        print("✅ Loss 그래프 ('please.png') 저장 완료.")

        # 2. Accuracy 그래프
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
        plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig('please.png')
        plt.close()
        print("✅ Accuracy 그래프 ('please.png') 저장 완료.")

    # 시각화 실행 
    plot_and_save_metrics(history, CONFIG['epochs'])
    # ⭐ ----------------------------- ⭐


    # --- F. 최종 테스트 (Test Set으로 최종 성능 확인) ---
    print("\n[Start Final Test on Test Set]")
    
    # ... (모델 로드 및 eval 모드 설정 코드) ...

    model.eval()
    test_correct = 0
    test_samples = 0
    
    # 벡터와 제목을 모을 리스트
    all_vectors_list = []
    all_titles_list = []
    
    with torch.no_grad():
        test_dataloader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)
        for texts, labels in test_dataloader:
            labels = labels.to(CONFIG['device'])
            logits, vectors = model(texts)
            
            preds = torch.argmax(logits, dim=1)
            test_correct += (preds == labels).sum().item()
            test_samples += labels.size(0)
            
            # 군집화에 사용될 벡터와 제목을 리스트에 추가
            all_vectors_list.append(vectors.cpu().numpy())
            all_titles_list.extend([t[0] for t in texts]) # 제목도 누적

    # 루프가 끝난 후, 모든 벡터를 하나의 Numpy 배열로 합침
    if len(all_vectors_list) > 0:
        all_vectors = np.concatenate(all_vectors_list, axis=0)
    else:
        all_vectors = np.array([]) # 빈 배열 처리
    
    # 최종 테스트 정확도 계산
    test_accuracy = test_correct / test_samples if test_samples > 0 else 0.0
    print(f"\n=== 최종 테스트 결과 ===\nTest Accuracy: {test_accuracy:.4f} (Total {test_samples} Samples)")

    # --- G. 군집화 (Clustering) ---
    if all_vectors.shape[0] >= CONFIG['num_classes']: # 군집화 조건 확인
        print("\n[Start Clustering on Test Data Embeddings]")
        # K-Means 군집화 실행
        kmeans = KMeans(n_clusters=CONFIG['num_classes'], random_state=42, n_init=10)
        labels = kmeans.fit_predict(all_vectors)

        print(f"\n=== 군집화 결과 (총 {all_vectors.shape[0]} 샘플 기준) ===")
        # 제목 출력 시 누적된 제목 리스트 사용
        for i, title in enumerate(all_titles_list):
            print(f"영상 제목: '{title}' -> 군집 ID: {labels[i]}")

    else:
        print(f"\n[군집화 건너뜀] 샘플 수({all_vectors.shape[0]}개)가 군집 수({CONFIG['num_classes']}개)보다 적습니다.")

    print("\n=== 작업 완료 ===")