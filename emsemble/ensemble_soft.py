import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# ==========================================
# 1. 설정 및 하이퍼파라미터 (KcELECTRA 기반)
# ==========================================
CONFIG = {
    # *** KcELECTRA 모델 사용을 위해 변경 ***
    'model_name': "beomi/KcELECTRA-base-v2022", 
    'num_classes': 3,          # 0:보수, 1:진보, 2:중립 
    'hidden_dim': 768,         # base 모델의 히든 사이즈 (768 유지)
    'nhead': 8,                
    'num_layers': 2,           
    'batch_size': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu' 
}

# 앙상블할 모델 파일 경로 목록
MODEL_PATHS = [
    'best_debias_kcelectra1.bin',
    'best_debias_kcelectra2.bin',
    'best_debias_kcelectra3.bin',
]

print(f"Current Device: {CONFIG['device']}")
print(f"Base Model: {CONFIG['model_name']}")

# ==========================================
# 2. 데이터 처리 함수
# ==========================================

def csv_to_list_of_dicts(file_path):
    """
    CSV 파일을 읽어 리스트 형태의 딕셔너리로 변환하고, 'label' 열을 정수로 변환합니다.
    """
    try:
        df = pd.read_csv(file_path)
        
        try:
            # NaN을 -1 등 임시값으로 채운 후 정수로 변환하거나, 정수 변환 시도
            df['label'] = df['label'].astype(int)
        except ValueError:
            print("경고: 'label' 열에 정수로 변환할 수 없는 값이 포함되어 있습니다. 문자열로 유지합니다.")
            
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

# ==========================================
# 3. 데이터셋 및 콜레이트 함수
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
# 4. 모델 아키텍처 (BiasAnalyzer)
# ==========================================
class BiasAnalyzer(nn.Module):
    def __init__(self, config):
        super(BiasAnalyzer, self).__init__()
        self.device = config['device']
        
        # A. Backbone: KcELECTRA 로드
        print(f"Loading model: {config['model_name']}...")
        self.bert = AutoModel.from_pretrained(config['model_name'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        
        # B. Interaction: Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['hidden_dim'], 
            nhead=config['nhead'], 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
        
        # C. Task Head: Classification
        self.classifier = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config['hidden_dim'] // 2, config['num_classes'])
        )

        self.to(self.device)

    def get_sbert_embeddings(self, flat_texts):
        # 학습 시 사용된 Mean Pooling 로직 유지
        inputs = self.tokenizer(
            flat_texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.bert(**inputs)
        
        # Mean Pooling
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask

    def forward(self, batch_texts):
        batch_size = len(batch_texts)
        seq_len = 2 
        
        # 1. Flatten & Embedding
        flat_texts = [t for video in batch_texts for t in video]
        flat_embeddings = self.get_sbert_embeddings(flat_texts)
        
        # 2. Reshape & Self-Attention
        context_input = flat_embeddings.view(batch_size, seq_len, -1)
        context_output = self.transformer(context_input)
        
        # 3. Aggregation
        video_vector = torch.mean(context_output, dim=1)
        
        # 4. Classification
        logits = self.classifier(video_vector)
        
        return logits, video_vector

# ==========================================
# 5. 메인 앙상블 실행 루프 (Soft Voting)
# ==========================================
if __name__ == "__main__":
    
    # 5-1. 데이터 로드 및 분할 (테스트 셋 추출)
    data_list = csv_to_list_of_dicts("silver_data_fixed.csv")
    
    if data_list is None or len(data_list) < 5:
        print("오류: CSV 파일 로드에 실패했거나 데이터가 너무 적어 앙상블을 시작할 수 없습니다.")
        exit()

    # 기존 학습 코드와 동일한 80/10/10 분할 로직을 따라 테스트 셋 확보
    _, temp_data = train_test_split(data_list, test_size=0.2, random_state=42)
    _, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    test_dataset = YouTubeBiasDataset(test_data)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False,  # 앙상블을 위해 순서를 고정
        collate_fn=collate_fn
    )

    print(f"\n[테스트 데이터셋 준비 완료] 총 {len(test_data)}개 샘플")
    
    # 5-2. 모든 모델 로드
    ensemble_models = []
    print("\n[Start Loading Ensemble Models]")
    for path in MODEL_PATHS:
        try:
            # KcELECTRA 모델 아키텍처로 초기화 및 가중치 로드
            model = BiasAnalyzer(CONFIG)
            model.load_state_dict(torch.load(path, map_location=CONFIG['device']))
            model.eval()
            ensemble_models.append(model)
            print(f"-> Model loaded from: {path}")
        except FileNotFoundError:
            print(f"오류: 모델 파일을 찾을 수 없습니다: {path}. 해당 모델을 건너뜁니다.")
        except Exception as e:
            print(f"오류: 모델 로딩 중 예상치 못한 오류 발생: {e}")

    if len(ensemble_models) == 0:
        print("!!! 로드된 모델이 없어 앙상블을 진행할 수 없습니다. !!!")
        exit()

    # 5-3. 앙상블 예측 수행 (Soft Voting)
    print("\n[Start Soft Voting Ensemble Prediction (확률 평균)]")
    
    final_test_correct = 0
    final_test_samples = 0

    with torch.no_grad():
        for texts, labels in test_dataloader:
            labels_cpu = labels.to('cpu')
            
            # 각 모델의 확률(Softmax)을 모으기
            # probabilities_list: [ (Batch Size, Num Classes) x (Num Models) ]
            probabilities_list = [] 
            
            for model in ensemble_models:
                logits, _ = model(texts)
                
                # Softmax를 적용하여 확률로 변환
                probs = torch.softmax(logits, dim=1) 
                probabilities_list.append(probs.cpu())

            # (Num Models, Batch Size, Num Classes) 형태로 스택
            probabilities_tensor = torch.stack(probabilities_list, dim=0)
            
            # Soft Voting (확률 평균)
            # 모든 모델의 확률을 평균 냄
            avg_probs = torch.mean(probabilities_tensor, dim=0) # (Batch Size, Num Classes)
            
            # 가장 높은 평균 확률을 가진 클래스를 최종 예측으로 선택
            ensemble_preds = torch.argmax(avg_probs, dim=1)
            
            # 최종 정확도 계산
            final_test_correct += (ensemble_preds == labels_cpu).sum().item()
            final_test_samples += labels_cpu.size(0)

    # 5-4. 최종 앙상블 결과 출력
    final_test_accuracy = final_test_correct / final_test_samples if final_test_samples > 0 else 0.0

    print(f"\n=============================================")
    print(f"=== 🏆 최종 앙상블 (Soft Voting) 테스트 결과 🏆 ===")
    print(f"Test Accuracy: {final_test_accuracy:.4f} (Total {final_test_samples} Samples)")
    print(f"=============================================")

    print("\n=== 작업 완료 ===")