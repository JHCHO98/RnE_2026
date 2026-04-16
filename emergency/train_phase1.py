import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sys
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from models import BiasAnalyzer, YouTubeBiasDataset, collate_fn

# 로그 기록용 클래스
class Logger(object):
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self): pass

# --- 설정 및 경로 ---
TRIAL=int(input('몇 번째 모델이신가요??: '))
CONFIG = {
    'model_name': "beomi/KcELECTRA-base",
    'num_classes': 3,
    'batch_size': 4,
    'epochs': 100,
    'learning_rate': 2e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
LOG_PATH = f'emergency/log/log_phase1_{TRIAL}.txt'
MODEL_PATH = f'emergency/model/asdf{TRIAL}.pt'
sys.stdout = Logger(LOG_PATH)

def train():
    print(f"--- Phase 1 학습 시작 (Device: {CONFIG['device']}) ---")
    
    # 데이터 로드
    df = pd.read_csv("data/data_pls.csv")
    data_list = df.to_dict('records')
    train_d, val_d = train_test_split(data_list, test_size=0.2, random_state=42)

    train_loader = DataLoader(YouTubeBiasDataset(train_d), batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(YouTubeBiasDataset(val_d), batch_size=CONFIG['batch_size'], collate_fn=collate_fn)

    model = BiasAnalyzer(CONFIG)
    for param in model.bert.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0
    for epoch in range(CONFIG['epochs']):
        model.train()
        t_loss = 0
        for texts, labels in train_loader:
            labels = labels.to(CONFIG['device'])
            optimizer.zero_grad()
            logits, _ = model(texts)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for texts, labels in val_loader:
                labels = labels.to(CONFIG['device'])
                logits, _ = model(texts)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {t_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model, MODEL_PATH)
            print(f"-> [모델 저장] Best Acc: {best_acc:.4f}")

if __name__ == "__main__":
    train()