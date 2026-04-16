import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model import BiasAnalyzer, YouTubeBiasDataset, collate_fn

# 1. Logger 설정
class Logger(object):
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self): pass

# --- 설정 및 경로 ---
TRIAL = "9"
LOG_PATH = f'emergency/log/log_phase2_{TRIAL}.txt'
LOAD_PATH = f'emergency/model/asdf{TRIAL}.pt'
SAVE_PATH = f'emergency/model/asdf{TRIAL}_final.pt'
LOSS_PLOT_PATH = f'emergency/plot/loss_phase2_{TRIAL}.png'
ACC_PLOT_PATH = f'emergency/plot/acc_phase2_{TRIAL}.png'

sys.stdout = Logger(LOG_PATH)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def resume_training():
    print(f"--- Phase 2 미세 조정 시작 (Device: {device}) ---")
    
    # [수정] weights_only=False 추가하여 로드 에러 해결
    model = torch.load(LOAD_PATH, weights_only=False)
    model.to(device)

    # 레이어 해동 전략 (상위 4개 레이어 + 분류기)
    for param in model.parameters(): param.requires_grad = False
    for i in range(8, 12): 
        for param in model.bert.encoder.layer[i].parameters(): param.requires_grad = True
    for module in [model.title_to_comment_attn, model.comment_to_title_attn, model.classifier]:
        for param in module.parameters(): param.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 데이터 준비
    df = pd.read_csv("data/data_pls.csv")
    data_list = df.to_dict('records')
    train_d, val_d = train_test_split(data_list, test_size=0.2, random_state=42)
    train_loader = DataLoader(YouTubeBiasDataset(train_d), batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(YouTubeBiasDataset(val_d), batch_size=16, collate_fn=collate_fn)

    # 시각화 데이터 저장용
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.6032 # Phase 1 최고 기록
    early_stop_count = 0

    for epoch in range(500):
        # --- TRAIN ---
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for texts, labels in train_loader:
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(texts)
            loss = criterion(logits, labels)
            loss.backward(); optimizer.step()
            
            t_loss += loss.item()
            t_correct += (torch.argmax(logits, 1) == labels).sum().item()
            t_total += labels.size(0)
        
        # --- VALIDATION ---
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for texts, labels in val_loader:
                labels = labels.to(device)
                logits, _ = model(texts)
                v_loss += criterion(logits, labels).item()
                v_correct += (torch.argmax(logits, 1) == labels).sum().item()
                v_total += labels.size(0)

        # 결과 계산 및 기록
        train_loss_avg = t_loss / len(train_loader)
        val_loss_avg = v_loss / len(val_loader)
        train_acc = t_correct / t_total
        val_acc = v_correct / v_total
        
        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        scheduler.step(val_acc)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model, SAVE_PATH)
            print(f"⭐️ [갱신] 최고 정확도: {best_acc:.4f}")
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= 50:
                print("🛑 조기 종료 (Early Stopping)")
                break

    # --- 시각화 함수 ---
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    # Loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.title('Phase 2 Loss')
    plt.legend(); plt.grid(True)
    
    # Acc 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc')
    plt.plot(epochs_range, history['val_acc'], label='Val Acc')
    plt.title('Phase 2 Accuracy')
    plt.legend(); plt.grid(True)
    
    plt.savefig(f'emergency/summary_phase2_{TRIAL}.png')
    print(f"✅ 그래프 저장 완료: emergency/summary_phase2_{TRIAL}.png")

if __name__ == "__main__":
    resume_training()