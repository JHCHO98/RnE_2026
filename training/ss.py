import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

# ==========================================
# 1. CSV -> list[dict]
# ==========================================
def csv_to_list_of_dicts(file_path):
    try:
        df = pd.read_csv(file_path)

        # label 정수화
        df["label"] = df["label"].astype(int)

        # 필요한 컬럼 결측 제거
        df = df.dropna(subset=["title", "comment", "label"]).copy()

        return df.to_dict("records")

    except Exception as e:
        print(f"오류 발생: {e}")
        return None


data_list = csv_to_list_of_dicts("data/data_channel.csv")

# ==========================================
# 2. 설정
# ==========================================
CONFIG = {
    "model_name": "beomi/KcELECTRA-base",
    "num_classes": 3,
    "batch_size": 8,
    "epochs": 10,
    "learning_rate": 2e-5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # 제목+댓글 pair 전체 최대 길이
    "max_length": 256,

    # 한 샘플당 최대 댓글 window 수
    "max_windows_per_sample": 4,
}

print("device:", CONFIG["device"])

tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])


# ==========================================
# 3. 긴 댓글 분할 함수
# ==========================================
def split_comments(comment_text):
    """
    '|||' 기준으로 댓글 분리
    """
    if pd.isna(comment_text):
        return []

    parts = [x.strip() for x in re.split(r"\s*\|\|\|\s*", str(comment_text)) if x.strip()]
    return parts


def build_comment_windows(title, comment_text, tokenizer, max_length=256, max_windows=4):
    """
    제목 길이를 고려해서 댓글을 여러 window로 나눔.
    각 window는 tokenizer(title, window) 했을 때 너무 길지 않도록 최대한 묶는다.
    """
    comments = split_comments(comment_text)

    if len(comments) == 0:
        return [""]

    # 제목 토큰 수를 먼저 계산
    title_ids = tokenizer.encode(str(title), add_special_tokens=False)
    # [CLS], [SEP], [SEP] 등 여유분 고려
    comment_budget = max(32, max_length - len(title_ids) - 8)

    windows = []
    current_comments = []
    current_len = 0

    for c in comments:
        c_ids = tokenizer.encode(c, add_special_tokens=False)
        c_len = len(c_ids)

        # 현재 window에 넣기 어려우면 새 window 시작
        if current_comments and current_len + c_len > comment_budget:
            windows.append(" ".join(current_comments))
            current_comments = [c]
            current_len = c_len
        else:
            current_comments.append(c)
            current_len += c_len

    if current_comments:
        windows.append(" ".join(current_comments))

    # window가 너무 많으면 앞/뒤를 균형 있게 남김
    if len(windows) > max_windows:
        front = max_windows // 2
        back = max_windows - front
        windows = windows[:front] + windows[-back:]

    return windows


# ==========================================
# 4. Dataset
# ==========================================
class YouTubeBiasDataset(Dataset):
    def __init__(self, data, tokenizer, config):
        self.samples = []

        for item in data:
            title = str(item["title"])
            comment = str(item["comment"])
            label = int(item["label"])

            windows = build_comment_windows(
                title=title,
                comment_text=comment,
                tokenizer=tokenizer,
                max_length=config["max_length"],
                max_windows=config["max_windows_per_sample"]
            )

            self.samples.append({
                "title": title,
                "comment_windows": windows,
                "label": label
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ==========================================
# 5. collate_fn
# ==========================================
def make_collate_fn(tokenizer, config):
    def collate_fn(batch):
        """
        batch 안의 각 샘플은 여러 comment window를 가질 수 있음.
        각 window를 title과 pair로 만들어 flatten 후 한 번에 tokenize.
        이후 pair_owner로 어떤 window가 어떤 원본 샘플 것인지 추적.
        """
        pair_titles = []
        pair_comments = []
        pair_owner = []
        labels = []

        for sample_idx, item in enumerate(batch):
            labels.append(item["label"])

            for window in item["comment_windows"]:
                pair_titles.append(item["title"])
                pair_comments.append(window)
                pair_owner.append(sample_idx)

        encodings = tokenizer(
            pair_titles,
            pair_comments,
            padding=True,
            truncation="only_second",   # 제목은 최대한 보존, 댓글만 자름
            max_length=config["max_length"],
            return_tensors="pt"
        )

        pair_owner = torch.tensor(pair_owner, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return encodings, pair_owner, labels

    return collate_fn


# ==========================================
# 6. 모델
# ==========================================
class PairChunkClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config["num_classes"]

        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"],
            num_labels=config["num_classes"]
        )

    def forward(self, encodings, pair_owner, num_samples):
        """
        encodings: flatten된 (title, comment_window) pair들
        pair_owner: 각 pair가 batch 내 몇 번째 샘플 소속인지
        num_samples: 원래 batch 샘플 수
        """
        outputs = self.backbone(**encodings)
        pair_logits = outputs.logits   # (num_pairs, num_classes)

        # 샘플별 logits 평균
        sample_logits = torch.zeros(
            num_samples,
            self.num_classes,
            device=pair_logits.device
        )
        counts = torch.zeros(
            num_samples,
            1,
            device=pair_logits.device
        )

        sample_logits.index_add_(0, pair_owner, pair_logits)
        counts.index_add_(
            0,
            pair_owner,
            torch.ones(pair_owner.size(0), 1, device=pair_logits.device)
        )

        sample_logits = sample_logits / counts.clamp(min=1.0)
        return sample_logits


# ==========================================
# 7. stratify split
# ==========================================
if data_list is None or len(data_list) < 10:
    raise ValueError("데이터가 너무 적음")

all_labels = [int(x["label"]) for x in data_list]

train_data, temp_data = train_test_split(
    data_list,
    test_size=0.2,
    random_state=42,
    stratify=all_labels
)

temp_labels = [int(x["label"]) for x in temp_data]

val_data, test_data = train_test_split(
    temp_data,
    test_size=0.5,
    random_state=42,
    stratify=temp_labels
)

print("train:", len(train_data))
print("val:", len(val_data))
print("test:", len(test_data))


# ==========================================
# 8. DataLoader
# ==========================================
train_dataset = YouTubeBiasDataset(train_data, tokenizer, CONFIG)
val_dataset = YouTubeBiasDataset(val_data, tokenizer, CONFIG)
test_dataset = YouTubeBiasDataset(test_data, tokenizer, CONFIG)

collate_fn = make_collate_fn(tokenizer, CONFIG)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    collate_fn=collate_fn
)


# ==========================================
# 9. 학습/평가 함수
# ==========================================
def run_epoch(model, dataloader, optimizer, criterion, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    for encodings, pair_owner, labels in dataloader:
        encodings = {k: v.to(device) for k, v in encodings.items()}
        pair_owner = pair_owner.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(encodings, pair_owner, num_samples=labels.size(0))
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, macro_f1


# ==========================================
# 10. 학습
# ==========================================
device = CONFIG["device"]
model = PairChunkClassifier(CONFIG).to(device)

optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
criterion = nn.CrossEntropyLoss()

best_val_f1 = -1

for epoch in range(CONFIG["epochs"]):
    train_loss, train_acc, train_f1 = run_epoch(
        model, train_loader, optimizer, criterion, device, train=True
    )
    val_loss, val_acc, val_f1 = run_epoch(
        model, val_loader, optimizer, criterion, device, train=False
    )

    print(
        f"[{epoch+1}/{CONFIG['epochs']}] "
        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "best_model.pt")
        print("best model saved.")


# ==========================================
# 11. 테스트
# ==========================================
model.load_state_dict(torch.load("best_model.pt", map_location=device))
test_loss, test_acc, test_f1 = run_epoch(
    model, test_loader, optimizer=None, criterion=criterion, device=device, train=False
)

print(f"TEST | loss={test_loss:.4f} acc={test_acc:.4f} macro_f1={test_f1:.4f}")