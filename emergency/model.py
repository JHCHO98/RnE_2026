import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

# 1. 데이터셋 클래스 (데이터 관리의 규격화)
class YouTubeBiasDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # [제목, 댓글] 리스트와 라벨 반환
        return [item['title'], item['comment']], item['label']

# 2. 배치 처리 함수
def collate_fn(batch):
    texts = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return texts, labels

# 3. 메인 모델 클래스
class BiasAnalyzer(nn.Module):
    def __init__(self, config):
        super(BiasAnalyzer, self).__init__()
        self.device = config['device']
        
        # Backbone & Tokenizer
        self.bert = AutoModel.from_pretrained(config['model_name'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.hidden_dim = self.bert.config.hidden_size # 768
        
        # Cross-Attention
        self.title_to_comment_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=8, batch_first=True
        )
        self.comment_to_title_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=8, batch_first=True
        )
        
        # Classifier
        combined_dim = self.hidden_dim * 4
        intermediate_dim = config.get('intermediate_dim', 512)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(combined_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(intermediate_dim, intermediate_dim // 2),
            nn.BatchNorm1d(intermediate_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(intermediate_dim // 2, config['num_classes'])
        )
        self.to(self.device)

    def forward(self, batch_texts):
        # (기존 forward 로직 동일)
        titles = [item[0] for item in batch_texts]
        comments = [item[1] for item in batch_texts]

        t_inputs = self.tokenizer(titles, return_tensors='pt', padding=True, truncation=True, max_length=256).to(self.device)
        c_inputs = self.tokenizer(comments, return_tensors='pt', padding=True, truncation=True, max_length=256).to(self.device)

        t_out = self.bert(**t_inputs).last_hidden_state
        c_out = self.bert(**c_inputs).last_hidden_state

        attn_t2c, _ = self.title_to_comment_attn(t_out, c_out, c_out, key_padding_mask=~(c_inputs['attention_mask'].bool()))
        attn_c2t, _ = self.comment_to_title_attn(c_out, t_out, t_out, key_padding_mask=~(t_inputs['attention_mask'].bool()))

        pool_t_self, pool_c_self = t_out.mean(dim=1), c_out.mean(dim=1)
        pool_t_cross, pool_c_cross = attn_t2c.mean(dim=1), attn_c2t.mean(dim=1)

        mega_vector = torch.cat([pool_t_self, pool_c_self, pool_t_cross, pool_c_cross], dim=-1)
        return self.classifier(mega_vector), mega_vector