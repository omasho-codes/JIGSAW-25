import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import gc

class MultiTaskModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        hidden_size = base_model.config.hidden_size
        self.attention_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1)
        )
        self.class_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None, class_labels=None, prompt_lens=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        lm_loss = outputs.loss
        hidden_states = outputs.hidden_states[-1]
        class_logits = None
        class_loss = None

        if class_labels is not None and prompt_lens is not None:
            batch_size, seq_len, _ = hidden_states.shape
            prompt_mask = torch.arange(seq_len, device=self.model.device)[None, :] < prompt_lens[:, None]
            prompt_hidden_states = hidden_states.float()
            attn_scores = self.attention_head(prompt_hidden_states).squeeze(-1)
            attn_scores = attn_scores.masked_fill(~prompt_mask, -1e9)
            attention_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
            pooled = (prompt_hidden_states * attention_weights).sum(dim=1).float()
            class_logits = self.classifier(pooled).squeeze(-1)
            class_labels = class_labels.float()
            class_loss = self.class_loss_fn(class_logits, class_labels)

        return lm_loss, class_logits, class_loss


class RedditDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        prompt = f"""
        You are an AI assistant that detects rule violations in Reddit comments. 
        Output format strictly:
        rule_violation: <Yes/No>
        reasoning: <Explanation>
        Subreddit: {row['subreddit']}
        Subreddit rule: {row['rule']}
        Comment: {row['body']}
        Positive examples: 1. {row['positive_example_1']} 2. {row['positive_example_2']} 
        Negative examples: 1. {row['negative_example_1']} 2. {row['negative_example_2']}
        Answer:
        """
        answer = f"rule_violation: {row['rule_violation']} \n reasoning: {row['reasoning']}"
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=True)
        answer_tokens = self.tokenizer(answer + self.tokenizer.eos_token, add_special_tokens=False)
        prompt_len = len(prompt_tokens['input_ids'])
        input_ids = prompt_tokens['input_ids'] + answer_tokens['input_ids']
        attention_mask = prompt_tokens['attention_mask'] + answer_tokens['attention_mask']
        labels = [-100] * prompt_len + answer_tokens['input_ids']
        padding_len = self.max_len - len(input_ids)

        if padding_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_len
            attention_mask += [0] * padding_len
            labels += [-100] * padding_len
        else:
            input_ids = input_ids[:self.max_len]
            attention_mask = attention_mask[:self.max_len]
            labels = labels[:self.max_len]
            
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'class_label': torch.tensor(row["rule_violation"], dtype=torch.float),
            'prompt_len': torch.tensor(prompt_len, dtype=torch.long)
        }


MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
N_SPLITS = 5
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
EPOCHS = 10
ALPHA = 0.5

print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    output_hidden_states=True
)
print("Base model and tokenizer loaded.")

df = pd.read_csv("../train.csv")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

fold_aucs = []
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['rule_violation'])):
    print(f"\n========== Fold {fold + 1} / {N_SPLITS} ==========")
    peft_model = get_peft_model(base_model, lora_config)
    multi_model = MultiTaskModel(peft_model).to(peft_model.device)
    optimizer = torch.optim.AdamW(multi_model.parameters(), lr=LEARNING_RATE)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    train_dataset = RedditDataset(train_df, tokenizer)
    val_dataset = RedditDataset(val_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2)

    for epoch in range(EPOCHS):
        multi_model.train()
        loop = tqdm(train_loader, desc=f"Fold {fold+1} | Epoch {epoch+1}")
        for batch in loop:
            input_ids = batch['input_ids'].to(peft_model.device)
            attention_mask = batch['attention_mask'].to(peft_model.device)
            labels = batch['labels'].to(peft_model.device)
            class_labels = batch['class_label'].to(peft_model.device)
            prompt_lens = batch['prompt_len'].to(peft_model.device)

            lm_loss, class_logits, class_loss = multi_model(
                input_ids, attention_mask, labels, class_labels, prompt_lens
            )
            
            if class_loss is not None and lm_loss is not None:
                total_loss = ALPHA * class_loss + (1 - ALPHA) * torch.clamp(lm_loss, -15, 15)
            else:
                total_loss = lm_loss if lm_loss is not None else class_loss

            optimizer.zero_grad()
            total_loss.backward()
            clip_grad_norm_(multi_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            loop.set_postfix(
                total_loss=total_loss.item(),
                lm_loss=lm_loss.item() if lm_loss else 'N/A',
                cls_loss=class_loss.item() if class_loss else 'N/A'
            )
        
        multi_model.eval()
        y_true, y_pred = [],[]
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Fold {fold+1} | Validation"):
                input_ids = batch['input_ids'].to(peft_model.device)
                attention_mask = batch['attention_mask'].to(peft_model.device)
                class_labels = batch['class_label'].to(peft_model.device)
                prompt_lens = batch['prompt_len'].to(peft_model.device)

                _, class_logits, _ = multi_model(input_ids, attention_mask, prompt_lens=prompt_lens, class_labels=class_labels)
                
                if class_logits is not None:
                    probs = torch.sigmoid(class_logits)
                    y_true.extend(class_labels.cpu().numpy())
                    y_pred.extend(probs.cpu().numpy())
        
        if y_true:
            auc = roc_auc_score(y_true, y_pred)
            print(f"Fold {fold+1} | Epoch {epoch+1} Validation AUC: {auc:.4f}")
            if epoch == EPOCHS -1: 
                fold_aucs.append(auc)

    model_save_path = f"qwen_multi_task_fold{fold+1}.pth"
    torch.save(multi_model.state_dict(), model_save_path)
    print(f"✅ Saved model for fold {fold+1}: {model_save_path}")
    del multi_model, peft_model, optimizer, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

print("\n========== Cross-Validation Results ==========")

for i, auc in enumerate(fold_aucs):
    print(f"Fold {i+1} AUC: {auc:.4f}")
print(f"\nMean AUC across {len(fold_aucs)} folds: {sum(fold_aucs)/len(fold_aucs):.4f}")
