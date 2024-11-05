import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ModelConfig:
    def __init__(self, config_file, labels):
        # Configuració general del model
        self.is_binary = config_file['model_configuration']['is_binary']
        self.class_labels = labels
        self.num_classes = 2 if self.is_binary else len(self.class_labels)

        # Configuració de camins
        self.data_path = config_file['paths']['data']
        self.trained_network_path = config_file['paths']['trained_network']
        self.metadata_path = config_file['paths']['metadata']

        # Configuració de columnes
        self.column_categories = config_file['columns']['categories']
        self.column_text = config_file['columns']['text']

        # Configuració del model i l'optimitzador
        self.batch_size = config_file['training']['batch_size']
        self.epochs = config_file['training']['epochs']
        self.learning_rate = config_file['training']['learning_rate']
        self.max_len = config_file['model_configuration']['max_len']
        
        # Configuració de capes dinàmiques
        self.layers = config_file['model_configuration']['layers']
        
        # Configuració d'early stopping
        self.patience = config_file['optimization']['early_stopping']['patience']
        self.min_delta = config_file['optimization']['early_stopping']['min_delta']

# Classes del model
class ModelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        category = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'category': torch.tensor(category, dtype=torch.long)
        }

class ModelClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super(ModelClassifier, self).__init__()
        self.config = config
        layers = []

        for layer_config in config.layers:
            layer_type = layer_config['type']
            
            if layer_type == 'Embedding':
                layers.append(nn.Embedding(
                    num_embeddings=layer_config['vocab_size'],
                    embedding_dim=layer_config['embedding_dim']
                ))
            elif layer_type == 'Dropout':
                layers.append(nn.Dropout(p=layer_config['p']))
            elif layer_type == 'Linear':
                num_out_features = layer_config['out_features']
                if not isinstance(num_out_features, int):
                    num_out_features = config.num_classes
                layers.append(nn.Linear(
                    in_features=layer_config['in_features'],
                    out_features=num_out_features
                ))
            elif layer_type == 'ReLU':
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask):
        x = self.network[0](input_ids)
        masked = x * attention_mask.unsqueeze(-1)
        x = masked.mean(dim=1)
        
        for layer in self.network[1:]:
            x = layer(x)

        return x

class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss: float, model: nn.Module):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0
            
    def get_best_model(self):
        return self.best_model
    
def getDevice():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" or device.type == "mps":
        print(f"Using device: {device} (GPU accelerated)")
    else:
        print(f"Using device: {device} (CPU based)")
    
    return device