#!/usr/bin/env python3

import os
import json
print("Loading AI libraries ..."); 
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from ai_utils_text import ModelConfig, ModelDataset, ModelClassifier, EarlyStopping, getDevice

CONFIG_FILE = "model_config.json"

def clearScreen():
    if os.name == 'nt':     # Si estàs a Windows
        os.system('cls')
    else:                   # Si estàs a Linux o macOS
        os.system('clear')

clearScreen()

from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Afegeix una barra de progrés
    for batch in tqdm(dataloader, desc="Training", leave=True, bar_format="{desc}:   {percentage:3.2f}% |{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['category'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        # Afegeix una barra de progrés
        for batch in tqdm(dataloader, desc="Evaluating", leave=True, bar_format="{desc}: {percentage:3.2f}% |{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['category'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

def main():

    # Carregar la configuració
    with open(CONFIG_FILE) as f:
        config_file = json.load(f)

    column_text = config_file['columns']['text']
    column_categories = config_file['columns']['categories']

    # Carregar i preprocessar les dades
    print("Loading data ...")
    df = pd.read_csv(config_file['paths']['data'], encoding='utf-8')
    df = df[[column_categories, column_text]]
    df = df.dropna()

    labels = df[column_categories].unique().tolist()
    print("Labels:", labels)

    # Configuració del model
    config = ModelConfig(config_file, labels)
    
    le = LabelEncoder()
    df[column_categories] = le.fit_transform(df[column_categories].astype(str))

    print("Initialize tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    VOCAB_SIZE = tokenizer.vocab_size

    # Dividir les dades
    X_train, X_test, y_train, y_test = train_test_split(
        df[column_text], df[column_categories], test_size=0.2, random_state=42
    )

    # Crear datasets i dataloaders
    train_dataset = ModelDataset(X_train.values, y_train.values, tokenizer, config.max_len)
    test_dataset = ModelDataset(X_test.values, y_test.values, tokenizer, config.max_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # Inicialitzar model i components d'entrenament
    device = getDevice()

    model = ModelClassifier(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=config.patience)

    # Entrenament
    best_accuracy = 0.0  # Per emmagatzemar la millor precisió observada
    for epoch in range(config.epochs):
        
        print("")
        
        # Entrena un nou model i el valora
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate_epoch(model, test_loader, criterion, device)
        print(f'Epoch {epoch+1}/{config.epochs} (Train: loss: {train_loss:.2f}, accuracy: {train_acc:.2f}) (Eval: loss: {test_loss:.2f}, accuracy: {test_acc:.2f})')
        
        # Compara l'ultim model entrenat amb el millor guardat
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), config_file['paths']['trained_network'])
            print(f"New best model saved with eval accuracy {(best_accuracy*100):.2f}%")

        # Guarda les metadades
        metadata = { "categories": labels, "label_encoder": le.classes_.tolist() }
        with open(config_file['paths']['metadata'], 'w') as metadata_file:
            json.dump(metadata, metadata_file)

        # Comprova l'early stopping
        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            model.load_state_dict(early_stopping.get_best_model())
            break

if __name__ == "__main__":
    main()