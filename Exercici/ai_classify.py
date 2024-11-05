#!/usr/bin/env python3

import os
import json
print("Loading AI libraries ..."); 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from typing import List
from ai_utils_text import ModelConfig, ModelDataset, ModelClassifier, getDevice

CONFIG_FILE = "model_config.json"

def clearScreen():
    if os.name == 'nt':     # Si estàs a Windows
        os.system('cls')
    else:                   # Si estàs a Linux o macOS
        os.system('clear')

clearScreen()

def predict_text(text: str, model: nn.Module, tokenizer, device: torch.device, config: ModelConfig, label_encoder):
    model.eval()  # Posa el model en mode d'avaluació
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=config.max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_label = label_encoder.inverse_transform([predicted.item()])[0]
    confidence = confidence.item()
    return predicted_label, confidence

def main():
    # Carregar la configuració
    with open(CONFIG_FILE) as f:
        config_file = json.load(f)

    # Carregar les metadades (s'han generat a l'entrenament)
    with open(config_file['paths']['metadata'], 'r') as f:
        metadata = json.load(f)
    labels = metadata["categories"]

    # Configuració del model
    config = ModelConfig(config_file, labels)

    # Inicialitzar tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Carregar dades
    df = pd.read_csv(config_file['paths']['data'], encoding='utf-8')
    df = df[[config_file['columns']['categories'], config_file['columns']['text']]].dropna()
    le = LabelEncoder()
    le.fit(metadata['label_encoder'])  # Inicialitza correctament les classes
    df[config_file['columns']['categories']] = le.transform(df[config_file['columns']['categories']].astype(str))

    # Seleccionar mostres i preparar dataset
    sample_df = df.sample(n=50)
    sample_dataset = ModelDataset(sample_df[config_file['columns']['text']].values, sample_df[config_file['columns']['categories']].values, tokenizer, config_file['model_configuration']['max_len'])
    sample_loader = DataLoader(sample_dataset, batch_size=1)

    # Carregar el model entrenat
    device = getDevice()

    model = ModelClassifier(config).to(device)
    model.load_state_dict(torch.load(config_file['paths']['trained_network'], map_location=device, weights_only=True))
    model.eval()

    # Validació
    correct = 0
    total = 0
    with torch.no_grad():
        for i, row in sample_df.iterrows():
            text = row[config_file['columns']['text']]
            true_label = le.inverse_transform([row[config_file['columns']['categories']]])[0]
            
            # Utilitza la funció predict_text per fer la predicció
            predicted_label, confidence = predict_text(text, model, tokenizer, device, config, le)
            
            # Comprova si la predicció és correcta
            is_correct = (predicted_label == true_label)
            correct += is_correct
            total += 1
            
            # Mostra el resultat de cada mostra
            print(f"Text: {text[:50].ljust(50)}..., Prediction: {f'{confidence:.2%}'.rjust(7)} = {'"'+predicted_label+'"':6} ({'"'+true_label+'"':6} > {'correct' if is_correct else 'wrong'})")

    # Càlcul de precisió i taxa d'error
    accuracy = correct / total
    error_rate = 1 - accuracy
    print(f'\nValidation of 50 examples: success: {correct}/{total}, accuracy: {(accuracy*100):.2f}, Error rate: {error_rate:.2f}')

if __name__ == "__main__":
    main()
