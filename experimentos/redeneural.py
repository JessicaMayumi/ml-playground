# FEITO COM INTELIGÊNCIA ARTIFICIAL

import os
import csv
import numpy as np
import pandas as pd
import random
import string
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class FinancialNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(FinancialNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.dropout1 = nn.Dropout(0.3)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(0.2)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(hidden_size2, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.dropout2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x
    
    def predict(self, x):
        with torch.no_grad():
            x = self.forward(x)
            return self.softmax(x)
        
class FinancialClassifier:
    def __init__(self):
        self.csv_file = 'dataset.csv'
        self.model_file = 'model.pt'
        self.vectorizer_file = 'vectorizer.pkl'
        self.label_encoder_file = 'label_encoder.pkl'
        self.input_size_file = 'input_size.pkl'
        
        # Inicializar ou carregar o CSV
        self.init_csv()
        
        # Carregar modelo se existir
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.input_size = None
        self.load_classifier_if_exists()
        
    def init_csv(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['transacao', 'categoria'])
    
    def load_classifier_if_exists(self):
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.input_size_file):
                # Carregar o tamanho de entrada para inicializar o modelo corretamente
                with open(self.input_size_file, 'rb') as f:
                    self.input_size = pickle.load(f)
                    num_categories = pickle.load(f)
                
                # Inicializar o modelo com as dimensões corretas
                self.model = FinancialNeuralNetwork(
                    input_size=self.input_size,
                    hidden_size1=128,
                    hidden_size2=64,
                    output_size=num_categories
                )
                
                # Carregar os pesos do modelo
                self.model.load_state_dict(torch.load(self.model_file))
                self.model.eval()  # Modo de avaliação
                print("Modelo carregado com sucesso!")
                
            if os.path.exists(self.vectorizer_file):
                with open(self.vectorizer_file, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                    
            if os.path.exists(self.label_encoder_file):
                with open(self.label_encoder_file, 'rb') as f:
                    self.label_encoder = pickle.load(f)
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")
    
    def add_transaction(self, transaction, category):
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([transaction, category])
        print(f"Transação '{transaction}' cadastrada na categoria '{category}'")
    
    def load_data(self):
        if os.path.exists(self.csv_file):
            return pd.read_csv(self.csv_file)
        return pd.DataFrame(columns=['transacao', 'categoria'])
    
    def augment_text(self, text):
        techniques = [
            self._shuffle_chars,
            self._add_prefix_suffix,
            self._remove_chars,
            self._change_case,
            self._swap_digits
        ]
        
        # Escolhe uma técnica aleatória
        technique = random.choice(techniques)
        return technique(text)
    
    def _shuffle_chars(self, text):
        if len(text) <= 3:
            return text
            
        parts = list(text)
        idx1, idx2 = random.sample(range(len(parts)), 2)
        parts[idx1], parts[idx2] = parts[idx2], parts[idx1]
        return ''.join(parts)
    
    def _add_prefix_suffix(self, text):
        prefixes = ['pg-', 'pag-', 'pagto-', 'rec-', 'recb-', 'fat-', 'nf-']
        suffixes = ['-01', '-02', '12345', '98765', '/21', '/22']
        
        if random.choice([True, False]):
            # Adiciona prefixo se não tiver
            for prefix in prefixes:
                if text.startswith(prefix):
                    return text
            return random.choice(prefixes) + text
        else:
            # Adiciona sufixo
            return text + random.choice(suffixes)
    
    def _remove_chars(self, text):
        if len(text) <= 3:
            return text
            
        chars = list(text)
        to_remove = random.randint(1, min(3, len(chars) - 1))
        
        for _ in range(to_remove):
            if chars:
                idx = random.randrange(len(chars))
                chars.pop(idx)
                
        return ''.join(chars)
    
    def _change_case(self, text):
        if not text:
            return text
            
        start = random.randint(0, max(0, len(text) - 3))
        length = random.randint(1, min(len(text) - start, 5))
        
        prefix = text[:start]
        middle = text[start:start+length]
        suffix = text[start+length:]
        
        if random.choice([True, False]):
            middle = middle.upper()
        else:
            middle = middle.lower()
            
        return prefix + middle + suffix
    
    def _swap_digits(self, text):
        result = ""
        for char in text:
            if char.isdigit():
                result += str(random.randint(0, 9))
            else:
                result += char
        return result
    
    def balance_dataset(self, df, samples_per_category=None):
        categories = df['categoria'].unique()
        
        if samples_per_category is None:
            # Define um número igual para todas as categorias
            samples_per_category = 100
        
        balanced_data = []
        
        for category in categories:
            category_data = df[df['categoria'] == category]
            
            # Se temos exemplos suficientes, pegamos uma amostra
            if len(category_data) >= samples_per_category:
                sampled = category_data.sample(samples_per_category, replace=False)
                balanced_data.append(sampled)
            else:
                # Se não temos exemplos suficientes, usamos todos os existentes
                balanced_data.append(category_data)
                
                # E geramos dados sintéticos para completar
                synthetic_count = samples_per_category - len(category_data)
                
                print(f"Gerando {synthetic_count} exemplos sintéticos para categoria '{category}'")
                
                for _ in range(synthetic_count):
                    # Escolhe um exemplo aleatório da categoria para aumentar
                    if len(category_data) > 0:
                        original = category_data.sample(1)['transacao'].values[0]
                        augmented = self.augment_text(original)
                        
                        balanced_data.append(pd.DataFrame({
                            'transacao': [augmented],
                            'categoria': [category]
                        }))
        
        return pd.concat(balanced_data, ignore_index=True)
    
    def train_model(self):
        df = self.load_data()
        
        if len(df) < 2:
            print("Dataset insuficiente para treinar o modelo. Adicione mais transações.")
            return False
        
        if len(df['categoria'].unique()) < 2:
            print("É necessário ter pelo menos duas categorias para treinar o modelo.")
            return False
            
        # Balancear o dataset
        num_categories = len(df['categoria'].unique())
        samples_per_category = max(10, 1000 // num_categories)  # 1000 exemplos no total
        
        print(f"Balanceando dataset com {samples_per_category} exemplos por categoria...")
        balanced_df = self.balance_dataset(df, samples_per_category)
        
        print(f"Dataset balanceado: {len(balanced_df)} transações em {num_categories} categorias")
        
        # Preparar os dados
        X = balanced_df['transacao'].values
        y = balanced_df['categoria'].values
        
        # Vetorizar o texto
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5))
        X_vectorized = self.vectorizer.fit_transform(X).toarray()
        
        # Codificar categorias
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y_encoded, test_size=0.2, random_state=42
        )
        
        # Converter para tensores PyTorch
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Criar datasets e dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Construir o modelo
        self.input_size = X_vectorized.shape[1]
        self.model = FinancialNeuralNetwork(
            input_size=self.input_size,
            hidden_size1=128,
            hidden_size2=64,
            output_size=num_categories
        )
        
        # Definir função de perda e otimizador
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Treinar o modelo
        print("Treinando o modelo...")
        epochs = 15
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            # Avaliar no conjunto de teste a cada época
            if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
                self.model.eval()
                correct = 0
                total = 0
                test_loss = 0.0
                
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        test_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                accuracy = correct / total
                print(f"Época {epoch+1}/{epochs}, Perda: {running_loss/len(train_loader):.4f}, "
                      f"Acurácia de teste: {accuracy:.4f}")
        
        # Avaliar o modelo final
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f"Acurácia final do modelo: {accuracy:.4f}")
        
        # Salvar o modelo
        torch.save(self.model.state_dict(), self.model_file)
        
        # Salvar o vectorizer e o label_encoder
        with open(self.vectorizer_file, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            
        with open(self.label_encoder_file, 'wb') as f:
            pickle.dump(self.label_encoder, f)
            
        # Salvar o tamanho de entrada e número de categorias para reconstruir o modelo posteriormente
        with open(self.input_size_file, 'wb') as f:
            pickle.dump(self.input_size, f)
            pickle.dump(num_categories, f)
            
        print("Modelo salvo com sucesso!")
        return True
    
    def classify_transaction(self, transaction):
        # Vetoriza e converte para tensor
        X = self.vectorizer.transform([transaction]).toarray()
        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(X_tensor)

        # Softmax para converter logits em probabilidades
        probabilities = torch.softmax(outputs, dim=1)

        # Pega o índice da classe com maior probabilidade
        predicted_index = torch.argmax(probabilities, dim=1).item()
        predicted_label = self.label_encoder.inverse_transform([predicted_index])[0]

        # Pega a probabilidade da classe prevista
        predicted_prob = probabilities[0][predicted_index].item()

        return predicted_label, predicted_prob


def main():
    classifier = FinancialClassifier()
    
    while True:
        print("\n" + "=" * 50)
        print("CLASSIFICADOR DE LANÇAMENTOS FINANCEIROS")
        print("=" * 50)
        print("Menu:")
        print("1. Cadastrar nova transação")
        print("2. Treinar modelo")
        print("3. Classificar uma transação")
        print("4. Sair")
        
        option = input("\nEscolha uma opção (1-4): ")
        
        if option == '1':
            transaction = input("Digite a descrição da transação: ")
            category = input("Digite a categoria da transação: ")
            classifier.add_transaction(transaction, category)
            
        elif option == '2':
            print("Iniciando treinamento do modelo...")
            classifier.train_model()
            
        elif option == '3':
            transaction = input("Digite a transação para classificar: ")
            result = classifier.classify_transaction(transaction)
            
            if result:
                category, probability = result
                print(f"Classificação: {category}")
                print(f"Confiança: {probability:.2%}")
            
        elif option == '4':
            print("Encerrando o programa...")
            break
            
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()