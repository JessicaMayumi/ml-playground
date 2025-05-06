import os
import csv
import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

class ClassificarTransacao: 
    def __init__(self, dataset_path="dataset.csv"):
        self.dataset_path = dataset_path
        self.categorias = []
        self.modelo = None
        self.pipeline = None

    def lerDataset(self):
        if not os.path.exists(self.dataset_path):
            with open(self.dataset_path, "w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["transacao", "categoria"])
            return pd.DataFrame(columns = ["transacao", "categoria"])
        
        return pd.read_csv(self.dataset_path)
    
    def adicionaEntrada(self, transacao, categoria):
        df = self.lerDataset()
        novaEntrada = pd.DataFrame([[transacao, categoria]], columns = ["transacao", "categoria"])
        df = pd.concat([df, novaEntrada], ignore_index=True)
        df.to_csv(self.dataset_path, index = False)
        print(f"Nova entrada adicionada: '{transacao}' como '{categoria}'")

    def getCategorias(self, df):
        return df["categoria"].unique().tolist()
    
    def aumentar_texto(self, texto): 
        tecnicas = [
            lambda t: t,
            lambda t: self._embaralhar_palavras(t),
            lambda t: self._adicionar_prefixo_sufixo(t),
            lambda t: self._trocar_caracteres(t),
            lambda t: self._remover_caractere_aleatorio(t)
        ]
        tecnica = random.choice(tecnicas)
        return tecnica(texto)
    
    def _embaralhar_palavras(self, texto):
        palavras = texto.split()
        if len(palavras) <= 1:
            return texto
        if len(palavras) > 2:
            meio = palavras[1:-1]
            random.shuffle(meio)
            palavras = [palavras[0]] + meio + [palavras[-1]]
        random.shuffle(palavras)
        return ' '.join(palavras)
    
    def _adicionar_prefixo_sufixo(self, texto):
        prefixos = ['tx-', 'pag-', 'tr-', 'pg-', 'pmt-']
        sufixos = ['-123', '-pagamento', '-tx', '-01', '-transferencia']
        
        if random.choice([True, False]):
            return random.choice(prefixos) + texto
        else:
            return texto + random.choice(sufixos)
    
    def _trocar_caracteres(self, texto):
        if len(texto) <= 1:
            return texto
        lista = list(texto)
        idx = random.randint(0, len(lista) - 2)
        lista[idx], lista[idx + 1] = lista[idx + 1], lista[idx]
        return ''.join(lista)
    
    def _remover_caractere_aleatorio(self, texto):
        if len(texto) <= 1:
            return texto
        lista = list(texto)
        idx = random.randint(0, len(lista) - 1)
        lista.pop(idx)
        return ''.join(lista)
    
    def gerarDataset(self, tamanho = 1000000): 
        df = self.lerDataset()
        if df.empty:
            print("Nenhum dado no dataset. Adicione entradas primeiro.")
            return None
        self.categorias = self.getCategorias(df)
        numCategorias = len(self.categorias)
        if numCategorias == 0:
            print("Nenhuma categoria encontrada.")
            return None
        amostras_por_categoria = max(1, tamanho // numCategorias)
        dados_balanceados = []

        for categoria in self.categorias:
            dadosCategoria = df[df["categoria"] == categoria]
            textos = dadosCategoria["transacao"].tolist()
            if len(textos) < amostras_por_categoria:
                aumentados = []
                for i in range(amostras_por_categoria - len(textos)):
                    if textos:
                        base = random.choice(textos)
                        aumentados.append(self.aumentar_texto(base))
                textos.extend(aumentados)
            if len(textos) > amostras_por_categoria:
                selecionados = random.sample(textos, amostras_por_categoria)
            else:
                selecionados = textos
            for texto in selecionados:
                dados_balanceados.append([texto, categoria])

        return pd.DataFrame(dados_balanceados, columns=["transacao","categoria"])
    
    def treinaModelo(self, datasetBalenceado = None):
        if datasetBalenceado is None:
            datasetBalenceado = self.gerarDataset()
        if datasetBalenceado is None or datasetBalenceado.empty:
            print("Impossível treinar: nenhum dado disponível.")
            return False
        
        X = datasetBalenceado["transacao"]
        y = datasetBalenceado["categoria"]

        self.pipeline = Pipeline([('vetorizador', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)), ('classificador', LinearSVC(max_iter=1000))]) 
        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
        self.pipeline.fit(X_treino, y_treino)

        y_pred = self.pipeline.predict(X_teste)
        acuracia = accuracy_score(y_teste, y_pred)
        print(f"Modelo treinado com sucesso. Acurácia: {acuracia:.2f}\n")
        print("Relatório de Classificação:")
        print(classification_report(y_teste, y_pred))
        return True
    
    def classificarCategoria(self, transacao):
        if self.pipeline is None:
            print("O modelo ainda não foi treinado.")
            return None
        predicao = self.pipeline.predict([transacao])[0]
        return predicao, None
    
def main():
    classificador = ClassificarTransacao()

    while True:
        print("\n=== Classificador de Transações Financeiras ===")
        print("1. Adicionar nova entrada rotulada")
        print("2. Treinar modelo")
        print("3. Prever categoria")
        print("4. Sair")
        escolha = input("Digite sua opção: ")

        if escolha == "1":
            texto = input("Digite o texto da transação: ")
            categoria = input("Digite a categoria: ")
            classificador.adicionaEntrada(texto, categoria)

        elif escolha == "2":
            dataset = classificador.gerarDataset()
            if dataset is not None:
                print(f"Dataset balanceado gerado com {len(dataset)} entradas.")
                classificador.treinaModelo(dataset)

        elif escolha == "3":
            if classificador.pipeline is None:
                print("Treine o modelo primeiro.")
            else:
                texto = input("Digite o texto da transação para classificar: ")
                pred, conf = classificador.classificarCategoria(texto)
                if conf is not None:
                    print(f"Categoria prevista: {pred} (confiança: {conf:.2f})")
                else:
                    print(f"Categoria prevista: {pred}")

        elif escolha == "4":
            print("Até mais!")
            break

        else: 
            print("Opção inválida. Digite um número entre 1 e 4.")

if __name__ == "__main__":
    main()