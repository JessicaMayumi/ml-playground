import os
import csv
import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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
        print(f"\033[32m\033[1mNova entrada adicionada: '{transacao}' como '{categoria}'\033[0;0m")

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

        numero_aleatorio = str(random.randint(0, 9999))

        modificadores = random.sample(['prefixo', 'sufixo', 'numeros'], random.randint(1, 3))

        resultado = texto
        resultado = resultado + numero_aleatorio

        if 'prefixo' in modificadores:
            resultado = random.choice(prefixos) + resultado
        if 'sufixo' in modificadores:
            resultado = resultado + random.choice(sufixos)
        if 'numeros' in modificadores:
            resultado = resultado + numero_aleatorio

        return resultado
    
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
    
    def gerarDataset(self, tamanho = 10000000): 
        df = self.lerDataset()
        if df.empty:
            print("\033[31m\033[1mNenhum dado no dataset. Adicione entradas primeiro.\033[0;0m")
            return None
        self.categorias = self.getCategorias(df)
        numCategorias = len(self.categorias)
        if numCategorias == 0:
            print("\033[31m\033[1mNenhuma categoria encontrada.\033[0;0m")
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

        df_final = pd.DataFrame(dados_balanceados, columns=["transacao", "categoria"])
        df_final.to_csv("dataset_aumentado.csv", index=False)
        print("\033[32m\033[1mDataset balanceado e aumentado salvo como 'dataset_aumentado.csv'\033[0;0m")
        return df_final
    
    def treinaModelo(self, datasetBalenceado = None):
        if datasetBalenceado is None:
            datasetBalenceado = self.gerarDataset()
        if datasetBalenceado is None or datasetBalenceado.empty:
            print("\033[31m\033[1mImpossível treinar: nenhum dado disponível.\033[0;0m")
            return False
        
        X = datasetBalenceado["transacao"]
        y = datasetBalenceado["categoria"]

        self.pipeline = Pipeline([('vetorizador', TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=1.0)), ('classificador', MultinomialNB())]) 

        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
        self.pipeline.fit(X_treino, y_treino)

        y_pred = self.pipeline.predict(X_teste)
        acuracia = accuracy_score(y_teste, y_pred)
        print(f"\033[32m\033[1mModelo treinado com sucesso. Acurácia: {acuracia:.2f}\n\033[0;0m")
        print("\033[36m\033[1mRelatório de Classificação:\033[0;0m")
        print(classification_report(y_teste, y_pred))
        return True
    
    def classificarCategoria(self, transacao):
        if self.pipeline is None:
            print("\033[31m\033[1mO modelo ainda não foi treinado.\033[0;0m")
            return None
        proba = self.pipeline.predict_proba([transacao])[0]
        probabilidade = np.max(proba)
        predicao = self.pipeline.classes_[np.argmax(proba)]

        if probabilidade < 0.7:
            return "outros", probabilidade
        return predicao, probabilidade
    
def main():
    classificador = ClassificarTransacao()

    while True:
        print("\n\033[36m\033[1m⊳  ⊳  ⊳   Classificar Transações   ⊳  ⊳  ⊳   \033[0m")
        print("\033[35m\033[1m1. Adicionar nova entrada\033[0m")
        print("\033[35m\033[1m2. Treinar modelo\033[0m")
        print("\033[35m\033[1m3. Prever categoria\033[0m")
        print("\033[35m\033[1m4. Sair\033[0m")
        escolha = input("\033[35mDigite sua opção: \033[0;0m\033[35m")

        if escolha == "1":
            texto = input("\033[35m\033[1mDigite o texto da transação: \033[0;0m\033[35m")
            categoria = input("\033[35m\033[1mDigite a categoria: \033[0;0m\033[35m")
            classificador.adicionaEntrada(texto, categoria)

        elif escolha == "2":
            dataset = classificador.gerarDataset()
            if dataset is not None:
                print(f"\033[32m\033[1mDataset balanceado gerado com {len(dataset)} entradas.\033[0;0m")
                classificador.treinaModelo(dataset)

        elif escolha == "3":
            if classificador.pipeline is None:
                print("\033[31m\033[1mTreine o modelo primeiro.\033[0;0m")
            else:
                texto = input("\033[35m\033[1mDigite o texto da transação para classificar: \033[0;0m\033[35m")
                pred, conf = classificador.classificarCategoria(texto)
                print(f"Categoria prevista: {pred} ({conf:.2f})")

        elif escolha == "4":
            print("\033[33m\033[1mAté mais!\033[0;0m")
            break

        else: 
            print("\033[31m\033[1mOpção inválida. Digite um número entre 1 e 4.\033[0;0m")

if __name__ == "__main__":
    main()