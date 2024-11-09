#!/usr/bin/env python
# coding: utf-8

# In[241]:


# Importar bibliotecas
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import (
    HillClimbSearch, BicScore, K2Score, MaximumLikelihoodEstimator, BayesianEstimator
)
from pgmpy.inference import VariableElimination
import numpy as np


# In[242]:


# Carregar o dataset Titanic
# Se estiver usando o Google Colab, você pode carregar o arquivo 'train.csv' diretamente
# Aqui, vamos usar um link direto para o dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Carregar o dataset
data = pd.read_csv(url)


# In[243]:


# Verificar as primeiras linhas do dataset
print("Primeiras linhas do dataset:")
print(data.head())

# Verificar valores ausentes
print("\nValores ausentes por coluna:")
print(data.isnull().sum())

# Preencher valores ausentes em 'Age' com a mediana
data['Age'].fillna(data['Age'].median(), inplace=True)

# Preencher valores ausentes em 'Embarked' com o valor mais frequente
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Excluir a coluna 'Cabin' devido ao grande número de valores ausentes
data.drop(columns=['Cabin'], inplace=True)

# Converter 'Sex' para valores numéricos
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Converter 'Embarked' para valores numéricos
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Selecionar as colunas relevantes
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Discretizar 'Age' e 'Fare'
from sklearn.preprocessing import KBinsDiscretizer

# Configurar o discretizador para criar 5 faixas com base na frequência (quantile)
age_discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

# Ajustar e transformar a coluna 'Age'
data['Age'] = age_discretizer.fit_transform(data[['Age']]).astype(int)

# Verificar os dados após a discretização
print("\nDados após a discretização de 'Age' com KBinsDiscretizer:")
print(data.head())






data['Fare'] = pd.qcut(data['Fare'], 4, labels=[0, 1, 2, 3])

# Converter todas as colunas para inteiro
data = data.astype(int)

# Verificar os dados após o pré-processamento
print("\nDados após o pré-processamento:")
print(data.head())


# In[244]:


print("\nEstatísticas descritivas do dataset:\n")
print(data.describe())


# In[245]:


# Gráfico de contagem de sobreviventes
sns.countplot(x='Survived', data=data)
plt.title('Distribuição de Sobreviventes')
plt.show()

# Gráfico de sobrevivência por sexo
sns.countplot(x='Sex', hue='Survived', data=data)
plt.title('Sobrevivência por Sexo')
plt.show()

# Gráfico de sobrevivência por classe
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title('Sobrevivência por Classe')
plt.show()

# Gráfico de dispersão para 'Age' vs 'Fare'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=data)
plt.title('Idade vs Tarifa')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Embarked', hue='Survived', data=data)
plt.title('sobrevivencia por embarked')
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(x='Parch', hue='Survived', data=data)
plt.title('sobrevivencia por parch')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='SibSp', hue='Survived', data=data)
plt.title('sobrevivencia por sibsp')
plt.show()


# In[246]:


from sklearn.feature_selection import mutual_info_classif

# Definir as variáveis independentes e a variável alvo
X = data.drop('Survived', axis=1)
y = data['Survived']

# Calcular a mutual information
mi = mutual_info_classif(X, y, discrete_features=True)

# Exibir a importância das variáveis
importance = pd.Series(mi, index=X.columns)
importance.sort_values(ascending=False, inplace=True)
print("Importância das variáveis (Mutual Information):")
print(importance)


# In[247]:


# Separar a variável alvo (Survived) das variáveis preditoras
X = data.drop('Survived', axis=1)
y = data['Survived']

# Dividir os dados em treinamento e teste
train_data, test_data = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data['Survived']
)

print(f"\nConjunto de Treinamento: {len(train_data)} registros")
print(f"Conjunto de Teste: {len(test_data)} registros")


# In[248]:


# Usando Hill Climbing e BIC Score
est_hc = HillClimbSearch(train_data)
best_model_bic = est_hc.estimate(scoring_method=BicScore(train_data))

# Usando Hill Climbing e K2 Score
k2score = K2Score(train_data)
best_model_k2 = est_hc.estimate(scoring_method=k2score)


# In[249]:


# Modelo com BIC Score
model_bic = BayesianNetwork(best_model_bic.edges())
model_bic.fit(train_data, estimator=BayesianEstimator, prior_type='BDeu')

# Modelo com K2 Score
model_k2 = BayesianNetwork(best_model_k2.edges())
model_k2.fit(train_data, estimator=BayesianEstimator, prior_type='BDeu')


# In[250]:


# Estrutura com BIC Score
plt.figure(figsize=(12, 8))
G_bic = nx.DiGraph(model_bic.edges())
pos_bic = nx.spring_layout(G_bic, seed=42)
nx.draw(G_bic, pos_bic, with_labels=True, node_size=3000, node_color="skyblue",
        font_size=12, font_weight="bold", arrows=True)
plt.title("Estrutura da Rede Bayesiana (BIC Score)")
plt.show()

# Estrutura com K2 Score
plt.figure(figsize=(12, 8))
G_k2 = nx.DiGraph(model_k2.edges())
pos_k2 = nx.spring_layout(G_k2, seed=42)
nx.draw(G_k2, pos_k2, with_labels=True, node_size=3000, node_color="lightgreen",
        font_size=12, font_weight="bold", arrows=True)
plt.title("Estrutura da Rede Bayesiana (K2 Score)")
plt.show()


# In[251]:


def predict(model, data, evidence_vars, target_var='Survived'):
    inference = VariableElimination(model)
    predictions = []
    for idx, row in data.iterrows():
        evidence = {var: row[var] for var in evidence_vars}
        try:
            query_result = inference.query(variables=[target_var], evidence=evidence)
            prob = query_result.values
            predicted_state = np.argmax(prob)
            predictions.append(int(predicted_state))
        except Exception as e:
            #print(f'Erro no índice {idx}: {e}')
            predictions.append(0)  # Valor padrão em caso de erro
    return predictions

# Definir as variáveis de evidência
evidence_vars = data.columns.tolist()
evidence_vars.remove('Survived')

# Avaliar o modelo BIC
predictions_bic = predict(model_bic, test_data, evidence_vars)
accuracy_bic = accuracy_score(test_data['Survived'], predictions_bic)
print(f"\nAcurácia do Modelo BIC no Conjunto de Teste: {accuracy_bic:.4f}")

# Avaliar o modelo K2
predictions_k2 = predict(model_k2, test_data, evidence_vars)
accuracy_k2 = accuracy_score(test_data['Survived'], predictions_k2)
print(f"\nAcurácia do Modelo K2 no Conjunto de Teste: {accuracy_k2:.4f}")


# In[252]:


print("=== Discussão dos Resultados ===\n")

print(f"A acurácia do Modelo BIC é {accuracy_bic:.2f}")
print(f"A acurácia do Modelo K2 é {accuracy_k2:.2f}")




# In[253]:


import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# Função de predição com probabilidade
def predict(model, data, evidence_vars, target_var='Survived'):
    """
    Realiza predições utilizando o modelo fornecido.

    Args:
        model: Modelo treinado (BIC ou K2).
        data (pd.DataFrame): Dados para predição.
        evidence_vars (list): Lista de variáveis de evidência.
        target_var (str): Variável alvo (padrão 'Survived').

    Returns:
        predictions (list): Lista de classes previstas.
        probabilities (list): Lista de probabilidades da classe positiva.
    """
    inference = VariableElimination(model)
    predictions = []
    probabilities = []
    for idx, row in data.iterrows():
        evidence = {var: row[var] for var in evidence_vars}
        try:
            query_result = inference.query(variables=[target_var], evidence=evidence)
            prob = query_result.values
            predicted_state = np.argmax(prob)
            predictions.append(int(predicted_state))
            # Assumindo que a classe positiva é '1'
            probabilities.append(prob[1] if len(prob) > 1 else prob[0])
        except Exception as e:
            print(f"Erro na linha {idx}: {e}")
            predictions.append(0)
            probabilities.append(0.0)
    return predictions, probabilities

# Definir as variáveis de evidência
evidence_vars = test_data.columns.tolist()
if 'Survived' not in evidence_vars:
    print("A variável 'Survived' não está presente no test_data.")
else:
    evidence_vars.remove('Survived')  # Remover a variável alvo

# Verificar se todas as variáveis de evidência estão presentes no modelo
model_vars = model_bic.nodes()
missing_vars_model = [var for var in evidence_vars if var not in model_vars]
if missing_vars_model:
    print(f"As seguintes variáveis de evidência não estão presentes no modelo BIC: {missing_vars_model}")

# Avaliar o modelo BIC
predictions_bic, probabilities_bic = predict(model_bic, test_data, evidence_vars)
logloss_bic = log_loss(test_data['Survived'], probabilities_bic)
# Verificar se há pelo menos duas classes para calcular o AUC
if len(test_data['Survived'].unique()) > 1:
    auc_bic = roc_auc_score(test_data['Survived'], probabilities_bic)
else:
    auc_bic = float('nan')  # Não é possível calcular o AUC com uma única classe

print(f"\n=== Avaliação do Modelo BIC ===")
print(f"Entropia Cruzada (Log Loss): {logloss_bic:.4f}")
print(f"AUC-ROC: {auc_bic:.4f}")

# Avaliar o modelo K2
predictions_k2, probabilities_k2 = predict(model_k2, test_data, evidence_vars)
logloss_k2 = log_loss(test_data['Survived'], probabilities_k2)
# Verificar se há pelo menos duas classes para calcular o AUC
if len(test_data['Survived'].unique()) > 1:
    auc_k2 = roc_auc_score(test_data['Survived'], probabilities_k2)
else:
    auc_k2 = float('nan')  # Não é possível calcular o AUC com uma única classe

print(f"\n=== Avaliação do Modelo K2 ===")
print(f"Entropia Cruzada (Log Loss): {logloss_k2:.4f}")
print(f"AUC-ROC: {auc_k2:.4f}")


# In[254]:


# Discussão dos resultados
print("\n=== Discussão dos Resultados ===\n")

print(f"A Entropia Cruzada (Log Loss) do Modelo BIC é {logloss_bic:.4f}")
print(f"O AUC-ROC do Modelo BIC é {auc_bic:.4f}")
print(f"A Entropia Cruzada (Log Loss) do Modelo K2 é {logloss_k2:.4f}")
print(f"O AUC-ROC do Modelo K2 é {auc_k2:.4f}")

print("""
Analisando as métricas de Entropia Cruzada e AUC-ROC, podemos observar:

- **Entropia Cruzada (Log Loss)**: Avalia a confiança das previsões, penalizando erros de forma mais intensa.
- **AUC-ROC**: Mede a capacidade do modelo de distinguir entre as classes, útil para dados desbalanceados.

Comparando os dois modelos, a análise das métricas indica possíveis áreas de melhoria, como:

- **Incluir mais variáveis**: Considerar outras características presentes no dataset.
- **Ajustar a discretização**: Experimentar diferentes formas de discretizar variáveis contínuas.
- **Experimentar outros algoritmos**: Como o PC Algorithm ou métodos híbridos.
""")


# In[255]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Calcular as curvas ROC
fpr_bic, tpr_bic, _ = roc_curve(test_data['Survived'], probabilities_bic)
roc_auc_bic = auc(fpr_bic, tpr_bic)

fpr_k2, tpr_k2, _ = roc_curve(test_data['Survived'], probabilities_k2)
roc_auc_k2 = auc(fpr_k2, tpr_k2)

# Plotar as curvas
plt.figure()
plt.plot(fpr_bic, tpr_bic, color='blue', lw=2, label=f'Modelo BIC (AUC = {roc_auc_bic:.2f})')
plt.plot(fpr_k2, tpr_k2, color='red', lw=2, label=f'Modelo K2 (AUC = {roc_auc_k2:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC dos Modelos BIC e K2')
plt.legend(loc='lower right')
plt.show()


# In[ ]:


from nbconvert import PythonExporter
import nbformat

# Caminho para o notebook Jupyter
notebook_path = 'C:\\programação\\projeto de IA\\versão2.ipynb'
# Caminho para salvar o código combinado
output_path = 'codigo_combinado.py'

# Carregar o notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = nbformat.read(f, as_version=4)

# Converter para Python
exporter = PythonExporter()
source_code, _ = exporter.from_notebook_node(notebook)

# Salvar o código combinado em um arquivo .py
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(source_code)

