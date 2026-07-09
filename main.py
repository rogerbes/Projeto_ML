# ==============================================================================
# FASE 1: ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
# ==============================================================================
"""
Importação das bibliotecas essenciais para manipulação de dados,
visualização e Machine Learning. Carregamento do arquivo 'credit_risk_dataset.csv'.
O primeiro passo da EDA é entender a dimensionalidade da base (linhas e colunas), 
identificar os tipos de variáveis e analisar o sumário estatístico para detectar 
médias, quartis e possíveis anomalias nos dados."
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Ignorar avisos para manter o output limpo
import warnings
warnings.filterwarnings('ignore')

# Carregamento do arquivo local conforme solicitado
df = pd.read_csv('credit_risk_dataset.csv')

print("--- DIMENSÕES DA BASE ---")
print(f"Linhas: {df.shape[0]} | Colunas: {df.shape[1]}\n")

print("--- TIPOS DE DADOS ---")
print(df.dtypes, "\n")

print("--- SUMÁRIO ESTATÍSTICO DESCRITIVO ---")
print(df.describe(), "\n")

# GERAÇÃO DOS GRÁFICOS (Mínimo 3 exigidos)
plt.figure(figsize=(18, 5))

# Gráfico 1: Histograma de Idade
plt.subplot(1, 3, 1)
sns.histplot(df['person_age'], bins=30, kde=True, color='blue')
plt.title('Distribuição de Idades dos Clientes')
plt.xlabel('Idade')

# Gráfico 2: Desbalanceamento da Variável Alvo
plt.subplot(1, 3, 2)
sns.countplot(x='loan_status', data=df, palette='Set2')
plt.title('Desbalanceamento da Variável Alvo (loan_status)')
plt.xticks([0, 1], ['Adimplente (0)', 'Inadimplente (1)'])

# Gráfico 3: Mapa de Calor (Correlação de Pearson nas numéricas)
plt.subplot(1, 3, 3)
numeric_cols = df.select_dtypes(include=[np.number]).columns
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlação de Pearson')

plt.tight_layout()
plt.savefig('eda_plots.png')
plt.show()

"""
INTERPRETAÇÃO DA EDA (TOMADA DE DECISÃO):
A análise descritiva revelou outliers graves (ex: idades acima de 100 anos).
O gráfico de barras provou um forte desbalanceamento: a maioria dos clientes é adimplente.
Isso exige o uso de técnicas de reamostragem (SMOTE) para o modelo não ficar viciado.
O mapa de calor mostra correlação moderada entre o valor do empréstimo e a renda.
"""