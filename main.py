# ==============================================================================
# FASE 1: ANÁLISE EXPLORATÓRIA DE DADOS (AED)
# ==============================================================================
# 1. Qual base de dados foi escolhida e qual o objetivo de negócio do modelo?

"""
Importação das bibliotecas essenciais para manipulação de dados,
visualização e Machine Learning. Carregamento do arquivo 'credit_risk_dataset.csv'.
O primeiro passo da AED é entender a dimensionalidade da base (linhas e colunas), 
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
A análise descritiva revela outliers graves (ex: idades acima de 100 anos).
O gráfico de barras provou um forte desbalanceamento: a maioria dos clientes é adimplente.
Isso exige o uso de técnicas de reamostragem (SMOTE) para o modelo não ficar viciado.
O mapa de calor mostra correlação moderada entre o valor do empréstimo e a renda.
"""
#2. Quais insights visuais e estatísticos mudaram sua visão na Análise Exploratória (EDA)?


# ==============================================================================
# FASE 2: TRATAMENTO E LIMPEZA (DATA PREP)
# ==============================================================================
"""
"Limpeza: Primeiro são removidas as linhas duplicadas. Depois, identificadas
as colunas com valores nulos. Os nulos são tratados utilizando a MEDIANA 
em vez da média. Essa decisão é justificada estatisticamente porque variáveis como 
renda e taxa de juros possuem distribuições assimétricas e outliers; a média seria 
distorcida por esses valores extremos, enquanto a mediana permanece robusta."
"""

# 1. Remoção de Duplicadas
duplicados_antes = df.duplicated().sum()
df = df.drop_duplicates()
print(f"Linhas duplicadas removidas: {duplicados_antes}\n")

# 2. Identificação e Imputação de Nulos
print("--- VALORES NULOS POR COLUNA ---")
print(df.isnull().sum(), "\n")

# Imputação por Mediana (Justificativa técnica: mitigar o impacto de outliers)
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# 3. Tratamento de Outliers (Tratamento via Clipping/Capagem)
# Decisão: Limitaremos a idade máxima em 100 anos e o tempo de trabalho em 60 anos,
# pois o KNN é sensível a distâncias euclidianas geradas por outliers extremos.
df = df[df['person_age'] <= 100]
df = df[df['person_emp_length'] <= 60]

#3. Como você tratou os nulos e os outliers, considerando os impactos específicos no KNN e na Árvore?

# ==============================================================================
# FASE 3: FEATURE ENGINEERING (COLUNA CALCULADA)
# ==============================================================================
"""
"Criação da nova feature para o setor financeiro: 'comprometimento_renda'.
Ela representa a porcentagem da renda anual do cliente que o empréstimo solicitado 
ocupa. O cálculo é feito de forma segura após o tratamento de nulos para evitar 
erros de divisão por zero ou valores indeterminados (NaN)."
"""

df['comprometimento_renda'] = (df['loan_amnt'] / df['person_income']) * 100
print("Nova coluna 'comprometimento_renda' criada com sucesso.\n")

# ==============================================================================
# FASE 4: SEPARAÇÃO, BALANCEAMENTO E ESCALONAMENTO SEGURO
# ==============================================================================
"""
"Aplicação da Regra de Ouro da Ciência de Dados. Primeiro, as 
variáveis categóricas são transformadas em numéricas via One-Hot Encoding. Depois, separo a base em X e y 
e realizo o split de Treino e Teste (80/20) com o parâmetro 'stratify', garantindo 
a mesma proporção de inadimplentes em ambas as partições. 
Para evitar o vazamento de dados (Data Leakage), aplico o SMOTE APENAS na base de treino.
Por fim, o escalonamento StandardScaler é aplicado exclusivamente nas variáveis contínuas 
para o KNN. A Árvore de Decisão será treinada com os dados originais, pois ela realiza 
cortes monotônicos e independe da escala dos dados."
"""

# 1. Encoding de Variáveis Categóricas
categorical_cols = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 2. Split de Dados (80% treino, 20% teste)
X = df_encoded.drop(columns=['loan_status'])
y = df_encoded['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Balanceamento de Classes (Apenas no TREINO)
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# 4. Escalonamento Seguro (Apenas variáveis contínuas e focado no KNN)
continuous_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'comprometimento_renda']

scaler = StandardScaler()
X_train_knn = X_train_bal.copy()
X_test_knn = X_test.copy()

X_train_knn[continuous_cols] = scaler.fit_transform(X_train_bal[continuous_cols])
X_test_knn[continuous_cols] = scaler.transform(X_test[continuous_cols])

# Para a árvore, usamos os dados balanceados, mas sem a necessidade de escalonamento.
X_train_tree = X_train_bal
X_test_tree = X_test

# ==============================================================================
# FASE 5: MODELAGEM, VALIDAÇÃO E DIAGNÓSTICO DE OVERFITTING
# ==============================================================================
"""
"Para encontrar o equilíbrio ideal e combater o Overfitting, os 
hiperparâmetros de ambos os modelos é variado. Testei o KNN com K variando entre 3, 5, 7 e 9. 
Para a Árvore de Decisão, testei as profundidades máximas de 3, 5, 7 e Ilimitada.
Monitorei a acurácia no Treino e no Teste simultaneamente para avaliar a capacidade 
de generalização dos modelos."
"""

print("--- OTIMIZAÇÃO DO KNN ---")
k_values = [3, 5, 7, 9]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_knn, y_train_bal)
    train_acc = knn.score(X_train_knn, y_train_bal)
    test_acc = knn.score(X_test_knn, y_test)
    print(f"KNN (K={k}) -> Acurácia Treino: {train_acc:.4f} | Teste: {test_acc:.4f}")

print("\n--- OTIMIZAÇÃO DA ÁRVORE DE DECISÃO ---")
depths = [3, 5, 7, None]
for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=42)
    tree.fit(X_train_tree, y_train_bal)
    train_acc = tree.score(X_train_tree, y_train_bal)
    test_acc = tree.score(X_test_tree, y_test)
    print(f"Árvore (Profundidade={d}) -> Acurácia Treino: {train_acc:.4f} | Teste: {test_acc:.4f}")

#4. Como você identificou e evitou a ocorrência de overfitting ao mudar os parâmetros do KNN (K) e da Árvore (max_depth)?

# ==============================================================================
# FASE 6: AVALIAÇÃO E VEREDITO DE NEGÓCIOS
# ==============================================================================
"""
"Selecionei as duas melhores configurações baseadas no equilíbrio entre treino e teste 
para evitar overfitting: KNN com K=9 e Árvore com max_depth=7. 
Agora exibo o Classification Report e a Matriz de Confusão de cada um para 
tomar a decisão final."
"""

best_knn = KNeighborsClassifier(n_neighbors=9)
best_knn.fit(X_train_knn, y_train_bal)
y_pred_knn = best_knn.predict(X_test_knn)

best_tree = DecisionTreeClassifier(max_depth=7, random_state=42)
best_tree.fit(X_train_tree, y_train_bal)
y_pred_tree = best_tree.predict(X_test_tree)

print("\n================ REPORT: KNN (K=9) ================")
print(classification_report(y_test, y_pred_knn))

print("================ REPORT: ÁRVORE (MAX_DEPTH=7) ================")
print(classification_report(y_test, y_pred_tree))

# Plotagem das Matrizes de Confusão
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Matriz de Confusão - KNN (K=9)')
ax[0].set_xlabel('Predito')
ax[0].set_ylabel('Real')

sns.heatmap(confusion_matrix(y_test, y_pred_tree), annot=True, fmt='d', cmap='Greens', ax=ax[1])
ax[1].set_title('Matriz de Confusão - Árvore (Profundidade=7)')
ax[1].set_xlabel('Predito')
ax[1].set_ylabel('Real')

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.show()

# 5. Olhando para a Matriz de Confusão, qual modelo você recomenda para a diretoria da empresa e por quê?