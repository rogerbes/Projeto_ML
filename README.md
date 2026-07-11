# Pipeline Preditivo de Risco de Crédito 💳

Este repositório contém o desenvolvimento de um pipeline completo de Machine Learning focado no setor financeiro. O objetivo do projeto é construir um modelo preditivo robusto capaz de identificar potenciais inadimplências, otimizando a tomada de decisão em concessões de crédito bancário.

---

## 📌 1. Contexto e Objetivo de Negócio

No mercado de crédito, o equilíbrio entre a expansão da carteira comercial e o controle da inadimplência é vital. O objetivo deste projeto é prever se um cliente se tornará inadimplente (`loan_status = 1`) ou se pagará o empréstimo em dia (`loan_status = 0`).

### Impacto das Falhas do Modelo (Matriz de Confusão)
*   **Falso Positivo:** O modelo classifica um bom pagador como "Risco de Calote". Isso gera um **custo de oportunidade**, pois o banco perde um cliente potencialmente lucrativo devido a um alarme falso.
*   **Falso Negativo:** O modelo classifica um mau pagador como "Seguro". Este é o **erro mais destrutivo e caro**, pois resulta na perda direta do capital emprestado (inadimplência pura).

---

## 📋 2. Dicionário de Dados

A base utilizada (`credit_risk_dataset.csv`) possui as seguintes variáveis estruturadas:

| Variável | Tipo de Dado | Descrição |
| :--- | :--- | :--- |
| `person_age` | Contínua (Int) | Idade do solicitante do crédito |
| `person_income` | Contínua (Float) | Renda anual informada pelo cliente |
| `person_home_ownership` | Categórica | Tipo de moradia (Alugada, Própria, Hipoteca, etc.) |
| `person_emp_length` | Contínua (Float) | Tempo de trabalho registrado (em anos) |
| `loan_intent` | Categórica | Intuito/Finalidade do empréstimo |
| `loan_grade` | Categórica | Nota de risco atribuída ao perfil do cliente |
| `loan_amnt` | Contínua (Float) | Valor total solicitado para o empréstimo |
| `loan_int_rate` | Contínua (Float) | Taxa de juros do empréstimo solicitado |
| `loan_percent_income` | Contínua (Float) | Percentual que o empréstimo representa na renda informada |
| `cb_person_default_on_file` | Categórica | Histórico prévio de inadimplência (Sim/Não) |
| `cb_person_cred_hist_length`| Contínua (Int) | Tempo de histórico de crédito ativo (em anos) |
| `comprometimento_renda` | Contínua (Calculada) | Nova feature: Porcentagem da renda anual ocupada pelo empréstimo |
| `loan_status` | Binária (Alvo) | **Variável Alvo:** `0` = Adimplente | `1` = Inadimplente |

---

## 🚀 3. Pipeline de Desenvolvimento Técnico

O projeto foi dividido nas seguintes fases metodológicas:

### Fase 1: Análise Exploratória de Dados (EDA)
*   Análise descritiva estatística e dimensionalidade das variáveis via `describe()`.
*   Identificação visual de um **desbalanceamento de classes** na variável alvo (predomínio de adimplentes).
*   Detecção de outliers (ex: idades acima de 100 anos e tempo de trabalho impossível).

### Fase 2: Tratamento e Limpeza (Data Prep)
*   **Duplicadas:** Remoção de registros repetidos para mitigar redundâncias.
*   **Valores Nulos:** Imputação realizada com a **Mediana** nas colunas numéricas, decisão fundamentada pelo fato de variáveis como renda e taxa de juros apresentarem alta assimetria e outliers que distorceriam uma média simples.
*   **Outliers:** Aplicação de filtragem limitando a idade máxima em 100 anos e o tempo de trabalho em 60 anos, protegendo a matemática dos algoritmos.

### Fase 3: Feature Engineering
*   Criação da variável calculada `comprometimento_renda` ($(\text{loan\_amnt} / \text{person\_income}) \times 100$). O cálculo foi executado estritamente após a limpeza de nulos para mitigar erros de divisão por zero ou infestações de valores `NaN`.

### Fase 4: Separação, Balanceamento e Escalonamento Seguro
*   **Encoding:** Dumificação de variáveis textuais via One-Hot Encoding (`pd.get_dummies`) com `drop_first=True` para evitar a colinearidade.
*   **Split Seguro:** Divisão em 80% Treino e 20% Teste utilizando amostragem estratificada (`stratify=y`) para manter a representação proporcional das classes.
*   **Balanceamento:** Aplicação da técnica **SMOTE** *apenas e estritamente na base de treino*, preservando a base de teste intacta para evitar o vazamento de dados (Data Leakage).
*   **Escalonamento:** Aplicação do `StandardScaler` exclusivamente nas variáveis contínuas destinadas ao modelo KNN. O modelo de Árvore de Decisão foi treinado sem necessidade de escala, dada a sua robustez inerente baseada em cortes monotônicos.

### Fase 5: Validação e Combate ao Overfitting
Realizamos múltiplos experimentos monitorando simultaneamente a performance nas bases de Treino e Teste:
*   **KNN:** Otimizado avaliando os vizinhos $K \in [3, 5, 7, 9]$.
*   **Árvore de Decisão:** Otimizada avaliando profundidades máximas (`max_depth`) em $[3, 5, 7, \text{None}]$. A profundidade ilimitada gerou overfitting clássico (100% de acerto no treino com degradação severa no teste). O equilíbrio foi estabelecido com `max_depth=7`.

---

## 📈 4. Resumo Executivo e Veredito de Negócios

Após a validação cruzada dos hiperparâmetros, as duas melhores configurações selecionadas para o duelo final foram o **KNN (K=9)** e a **Árvore de Decisão (Profundidade=7)**.

### Análise das Matrizes de Confusão
*   **Eficiência de Risco (Minimização de Calotes):** A Árvore de Decisão deixou passar apenas 453 Falsos Negativos contra 462 do KNN, entregando um **Recall superior para a classe de inadimplência (Classe 1)**.
*   **Eficiência Comercial (Aproveitamento de Clientes):** O KNN bloqueou erroneamente 402 clientes saudáveis (Falsos Positivos). A Árvore de Decisão gerou apenas 155 Falsos Positivos, minimizando drasticamente a perda por custo de oportunidade.

### 🏆 Veredito Final
O modelo recomendado para ser colocado em ambiente produtivo é a **Árvore de Decisão (max_depth=7)**. Estatisticamente e financeiramente, o modelo se provou superior: ele blinda o caixa da instituição capturando mais inadimplentes (maior Recall), ao mesmo tempo em que preserva a experiência do cliente e a receita comercial ao diminuir em mais de 60% os alarmes falsos causados pelo KNN.

---

## 🛠️ Tecnologias Utilizadas

*   **Python 3.x**
*   **Pandas** e **NumPy** (Manipulação de Dados)
*   **Matplotlib** e **Seaborn** (Visualizações Gráficas)
*   **Scikit-Learn** (Módulos de ML, Pré-processamento e Métricas)
*   **Imbalanced-Learn** (Aplicação do SMOTE)

---

## 🎬 5. Apresentação em Vídeo

As explicações do pipeline de código, justificativas estatísticas e a defesa do veredito de negócios encontram-se disponíveis no link de vídeo abaixo:
*   [Assista à Apresentação do Projeto no Drive Senai](https://drive.google.com/drive/folders/1HpbF0j9gBSvrLUJsCwAz8sQDoevIXxiY?usp=drive_link) *(Substitua este link pelo seu link de gravação)*