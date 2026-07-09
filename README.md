# Pipeline Preditivo de Risco de Crédito

Este repositório contém a implementação completa de um pipeline de Machine Learning voltado para o setor financeiro, com o objetivo de prever o risco de inadimplência de clientes de uma instituição bancária.

## Contexto do Problema de Negócio
Para as instituições financeiras, conceder crédito de forma assertiva é vital para a saúde de caixa. O objetivo deste modelo é classificar os clientes em duas categorias:
* **Classe 0 (Adimplente):** Paga o empréstimo em dia.
* **Classe 1 (Inadimplente):** Risco de calote.

### Dicionário de Dados Atualizado
* `person_age`: Idade do cliente.
* `person_income`: Renda anual do cliente.
* `person_emp_length`: Tempo de permanência no emprego atual (em anos).
* `loan_amnt`: Valor do empréstimo solicitado.
* `loan_int_rate`: Taxa de juros do empréstimo.
* `loan_status`: Alvo (0 = Adimplente, 1 = Inadimplente).
* `comprometimento_renda` (Nova Coluna): Percentual de comprometimento da renda anual calculado pela fórmula: `(loan_amnt / person_income) * 100`.

---

## Como Executar o Projeto

### Pré-requisitos
Certifique-se de ter o Python 3.8+ instalado e as bibliotecas necessárias. Você pode instalá-las rodando:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

Adicione o arquivo .csv da base de dados credit_risk_dataset.csv na variável
df = pd.read_csv('credit_risk_dataset.csv')

O arquivo deve estar no mesmo local de execuçao do main.py.

Durante a execução feche os graficos gerados para prosseguir com as demais etapas.