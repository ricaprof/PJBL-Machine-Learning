
# 🧠 Projeto PjBL - Modelos de Classificação e Regressão (Aprendizado de Máquina)

Este projeto foi desenvolvido como parte da disciplina de **Aprendizado de Máquina**, com o objetivo de aplicar e comparar diferentes algoritmos supervisionados nos contextos de **classificação** e **regressão**.  

O trabalho é composto por quatro partes (A, B, C e D), correspondendo a diferentes protocolos experimentais, utilizando dois datasets distintos — um voltado para **classificação** e outro para **regressão**.

---

## 🎯 Objetivos

- Compreender e aplicar algoritmos clássicos de **classificação** e **regressão**.  
- Avaliar o desempenho de diferentes indutores (modelos) sob os protocolos:
  - **Hold-out (65% treino / 35% teste)**  
  - **Validação cruzada (5 folds)**  
- Comparar métricas de desempenho (acurácia, F1-score, precisão, sensibilidade, especificidade, R², MSE, RMSE, MAE).  
- Desenvolver a capacidade de análise crítica sobre a performance e generalização dos modelos.

---

## 📊 Datasets Utilizados

### 1. **Student Performance (UCI Machine Learning Repository)**  
📂 [Link para o dataset](https://archive.ics.uci.edu/ml/datasets/student%2Bperformance)  

Este conjunto de dados contém informações sobre estudantes do ensino médio em Portugal, abrangendo fatores pessoais, sociais e acadêmicos que podem influenciar o desempenho escolar.  

**Principais características:**
- Contém dois arquivos: `student-mat.csv` (matemática) e `student-por.csv` (português).  
- Variáveis incluem idade, gênero, tempo de estudo, absenteísmo, nível educacional dos pais, entre outros.  
- A variável alvo (`G3`) representa a **nota final** do aluno (0 a 20).  

**Uso neste projeto:**  
- **Problema de Regressão:** prever a nota final (`G3`) com base nas demais características dos estudantes.  

---

### 2. **Wine Quality (UCI Machine Learning Repository)**  
📂 [Link para o dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)  

O dataset contém dados físico-químicos de amostras de vinho tinto e branco, coletados para estudar a relação entre as propriedades químicas e a qualidade do vinho avaliada por especialistas.  

**Principais características:**
- Dois arquivos: `winequality-red.csv` e `winequality-white.csv`.  
- Atributos incluem acidez, teor alcoólico, pH, densidade e açúcar residual.  
- A variável alvo (`quality`) varia de 0 a 10.  

**Uso neste projeto:**  
- **Problema de Classificação:** prever a **qualidade do vinho** a partir de suas propriedades físico-químicas.  

---

## ⚙️ Etapas do Projeto

O projeto será dividido em quatro partes:

| Parte | Tarefa | Protocolo | Tipo |
|:--|:--|:--|:--|
| A | Classificação | Hold-out (65/35) | Wine Quality |
| B | Classificação | Validação cruzada (5 folds) | Wine Quality |
| C | Regressão | Hold-out (65/35) | Student Performance |
| D | Regressão | Validação cruzada (5 folds) | Student Performance |

Cada parte avaliará os seguintes algoritmos:
- **KNN**
- **Decision Tree**
- **Naive Bayes**
- **MLP**
- **SVM**
- **Random Forest**
- **Bagging**
- **Boosting**
- **Stacking**
- **Blending**
- **Ensemble geral (média dos modelos)**

---

## 👩‍💻 Equipe

Projeto desenvolvido por alunos da disciplina de **Aprendizado de Máquina (PjBL)** — 2025/2
- Crystofer Samuel
- Murilo Pedrazzani
- Ricardo Vinicius
- **Professor:** Joelton Deonei Gotz

