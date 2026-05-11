# Identificador de Letras Maiúsculas e Minúsculas

Projeto desenvolvido para a disciplina de Inteligência Artificial da UNIP com o objetivo de criar um modelo utilizando Redes Neurais capaz de identificar letras maiúsculas e minúsculas.

---

# 📌 Objetivo

Desenvolver um processo utilizando Redes Neurais para:

* Identificar letras maiúsculas e minúsculas;
* Gerar dados de entrada aleatórios;
* Aplicar o modelo treinado;
* Avaliar métricas de desempenho;
* Explicar a lógica utilizada durante o processamento.

---

# 👥 Integrantes do Grupo

| Nome             | RA      |
| ---------------- | ------- |
| Sydhiney Silva   | G75EJI5 |
| Eduardo Theodoro | R153FJ3 |
| Ariane Veras     | R197123 |

---

# 🧠 Tecnologias Utilizadas

* Python 3
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-Learn
* OpenCV
* Jupyter Notebook
* GitHub

---

# 📂 Estrutura do Projeto

```bash
Identificador_maiuscula_minuscula/
│
├── notebooks/
│   ├── treinamento_modelo.ipynb
│   ├── testes_modelo.ipynb
│
├── dataset/
│   ├── maiusculas/
│   ├── minusculas/
│
├── resultados/
│   ├── graficos/
│   ├── metricas/
│
├── src/
│   ├── preprocessamento.py
│   ├── modelo.py
│   ├── previsao.py
│
├── requirements.txt
├── README.md
└── .gitignore
```
---

# ⚙️ Funcionamento do Projeto

O projeto segue as seguintes etapas:

## 1️⃣ Geração dos Inputs

Foi criado um conjunto de imagens contendo letras maiúsculas e minúsculas de forma aleatória.

Exemplos:

* A, B, C, D...
* a, b, c, d...

As imagens são convertidas para escala de cinza e redimensionadas para um tamanho padrão, facilitando o treinamento da rede neural.

---

# 🔄 Pré-processamento

Antes do treinamento, os dados passam pelas seguintes etapas:

* Conversão para escala de cinza;
* Redimensionamento;
* Normalização dos pixels;
* Separação entre treino e teste.

---

# 🧱 Estrutura da Rede Neural

Foi utilizada uma rede neural sequencial composta por:

```python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

---

# 📖 Explicação dos Blocos

## 🔹 Flatten

Transforma a imagem 2D em um vetor 1D.

## 🔹 Dense(128)

Camada neural responsável por aprender padrões mais complexos das imagens.

## 🔹 Dense(64)

Camada intermediária para melhorar o reconhecimento dos padrões.

## 🔹 Dense(1)

Camada de saída:

* Resultado próximo de 0 → letra minúscula;
* Resultado próximo de 1 → letra maiúscula.

A função sigmoid foi utilizada para classificação binária.

---

# 🎲 Geração Aleatória de Inputs

Exemplo de geração aleatória:

```python
import random
import string

letra = random.choice(string.ascii_letters)
print(letra)
```

Esse processo foi utilizado para criar entradas diferentes durante os testes do modelo.

---

# 🚀 Aplicação do Modelo

Após o treinamento, o modelo recebe uma imagem como entrada.

Fluxo do processo:

1. Recebe a imagem;
2. Realiza o pré-processamento;
3. Converte os pixels em números;
4. Envia os dados para a rede neural;
5. O modelo realiza a previsão;
6. Retorna se a letra é maiúscula ou minúscula.

---

# 📊 Métricas de Desempenho

As métricas utilizadas para avaliação do modelo foram:

* Accuracy;
* Loss;
* Matriz de Confusão;
* Precisão (Precision);
* Recall.

Exemplo de métricas obtidas após o treinamento:

| Métrica  | Resultado |
| -------- | --------- |
| Accuracy | 98%       |
| Loss     | 0.04      |
| Precisão | 97%       |
| Recall   | 98%       |

---

# 📈 Exemplo de Treinamento

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'Precision', 'Recall']
)

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)
```

---

# 🧪 Resultado Final

O modelo apresentou uma alta taxa de acerto na identificação de letras maiúsculas e minúsculas, demonstrando que Redes Neurais podem ser aplicadas com eficiência em tarefas de classificação de imagens.

Os resultados obtidos mostraram boa capacidade de generalização do modelo durante os testes realizados com entradas aleatórias.

---

# 🔗 Repositório GitHub

Repositório do projeto:

https://github.com/Trabalhos-UNIP-IA/Identificador_maiuscula_minuscula.git

---

