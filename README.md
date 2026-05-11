 
# ✍️ Reconhecimento de Letras com IA (EMNIST)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Status](https://img.shields.io/badge/Status-Concluído-success)
![Machine Learning](https://img.shields.io/badge/ML-EMNIST-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Interface-green)

---

## 📌 Sobre o projeto

Este projeto implementa um sistema de **reconhecimento de caracteres manuscritos em tempo real**.

O usuário desenha uma letra com o mouse e um modelo de Machine Learning é capaz de:

* 🔤 Identificar qual letra foi desenhada
* 🔠 Diferenciar entre **maiúscula e minúscula**
* 📊 Exibir o nível de confiança da previsão

O modelo foi treinado com o dataset **EMNIST (byclass)**, que contém **62 classes** (números + letras maiúsculas + minúsculas).

---

## 🎥 Demonstração

![Demonstração do projeto](assets/demo1.gif)

---

## 🧠 Tecnologias utilizadas

* **Python**
* **OpenCV**
* **NumPy**
* **Scikit-learn / TensorFlow**
* **EMNIST Dataset**

---

## 🚀 Funcionalidades

* ✔ Interface interativa para desenhar letras
* ✔ Reconhecimento em tempo real
* ✔ Classificação de maiúscula e minúscula
* ✔ Exibição da confiança (%)
* ✔ Pré-processamento consistente com o treino
* ✔ Estrutura pronta para evolução (CNN, Web, etc.)

---

## 🎮 Como usar

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Execute o projeto

```bash
primeiro:
python modelo.py
depois:
python interface_previsao.py
```

### 4. Utilização

* Desenhe uma letra com o mouse
* Pressione **P** para prever
* Pressione **C** para limpar a tela

---

## 🧪 Exemplo de saída

```
Letra: A
Tipo: Maiúscula
Confiança: 96.45%
```

---

## 🧠 Como funciona

### 🔹 1. Entrada do usuário

O usuário desenha uma letra em uma tela (canvas)

### 🔹 2. Pré-processamento

* Redimensionamento para **28x28**
* Normalização (**0–1**)
* Ajuste para o padrão do modelo

### 🔹 3. Predição

O modelo retorna probabilidades para **62 classes**

### 🔹 4. Pós-processamento

* Conversão de índice → caractere
* Verificação:

  * `isupper()` → Maiúscula
  * `islower()` → Minúscula

---

## ⚠️ Observações importantes

O pré-processamento deve ser **idêntico ao treino**.

Diferenças como:

* rotação
* escala
* espessura do traço

podem reduzir a precisão.

---

## 📈 Melhorias futuras

* 🔥 Implementar CNN (maior precisão)
* 📊 Mostrar Top 3 previsões
* ✍️ Criar dataset próprio com desenhos reais
* 🌐 Versão web (Flask ou Streamlit)
* 📱 Transformar em aplicação mobile
* 🎯 Melhorar centralização da letra

---

## 💼 Aplicações

* Reconhecimento de escrita manual
* Sistemas educacionais
* Interfaces interativas com IA
* Base para projetos de visão computacional

---

## 📁 Estrutura do projeto

```
📂 projeto/
│
├── 📂 assets/
├── 📂 data/
├── modelo.py
├── interface_previsao.py
├── modelo_emnist.pkl
├── requirements.txt
└── README.md
```

---

## 🧑‍💻 Autor

**Ariane Silva**

---

## 👥 Integrantes

* Nome: XXXXX | RA: XXXXX
* Nome: XXXXX | RA: XXXXX

---

## 🚀 Como executar (Notebook)

### 1. Instalar dependências

```bash
pip install numpy scikit-learn tensorflow
```

### 2. Executar

```bash
interface_previsão.py
```

---

## 🔗 Repositório

https://github.com/Trabalhos-UNIP-IA/Identificador_maiuscula_minuscula

---

## 📌 Observações

Projeto desenvolvido para fins acadêmicos, seguindo boas práticas de versionamento com Git e GitHub.
