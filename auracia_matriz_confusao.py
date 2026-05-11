import numpy as np
import joblib
import string
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================
# 1. CONFIGURAÇÕES E PREPARAÇÃO
# ==========================================
# Mapa de classes (0-9, A-Z, a-z)
classes = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)

def preprocess_images(data):
    """Aplica rotação e flip para corrigir a orientação do EMNIST"""
    print("Processando imagens...")
    data_fixed = np.array([np.fliplr(np.rot90(img, k=3)) for img in data])
    # Normalizar e Achatar (Flatten)
    data_fixed = data_fixed.astype("float32") / 255.0
    return data_fixed.reshape(len(data_fixed), -1)

# ==========================================
# 2. TREINAMENTO DO MODELO
# ==========================================
print("--- FASE DE TREINAMENTO ---")
train_data = datasets.EMNIST(root="./data", split="byclass", train=True, download=True)

X_train = train_data.data.numpy()[:50000]
y_train = train_data.targets.numpy()[:50000]

X_train_final = preprocess_images(X_train)

print(f"Treinando MLP com {len(X_train_final)} amostras...")
model = MLPClassifier(
    hidden_layer_sizes=(256, 128), 
    max_iter=50, 
    verbose=True, 
    random_state=42
)

model.fit(X_train_final, y_train)
joblib.dump(model, "modelo_emnist.pkl")
print("✅ Modelo salvo com sucesso!")

# ==========================================
# 3. AVALIAÇÃO DO MODELO
# ==========================================
print("\n--- FASE DE AVALIAÇÃO ---")
test_data = datasets.EMNIST(root="./data", split="byclass", train=False, download=True)

X_test = test_data.data.numpy()[:10000]
y_test = test_data.targets.numpy()[:10000]

X_test_final = preprocess_images(X_test)

print("Gerando predições...")
y_pred = model.predict(X_test_final)

# Métricas de Texto
acuracia = accuracy_score(y_test, y_pred)
print(f"\n🎯 ACURÁCIA FINAL: {acuracia * 100:.2f}%")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=classes))

# ==========================================
# 4. VISUALIZAÇÃO DA MATRIZ DE CONFUSÃO
# ==========================================
print("Gerando gráfico da Matriz de Confusão...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(15, 10))
sns.heatmap(
    cm, 
    annot=False, 
    cmap="Blues", 
    xticklabels=classes, 
    yticklabels=classes
)

plt.title(f"Matriz de Confusão EMNIST - Acurácia: {acuracia * 100:.2f}%", fontsize=18)
plt.xlabel("Predito (O que o modelo achou)", fontsize=14)
plt.ylabel("Real (O que realmente é)", fontsize=14)
plt.show()