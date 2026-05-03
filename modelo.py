import numpy as np
from torchvision import datasets
from sklearn.neural_network import MLPClassifier
import joblib # Para salvar o modelo e não precisar treinar toda vez

# 📦 Carregar dataset EMNIST ByClass (62 classes: 0-9, A-Z, a-z)
print("Carregando dataset...")
train = datasets.EMNIST(root="./data", split="byclass", train=True, download=True)

X = train.data.numpy()
y = train.targets.numpy() # REMOVIDO o -1. ByClass já é 0-61.

# 🔄 Corrigir a orientação do EMNIST (O segredo do sucesso)
# O EMNIST original é transposto. Rot90(k=3) + Flip corrige para o que vemos no Canvas.
print("Processando imagens...")
X = np.array([np.fliplr(np.rot90(img, k=3)) for img in X])

# ⚡ Aumentamos um pouco o dataset para 50k para melhorar a precisão
X = X[:50000]
y = y[:50000]

# 🎯 Normalizar e Achatar
X = X.astype('float32') / 255.0
X = X.reshape(len(X), -1)

print(f"Treinando modelo com {len(X)} amostras...")

# 🧠 Modelo Robusto
# hidden_layer_sizes: (256, 128) ajuda a captar a diferença entre 'A' e 'a'
# max_iter: 50 é o mínimo para o modelo convergir decentemente
model = MLPClassifier(
    hidden_layer_sizes=(256, 128), 
    max_iter=50, 
    verbose=True, 
    random_state=42
)

model.fit(X, y)

# 💾 Salvar o modelo para usar na interface
joblib.dump(model, 'modelo_emnist.pkl')
print("Modelo pronto e salvo como 'modelo_emnist.pkl'!")