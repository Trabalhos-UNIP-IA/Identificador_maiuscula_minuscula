import sys
import subprocess
import importlib.util

def instalar_se_precisar(pacote, nome_import=None):
    nome_import = nome_import or pacote
    if importlib.util.find_spec(nome_import) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pacote])
instalar_se_precisar("opencv-python", "cv2")
instalar_se_precisar("scikit-image", "skimage")
instalar_se_precisar("numpy","numpy")
instalar_se_precisar("gtts", "gtts")
instalar_se_precisar("pygame", "pygame")
instalar_se_precisar("joblib", "joblib")

import cv2
import numpy as np
import string
import joblib # Recomendado para carregar o modelo treinado
from gtts import gTTS
from pygame import mixer
import tempfile

# 🎯 mapa de classes (EMNIST byclass: 0-9, A-Z, a-z)
classes = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
# engine.setProperty('rate', 150)  # Velocidade da fala
# 🧠 Carregar o modelo (certifique-se de que o arquivo existe após o treino)
mixer.init()  # Inicializa o mixer para áudio
palavra=[]

try:
    # Se você salvou com joblib no script anterior:
    model = joblib.load('modelo_emnist.pkl')
except:
    # Caso ainda esteja importando do script modelo.py diretamente:
    from modelo import model

# 🖌️ canvas
canvas = np.zeros((280, 280), dtype=np.uint8)
drawing = False

def draw(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), 12, 255, -1) # Aumentei um pouco o pincel
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        
def tipo_letra(idx):
    if idx < 10:
        return "Número"
    elif idx < 36:
        return "Letra Maiúscula"
    else:
        return "Letra Minúscula"
import os
from gtts import gTTS
from pygame import mixer

# inicializa o mixer só uma vez (fora da função, idealmente)
mixer.init()
def criar_audio_google(texto):
    # cria arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        caminho = fp.name

    # gera áudio
    tts = gTTS(text=texto, lang='pt-br')
    tts.save(caminho)

    # toca
    mixer.music.load(caminho)
    mixer.music.play()

def falar_google(idx):
    
    if idx < 10:
        texto = f"Você desenhou o número {classes[idx]}"
    elif idx < 36:
        texto = f"Você desenhou a letra {classes[idx]}  maiúscula"
    else:
        texto = f"Você desenhou a letra {classes[idx]} minúscula"

    # cria arquivo temporário
    criar_audio_google(texto)

def criar_palavra(idx):
    if idx < 10:
        print("Número detectado, não adicionado à palavra.")
    else:
        palavra.append(classes[idx])
        
def falar_palavra():
    if not palavra:
        print("Nenhuma letra na palavra para falar!")
        return

    texto = f"A palavra formada é: {''.join(palavra)}"
    print(texto)
    criar_audio_google(texto)


        

cv2.namedWindow("Desenhe uma letra")
cv2.setMouseCallback("Desenhe uma letra", draw)

print("Desenhe com o mouse")
print("P = prever | C = limpar | ESC = sair | F = falar palavra | R = resetar palavra")

while True:
    cv2.imshow("Desenhe uma letra", canvas)
    key = cv2.waitKey(1)

    if key == ord('p'):
        # 1. Encontrar o desenho e centralizar (Bounding Box)
        coords = cv2.findNonZero(canvas)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # Criar um ROI quadrado para não distorcer a letra
            side = max(w, h)
            roi = np.zeros((side, side), dtype=np.uint8)
            start_x = (side - w) // 2
            start_y = (side - h) // 2
            roi[start_y:start_y+h, start_x:start_x+w] = canvas[y:y+h, x:x+w]
            
            # 2. Adicionar margem e redimensionar
            roi = cv2.copyMakeBorder(roi, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=0)
            img = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(canvas, (28, 28))
            

        # img_pre = np.rot90(img, k=1)
        # img_pre = np.flipud(img_pre)
        img_pre = img.copy()

        img_input = img_pre.astype('float32') / 255.0
        img_input = img_input.reshape(1, 784) 

        # 4. Predição
        # O MLPClassifier do sklearn já retorna a classe no .predict()
        # Mas para ver a porcentagem de confiança, usamos .predict_proba()
        probabilidades = model.predict_proba(img_input)[0]
        pred_idx = np.argmax(probabilidades)

        confianca = probabilidades[pred_idx] * 100
        if confianca < 80:  # Apenas confirma a predição se a confiança for alta o suficiente
                print(f"Falha ao reconhecer , só foi possivel reconhecer ({confianca:.2f}%), tente desenhar novamente.")
                criar_audio_google(f"Falha ao reconhecer , só foi possivel reconhecer ({confianca:.2f}%), tente desenhar novamente.")
        else:
            print("-" * 30)
            print(f"Letra/Número: {classes[pred_idx]}")
            print(f"Tipo: {tipo_letra(pred_idx)}")
            falar_google(pred_idx)
            print(f"Reconhecimento: {confianca:.2f}%")
            criar_palavra(pred_idx)
            print(f"Palavra formada: {''.join(palavra)}")
        # Visualização para Debug (ajuda a ver se a letra não está deformada)
        cv2.imshow("O que o modelo ve", cv2.resize(img_pre, (140, 140)))
        
    if key == ord('f'):
        if 'pred_idx' in locals():
            nova_letra = classes[pred_idx]

        # 🔥 só adiciona se for diferente da última
            if len(palavra) == 0 or palavra[-1] != nova_letra:
                 palavra.append(nova_letra)

            texto = f"Palavra formada: {''.join(palavra)}"
            print(texto)

            falar_palavra()
        else:
            print("Nenhuma letra foi detectada ainda!")

    if key == ord('r'):
        palavra.clear()
        print("Palavra resetada.")

    elif key == ord('c'):
        canvas[:] = 0

    elif key == 27:
        break

cv2.destroyAllWindows()