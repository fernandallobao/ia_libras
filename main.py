import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Configurações iniciais
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

# Configurações do MediaPipe
hands = mp.solutions.hands
Hands = hands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Classes do modelo
classes = ['A', 'E', 'I', 'O', 'U']

# Carregar o modelo
model = load_model('./analise/modelo3.h5')

# Preparar array de dados - MODIFICAÇÃO IMPORTANTE AQUI
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Erro: Não foi possível capturar a imagem da câmera!")
        break
        
    frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Hands.process(frameRGB)
    handsPoints = results.multi_hand_landmarks
    h, w, _ = img.shape
    pontos = []
    
    if handsPoints:
        for points in handsPoints:
            mpDraw.draw_landmarks(img, points, hands.HAND_CONNECTIONS)
            
            for id, cord in enumerate(points.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)
                cv2.circle(img, (cx, cy), 4, (255, 0, 0), -1)
                pontos.append((cx, cy))
                
            x_max = y_max = 0
            x_min = w
            y_min = h
            
            for lm in points.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_max, x_min = max(x, x_max), min(x, x_min)
                y_max, y_min = max(y, y_max), min(y, y_min)
                
            cv2.rectangle(img, (x_min-40, y_min-40), (x_max+50, y_max+50), (0, 255, 0), 2)
            
            # Ajuste das coordenadas para não sair da imagem
            x1, y1 = max(x_min - 40, 0), max(y_min - 40, 0)
            x2, y2 = min(x_max + 50, w), min(y_max + 50, h)
            
            if x2 - x1 > 0 and y2 - y1 > 0:
                try:
                    imgCrop = img[y1:y2, x1:x2]
                    imgCrop = cv2.resize(imgCrop, (224, 224))
                    
                    # Pré-processamento da imagem
                    imgArray = np.asarray(imgCrop) /255
                    # normalized_image_array = (imgArray.astype(np.float32) / 127.0) - 1
                    
                    # Verificação adicional do shape
                    # if normalized_image_array.shape != (224, 224, 3):
                    #     print(f"Shape incorreto: {normalized_image_array.shape}")
                    #     continue
                        
                    # data[0] = imgArray
                    imgArray = imgArray.reshape(1,224,224,3)
                    prediction = model.predict(imgArray, verbose=0)
                    
                    # Verificação da predição
                    if prediction.size == 0:
                        print("Erro: Nenhuma predição retornada!")
                        continue
                        
                    indexVal = np.argmax(prediction)
                    
                    if indexVal >= len(classes):
                        print(f"Índice inválido: {indexVal}")
                        continue
                        
                    letra = classes[indexVal]
                    print(f"Letra predita: {letra}")
                    
                    # Posicionamento do texto
                    text_x = max(x_min, 10)
                    text_y = max(y_min - 20, 20)
                    cv2.putText(img, letra, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)
                    
                except Exception as e:
                    print(f"Erro ao processar imagem: {str(e)}")
                    continue
    
    cv2.imshow('Reconhecimento de Libras', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()