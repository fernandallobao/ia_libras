import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

class DetectorLibras:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(max_num_hands=1)
        self.classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
                       'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
                       'W', 'X', 'Y', 'Z']
        self.model = load_model('./analise/keras_model.h5')  # ESPERAR P ARQUIVO CERTO
        self.data = np.ndarray(shape=(1, 64, 64, 3), dtype=np.float32) #Prepara um array numpy no formato que o modelo espera (1 imagem 224x224 RGB)
        self.letra_detectada = None #Variável para armazenar a última letra detectada

    def detectar_letra(self, img):
        """
        Detecta a letra de Libras na imagem fornecida
        Retorna a imagem com anotações e a letra detectada
        """
        frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frameRGB)
        handsPoints = results.multi_hand_landmarks
        h, w, _ = img.shape
        self.letra_detectada = None

        if handsPoints is not None:
            for hand in handsPoints:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                
                for lm in hand.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_max, x_min = max(x, x_max), min(x, x_min)
                    y_max, y_min = max(y, y_max), min(y, y_min)
                
                cv2.rectangle(img, (x_min-50, y_min-50), (x_max+50, y_max+50), (0, 255, 0), 2)

                try:
                    imgCrop = img[y_min-50:y_max+50, x_min-50:x_max+50]
                    imgCrop = cv2.resize(imgCrop, (224, 224))
                    imgArray = np.asarray(imgCrop)
                    normalized_image_array = (imgArray.astype(np.float32) / 127.0) - 1
                    self.data[0] = normalized_image_array
                    prediction = self.model.predict(self.data)
                    indexVal = np.argmax(prediction)
                    self.letra_detectada = self.classes[indexVal]
                    
                    cv2.putText(img, self.letra_detectada, (x_min-50, y_min-65), 
                               cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5)
                except Exception as e:
                    print(f"Erro ao processar imagem: {e}")
                    continue

        return img, self.letra_detectada

    def get_letra_detectada(self):
        """Retorna a última letra detectada"""
        return self.letra_detectada