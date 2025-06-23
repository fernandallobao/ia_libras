import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

# Carrega o modelo IA treinado
model = tf.keras.models.load_model('./analise/modelo3.h5')

# Lista de classes previstas pelo seu modelo (ajuste conforme seu modelo)
classes = ['A', 'E', 'I', 'O', 'U']

# Inicializa o Mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Captura da câmera
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    h, w, _ = img.shape
    letra = None

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            pontos = [(int(lm.x * w), int(lm.y * h)) for lm in handLms.landmark]

            if len(pontos) == 21:
                # Bounding box da mão
                x_list = [pt[0] for pt in pontos]
                y_list = [pt[1] for pt in pontos]

                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)

                # Adiciona margem
                margin = 40
                x1 = max(x_min - margin, 0)
                y1 = max(y_min - margin, 0)
                x2 = min(x_max + margin, w)
                y2 = min(y_max + margin, h)

                # Recorta e prepara a imagem para o modelo
                hand_img = img[y1:y2, x1:x2]

                try:
                    hand_img = cv2.resize(hand_img, (224, 224))
                    hand_img = hand_img.astype('float32') / 255.0
                    hand_img = np.expand_dims(hand_img, axis=0)

                    prediction = model.predict(hand_img, verbose=0)
                    class_index = np.argmax(prediction)
                    letra = classes[class_index]

                except Exception as e:
                    print("Erro ao processar IA:", e)

    # Exibe o resultado
    if letra:
        cv2.putText(img, f'Letra: {letra}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        print(f"Letra detectada: {letra}")

    cv2.imshow("Detector por IA de Libras", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
