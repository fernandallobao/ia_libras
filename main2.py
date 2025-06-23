import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)

hands = mp.solutions.hands
Hands = hands.Hands(max_num_hands=1)
mpDwaw = mp.solutions.drawing_utils

# Classes do modelo
classes = ['A', 'E', 'I', 'O', 'U']

# Carregar o modelo
# model = load_model('./analise/modelo3.h5')

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Erro: Não foi possível capturar a imagem da câmera!")
        break

    frameRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = Hands.process(frameRGB)
    handPoints = results.multi_hand_landmarks
    h, w, _ = img.shape
    pontos = []

    if handPoints:
        for points in handPoints:
            mpDwaw.draw_landmarks(img, points, hands.HAND_CONNECTIONS)
            pontos = []  # precisa reiniciar os pontos por mão
            for id, cord in enumerate(points.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)
                cv2.circle(img, (cx, cy), 4, (255, 0, 0), -1)
                pontos.append((cx, cy))

            if len(pontos) == 21:
                # Calculando as distâncias dos dedos (em Y)
                distPolegar = abs(pontos[4][0] - pontos[3][0])
                distIndicador = pontos[6][1] - pontos[8][1]
                distMedio = pontos[10][1] - pontos[12][1]
                distAnelar = pontos[14][1] - pontos[16][1]
                distMinimo = pontos[18][1] - pontos[20][1]

                # Convertendo para valores binários (0=fechado, 1=aberto)
                polegar = 1 if distPolegar > 30 else 0
                indicador = 1 if distIndicador > 20 else 0
                medio = 1 if distMedio > 20 else 0
                anelar = 1 if distAnelar > 20 else 0
                minimo = 1 if distMinimo > 20 else 0

                dedos = [polegar, indicador, medio, anelar, minimo]

                letra = None
                
                if dedos == [0, 0, 0, 0, 0]:
                    letra = 'A'
                elif dedos == [0, 0.5, 0.5, 0.5, 0.5]:
                    letra = 'E'
                elif dedos == [0, 0, 0, 0, 1]:
                    letra = 'I'
                elif dedos == [0, 0, 0, 0, 0] and distPolegar > 50:
                    letra = 'O'
                elif dedos == [1, 0, 0, 0, 1]:
                    letra = 'U'

                if letra:
                    cv2.putText(img, f'Letra: {letra}', (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    print(f"Letra detectada: {letra}")


    cv2.imshow('Imagem',img)
    # cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

cap.release()
cv2.destroyAllWindows()