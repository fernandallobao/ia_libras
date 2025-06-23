import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

hands = mp.solutions.hands
Hands = hands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Classes que vamos identificar
classes = ['A', 'E', 'I', 'O', 'U']

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Erro: Não foi possível capturar a imagem da câmera!")
        break

    frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Hands.process(frameRGB)
    handPoints = results.multi_hand_landmarks
    h, w, _ = img.shape

    if handPoints:
        for points in handPoints:
            mpDraw.draw_landmarks(img, points, hands.HAND_CONNECTIONS)
            pontos = []

            for id, cord in enumerate(points.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)
                cv2.circle(img, (cx, cy), 4, (255, 0, 0), -1)
                pontos.append((cx, cy))

            if len(pontos) == 21:
                # Calculando as distâncias dos dedos (em Y ou X, conforme o dedo)
                distPolegar = abs(pontos[4][0] - pontos[3][0])  # lateral (X)
                distIndicador = pontos[6][1] - pontos[8][1]
                distMedio = pontos[10][1] - pontos[12][1]
                distAnelar = pontos[14][1] - pontos[16][1]
                distMinimo = pontos[18][1] - pontos[20][1]

                letra = None

                # Regras com base nas distâncias
                if (distPolegar < 30 and
                    distIndicador < 10 and
                    distMedio < 10 and
                    distAnelar < 10 and
                    distMinimo < 10):
                    letra = 'A'

                elif (distPolegar < 30 and
                      10 <= distIndicador <= 20 and
                      10 <= distMedio <= 20 and
                      10 <= distAnelar <= 20 and
                      10 <= distMinimo <= 20):
                    letra = 'E'

                elif (distPolegar < 30 and
                      distIndicador < 10 and
                      distMedio < 10 and
                      distAnelar < 10 and
                      distMinimo > 20):
                    letra = 'I'

                elif (distPolegar > 90 and
                      distIndicador < 20 and
                      distMedio < 20 and
                      distAnelar < 20 and
                      distMinimo < 20):
                    letra = 'O'

                elif (distPolegar > 30 and
                      distIndicador < 20 and
                      distMedio < 20 and
                      distAnelar < 10 and
                      distMinimo > 10):
                    letra = 'U'

                if letra:
                    cv2.putText(img, f'Letra: {letra}', (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    print(f"Letra detectada: {letra}")

    cv2.imshow('Detector de Libras', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
