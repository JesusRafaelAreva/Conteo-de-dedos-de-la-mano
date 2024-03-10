# IMPORTAR RECURSOS NECESARIOS.
import cv2
import mediapipe as mp

# INICIAR SISTEMA DE DETECCIÓN.
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# INICIAR CAMARA
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5, max_num_hands=2) as hands:
    while cap.isOpened():
        # COMPROBAR ENTRADA
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # CONVERTIR IMAGEN A RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        # DIBUJAR PUNTOS DE DETECCIÓN Y CONTAR DEDOS
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Obtener las coordenadas de los puntos de referencia de la mano
                landmarks = hand_landmarks.landmark

                # Coordenadas de los puntos relevantes para contar dedos
                thumb_tip = landmarks[4]  # Pulgar
                index_tip = landmarks[8]  # Índice
                middle_tip = landmarks[12]  # Medio
                ring_tip = landmarks[16]  # Anular
                pinky_tip = landmarks[20]  # Meñique

                # Determinar si los dedos están extendidos
                fingers_extended = [1 if landmark.y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y else 0
                                    for landmark in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]]

                # Contar dedos extendidos
                finger_count = sum(fingers_extended)

                # Mostrar el número de dedos en la pantalla
                cv2.putText(image, f'Dedos: {finger_count}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Dibujar círculos en las puntas de los dedos
                for landmark in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]:
                    x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                    cv2.circle(image, (x, y), 10, (0, 0, 255), -1)

                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
