import cv2
from cvzone.HandTrackingModule import HandDetector


# Inicializar la cámara
DdC = cv2.VideoCapture(0)

# Inicializar el Dt de manos con los mismos parámetros que tenías
Dt = HandDetector(maxHands=2, detectionCon=0.8, minTrackCon=0.5)
      # equivalente a min_tracking_confidence
CD=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),
    (11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
while True:
    # Leer el frame de la cámara
    success, img = DdC.read()
    if not success:
        break

    # Detectar las manos
    hands, img = Dt.findHands(img, draw=False)  # No dibujamos aquí para hacerlo manual

    # Si se detectan manos
    if hands:
        hand = hands[0]  # Tomamos la primera mano
        lmList = hand["lmList"]  # Lista de landmarks
        
        # Dibujar círculos en los puntos específicos (4 y 20)
        for id, lm in enumerate(lmList):
            cv2.circle(img, (lm[0], lm[1]), 10, (255,255,0), cv2.FILLED)
        for start, end in CD:
            cv2.line(img, (lmList[start][0], lmList[start][1]), (lmList[end][0], lmList[end][1]), (0,255,0), 3)
        
    # Mostrar imagen
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Agregamos opción para salir con 'q'
        break
