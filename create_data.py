import cv2
import os
from datetime import datetime
from cvzone.HandTrackingModule import HandDetector
from utils import create_folder_structure


# Crear estructura de carpetas al inicio
base_path = create_folder_structure()

# Inicializar la cámara
DdC = cv2.VideoCapture(0)

# Configurar resolución de la cámara (width x height) - imagen más corta
# Reduce el tamaño para evitar procesamiento innecesario
DdC.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
DdC.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

# Inicializar el Dt de manos con los mismos parámetros que tenías
Dt = HandDetector(maxHands=2, detectionCon=0.8, minTrackCon=0.5)

# Conexiones de la mano
CD = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),
      (11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]

# Modos de captura
modes = {1: 'train', 2: 'test', 3: 'val'}
current_mode = 1  # Por defecto en train

# Contador de imágenes
counters = {}

# Letras válidas
valid_letters = set('abcdefghijklmnopqrstuvwxyz')

# Información inicial
print("\n" + "="*60)
print("SISTEMA DE CAPTURA - DETECCIÓN DE MANOS")
print("="*60)
print("\nControles:")
print("  [1] = Modo TRAIN")
print("  [2] = Modo TEST")
print("  [3] = Modo VAL")
print("  [a-z] = Capturar imagen con esa letra como etiqueta")
print("  [Q] = Salir")
print("="*60)
print(f"\nModo actual: {modes[current_mode].upper()}\n")

while True:
    # Leer el frame de la cámara
    success, img = DdC.read()
    if not success:
        break

    # Detectar las manos. Disable drawing overlays so captured images stay clean.
    # HandDetector.findHands has a `draw` parameter; set draw=False to avoid
    # annotating the passed image. We keep lmList for potential processing but
    # do NOT draw circles/lines on `img` so saved images are annotation-free.
    hands, img = Dt.findHands(img, draw=False)

    # Si se detectan manos, solo extraemos la lista de landmarks (no dibujamos)
    if hands:
        hand = hands[0]  # Tomamos la primera mano
        lmList = hand["lmList"]  # Lista de landmarks
        # NOTE: If you ever want to show debug overlays while keeping saved
        # images clean, draw on a copy: disp = img.copy(); draw on disp; cv2.imshow(disp)

    # Create a copy for display (with text overlays)
    display_img = img.copy()
    
    # Add text overlays to display image only
    mode_text = f"Modo: {modes[current_mode].upper()}"
    cv2.putText(display_img, mode_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(display_img, "Presiona [a-z] para capturar", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(display_img, "Presiona [1,2,3] para cambiar modo", (10, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Show the display image (with text)
    cv2.imshow("Image", display_img)

    # Capturar tecla presionada
    key = cv2.waitKey(1) & 0xFF

    # Salir con 'Q'
    if key == ord('Q'):
        print("\nSaliendo...")
        break

    # Cambiar modo (1, 2, o 3)
    elif key == ord('1'):
        current_mode = 1
        print(f"\n→ Modo cambiado a: {modes[current_mode].upper()}")
    elif key == ord('2'):
        current_mode = 2
        print(f"\n→ Modo cambiado a: {modes[current_mode].upper()}")
    elif key == ord('3'):
        current_mode = 3
        print(f"\n→ Modo cambiado a: {modes[current_mode].upper()}")

    # Capturar imagen si se presiona una letra válida
    elif chr(key) in valid_letters:
        letter = chr(key)
        mode_folder = modes[current_mode]

        # Crear clave para el contador
        counter_key = f"{mode_folder}_{letter}"
        if counter_key not in counters:
            counters[counter_key] = 0

        counters[counter_key] += 1

        # Generar nombre de archivo corto
        filename = f"{letter}_{counters[counter_key]:04d}.jpg"

        # Ruta completa
        save_path = os.path.join(base_path, mode_folder, letter, filename)

        # Guardar imagen
        cv2.imwrite(save_path, img)

        print(f"✓ Imagen guardada: {mode_folder}/{letter}/{filename}")

# Liberar recursos
DdC.release()
cv2.destroyAllWindows()

# Mostrar resumen
print("\n" + "="*60)
print("RESUMEN DE CAPTURA")
print("="*60)
if counters:
    for key, count in sorted(counters.items()):
        mode, letter = key.split('_')
        print(f"  {mode.upper()}/{letter.upper()}: {count} imágenes")
else:
    print("  No se capturaron imágenes")
print("="*60)