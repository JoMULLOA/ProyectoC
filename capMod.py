import cv2
import os

dataPath = r'C:\Users\manri\OneDrive\Escritorio\PCim'

# Preguntar cuántas personas se desean ingresar
num_personas = int(input("¿Cuántas personas deseas ingresar para el entrenamiento? "))

for i in range(num_personas):
    personName = input(f"Introduce el nombre de la persona {i + 1}: ")
    personPath = os.path.join(dataPath, personName)

    if not os.path.exists(personPath):
        os.makedirs(personPath)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceClassif.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(50, 50), maxSize=(300, 300))
        for (x, y, w, h) in faces:
            if w < 50 or h < 50:  # Filtrar caras muy pequeñas
                continue
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (150, 150))
            face = cv2.equalizeHist(face)  # Normalizar el histograma de la imagen
            cv2.imwrite(os.path.join(personPath, f'face_{count}.jpg'), face)
            count += 1
            print(f"Capturas: {count}")  # Mostrar en pantalla cuántas capturas lleva
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 2000:  # Captura 500 imágenes o presiona 'q' para salir
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Imágenes de {personName} capturadas.")

print("Captura de imágenes completada.")
