import cv2
import os

dataPath = r'C:\Users\manri\OneDrive\Escritorio\PCim'  # Cambia a la ruta donde almacenarás las imágenes

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
    
    delay = 1000

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (150, 150))
            cv2.imwrite(os.path.join(personPath, f'face_{count}.jpg'), face)
            count += 1
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 500:  # Captura 100 imágenes o presiona 'q' para salir
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Imágenes de {personName} capturadas.")

print("Captura de imágenes completada.")