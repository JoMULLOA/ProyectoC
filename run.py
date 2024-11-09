import cv2
import os
import numpy as np

# Cargar el modelo entrenado
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace.xml')

# Directorio donde se encuentran las imágenes de entrenamiento
dataPath = r'C:\Users\manri\OneDrive\Escritorio\PCim'
peopleList = os.listdir(dataPath)
print('Lista de personas:', peopleList)

# Inicializar la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (150, 150))
        
        # Realizar la predicción
        label, confidence = face_recognizer.predict(face)
        print(f"Etiqueta: {label}, Confianza: {confidence}")
        
        # Mostrar el nombre de la persona reconocida
        if confidence < 60:
            cv2.putText(frame, f'{peopleList[label]}', (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Desconocido', (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()