import cv2
import os
import numpy as np

dataPath = r'C:\Users\manri\OneDrive\Escritorio\PCim'  # Cambia a la ruta donde hayas almacenado Data
peopleList = os.listdir(dataPath)
print('Lista de personas:', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = os.path.join(dataPath, nameDir)
    print('Leyendo imágenes de', nameDir)

    for fileName in os.listdir(personPath):
        imagePath = os.path.join(personPath, fileName)
        print('Rostros:', imagePath)
        image = cv2.imread(imagePath, 0)
        if image is None:
            print(f"Error al cargar la imagen: {imagePath}")
            continue
        labels.append(label)
        facesData.append(image)
    label += 1

if len(facesData) == 0:
    print("No se encontraron datos de entrenamiento.")
else:
    # Entrenando el reconocedor de rostros
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("Entrenando...")
    face_recognizer.train(facesData, np.array(labels))
    # Almacenando el modelo
    face_recognizer.write('modeloLBPHFace.xml')
    print("Modelo almacenado como modeloLBPHFace.xml")