import cv2
import os
import numpy as np

# Définition des fichiers de modèle
model_files = {
    'face_cascade': 'haarcascade_frontalface_default.xml',
    'gender_deploy': 'gender_deploy.prototxt',
    'gender_model': 'gender_net.caffemodel',
    'age_deploy': 'age_deploy.prototxt',
    'age_model': 'age_net.caffemodel'
}

# Vérification de l'existence des fichiers
for key, file in model_files.items():
    if not os.path.isfile(file):
        print(f"Erreur: Le fichier {file} n'existe pas ou n'est pas accessible")
        exit(1)

# Chargement du détecteur de visage
print("Chargement du détecteur de visage...")
face_cascade = cv2.CascadeClassifier(model_files['face_cascade'])

# Chargement du modèle de genre
print("Chargement du modèle de genre...")
try:
    gender_net = cv2.dnn.readNetFromCaffe(model_files['gender_deploy'], model_files['gender_model'])
    print("Modèle de genre chargé avec succès")
except Exception as e:
    print(f"Erreur lors du chargement du modèle de genre: {e}")
    exit(1)

# Chargement du modèle d'âge
print("Chargement du modèle d'âge...")
try:
    age_net = cv2.dnn.readNetFromCaffe(model_files['age_deploy'], model_files['age_model'])
    print("Modèle d'âge chargé avec succès")
except Exception as e:
    print(f"Erreur lors du chargement du modèle d'âge: {e}")
    exit(1)

# Listes pour la classification
gender_list = ['Homme', 'Femme']
age_list = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']

# Initialisation de la webcam
print("Initialisation de la webcam...")
webcam = cv2.VideoCapture(0)

# Vérification de l'ouverture de la webcam
if not webcam.isOpened():
    print("Erreur: Impossible d'ouvrir la webcam")
    exit(1)

print("Démarrage de la détection de visage, genre et âge. Appuyez sur 'Échap' pour quitter.")

while True:
    # Capture d'image
    ret, img = webcam.read()
    if not ret:
        print("Erreur: Impossible de lire l'image de la webcam")
        break
    
    # Conversion en niveaux de gris pour la détection de visage
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Détection des visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        print(f"Nombre de visages détectés: {len(faces)}")
    
    for (x, y, w, h) in faces:
        # Rectangle autour du visage
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Extraction de la région du visage
        face_img = img[y:y+h, x:x+w].copy()
        
        # Vérification que l'image du visage n'est pas vide
        if face_img.size == 0:
            continue
        
        if face_img.shape[0] > 0 and face_img.shape[1] > 0:
            try:
                # Préparation de l'image pour les modèles
                # Pour le genre
                gender_blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.42, 87.76, 114.89), swapRB=False)
                gender_net.setInput(gender_blob)
                gender_pred = gender_net.forward()
                gender_idx = gender_pred[0].argmax()
                gender = gender_list[gender_idx]
                gender_confidence = gender_pred[0][gender_idx] * 100
                
                # Pour l'âge
                age_blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.42, 87.76, 114.89), swapRB=False)
                age_net.setInput(age_blob)
                age_pred = age_net.forward()
                age_idx = age_pred[0].argmax()
                age = age_list[age_idx]
                age_confidence = age_pred[0][age_idx] * 100
                
                # Affichage des résultats
                gender_label = f"{gender} ({gender_confidence:.1f}%)"
                age_label = f"{age} ans ({age_confidence:.1f}%)"
                
                # Position du texte
                y_pos_gender = max(y - 10, 20)
                y_pos_age = max(y - 30, 5)
                
                # Ajout du texte à l'image
                cv2.putText(img, gender_label, (x, y_pos_gender), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(img, age_label, (x, y_pos_age), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            except Exception as e:
                print(f"Erreur lors de la prédiction: {e}")
    
    # Affichage de l'image avec les résultats
    cv2.imshow("Détection de visage, genre et âge", img)
    
    # Gestion de la touche d'échappement
    key = cv2.waitKey(1)
    if key == 27:  # Code ASCII pour la touche Échap
        print("Arrêt demandé par l'utilisateur")
        break

# Libération des ressources
print("Nettoyage des ressources....")
webcam.release()
cv2.destroyAllWindows()
print("Programme terminé avec succès")
