from ultralytics import YOLO
import os
from IPython.display import display, Image
from IPython import display
display.clear_output()
import logging

logger = logging.getLogger(__name__)

class YoloManager:
    def __init__(self, model_path='yolo11n.pt'):
        self.model_path = model_path

    def predict(self, image_path):
        # Charger le modèle YOLO
        model = YOLO(self.model_path)

        # Vérifier que le fichier existe
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"L'image {image_path} n'existe pas.")

        # Faire la prédiction
        results = model(image_path)

        # Définir le dossier de sauvegarde des résultats
        result_folder = 'static/results/'

        # Assurer que le dossier existe
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        # Créer le chemin d'enregistrement du fichier résultat
        filename = os.path.basename(image_path)
        result_image_path = os.path.join(result_folder, 'result_' + filename)

        # Sauvegarder l'image avec les résultats
        results[0].save(filename=result_image_path)

        # Extraire les détections
        detections = {
            "boxes": results[0].boxes.xyxy.tolist() if results[0].boxes is not None else [],
            "scores": results[0].boxes.conf.tolist() if results[0].boxes is not None else [],
            "class_ids": results[0].boxes.cls.tolist() if results[0].boxes is not None else [],
            "result_image": result_image_path
        }

        return detections
