from ultralytics import YOLO
import torch

class Yolotrainer:
    def __init__(self, model_type='yolov8n.pt'):
        self.model = YOLO(model_type)  # Proper initialization
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, data_yaml, epochs=10, batch=16):
        """Train with proper YOLOv8 arguments"""
        try:
            # Verify dataset config
            with open(data_yaml) as f:
                import yaml
                data = yaml.safe_load(f)
                assert len(data['names']) == data['nc'], "names length must match nc"
            
            # Train with correct parameters
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                batch=batch,  # Correct argument name
                imgsz=640,
                device='0' if torch.cuda.is_available() else 'cpu'
            )
            return results.save_dir

        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")