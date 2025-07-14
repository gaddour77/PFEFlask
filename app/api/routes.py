from flask import Flask, render_template, request, jsonify, redirect, url_for
from ..utils.logger import logger
import traceback
import os
from ..core.yolo_manager import YoloManager
from ..core.yolo import Yolotrainer
from werkzeug.utils import secure_filename

def register_routes(app, socketio, yolo_manager, Yolotrainer):
    UPLOAD_FOLDER = 'static/uploads/'
    RESULT_FOLDER = 'static/results/'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['RESULT_FOLDER'] = RESULT_FOLDER
    # Créer le dossier uploads s'il n'existe pas
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)
    # Vérifier si l'extension du fichier est autorisée
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    # Ajouter `zip` au contexte des templates
    @app.context_processor
    def utility_processor():
        return dict(zip=zip)

    @app.route("/hello", methods=['GET'])
    def home():
        return jsonify("Hello World!")

    @app.route("/train", methods=["GET"])
    def train():
        try:
            result = yolo_manager.predict()
            if not result:  
                return jsonify({'error': 'NO RESULTS'}), 404
            logger.info(f"results:{result}")
            return jsonify(result)
        except Exception as e:
            logger.error("Error training")
            logger.error(f"Exception Traceback: {traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500

    @app.route("/", methods=['GET', 'POST'])
    def upload_image():
        if request.method == 'POST':
            if 'file' not in request.files:
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Faire la prédiction avec YOLO
                detections = yolo_manager.predict(filepath)

                return render_template("result.html", detections=detections, uploaded_image=filepath)
        return render_template("upload.html")
    @app.route("/detect", methods=['POST'])
    def detect_image():
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Save the file to a temporary location
        upload_folder = 'static/uploads/'
        result_folder = 'static/results/'
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(result_folder, exist_ok=True)

        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        # Initialize YOLO Manager
        yolo_manager = YoloManager()

        # Perform prediction
        detections = yolo_manager.predict(file_path)

        # Return the URL to the processed image with detections
        result_image_url = f'http://10.150.10.193:5000/{detections["result_image"]}'
        # Return the processed image URL and detections to the frontend
        # Format detections CORRECTLY
        return jsonify({
        'processed_image_url': result_image_url,
        'detections': {
            'boxes': detections['boxes'],
            'class_ids': detections['class_ids'],
            'scores': detections['scores']
        }
       })
    @app.route("/trainyolo", methods=['GET'])
    def trainyolo():
        try:
            # Path resolution
            current_script_path = os.path.dirname(os.path.abspath(__file__))  # app/api/
            app_folder = os.path.dirname(current_script_path)                # app/
            data_yaml = os.path.join(app_folder, 'DatasetYOLO', 'config.yaml')
            
            # Verify config exists
            if not os.path.exists(data_yaml):
                return jsonify({
                    "error": "config.yaml not found",
                    "searched_path": data_yaml
                }), 400

            # Initialize trainer with correct parameters
            yolotrainer = Yolotrainer(model_type='yolov8n-seg.pt')
            
            # Start training with proper arguments
            model_path = yolotrainer.train(
                data_yaml=data_yaml,
                epochs=20,
                batch=8  # Key change: batch_size → batch
            )

            return jsonify({
                "status": "success",
                "model_path": model_path,
                "config_used": data_yaml
            })

        except Exception as e:
            return jsonify({
                "error": str(e),
                "solution": "1. Ensure nc matches names length in config.yaml\n2. Use batch instead of batch_size",
                "traceback": traceback.format_exc()
            }), 500

    return app