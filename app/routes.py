from src.detect import detect
from flask import Blueprint, request, render_template, send_from_directory, jsonify
import os
import sys
import logging
import uuid
from datetime import datetime
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

routes = Blueprint('routes', __name__)

UPLOAD = os.environ.get('UPLOAD_DIR', 'uploads')
os.makedirs(UPLOAD, exist_ok=True)

MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), 'models', 'weights', 'best.pt'))

try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        logger.info(f"✅ Loaded trained model from {MODEL_PATH}")
        class_mapping = {0: 'fire_extinguisher',
                         1: 'toolbox', 2: 'oxygen_tank'}
        logger.info(f"✅ Model classes: {model.names}")
    else:
        logger.warning(
            f"❌ Trained model not found at {MODEL_PATH}, using default YOLOv8n")
        model = YOLO('yolov8n.pt')
        class_mapping = {}
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    model = YOLO('yolov8n.pt')
    class_mapping = {}


@routes.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        path = os.path.join(UPLOAD, file.filename)
        file.save(path)
        results = detect(path, model_path=MODEL_PATH)
        saved = results[0].masks.data if results else None
        return render_template('index.html', img_out=os.path.basename(results[0].path[0]))
    return render_template('index.html')


@routes.route('/api/detect', methods=['POST'])
def api_detect():
    try:
        logger.info("API detect endpoint called")
        if 'image' not in request.files:
            logger.warning("No image file in request")
            return jsonify({'error': 'No image file provided'}), 400
        file = request.files['image']
        if file.filename == '':
            logger.warning("Empty filename")
            return jsonify({'error': 'No image selected'}), 400
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{file.filename}"
        path = os.path.join(UPLOAD, filename)
        file.save(path)
        logger.info(f"File saved to {path}")

        results = model.predict(
            path,
            conf=0.25,
            imgsz=320,
            device='cpu',
            verbose=False
        )
        logger.info(f"Detection complete, processing results")

        detections = []
        if results and len(results) > 0:
            result = results[0]
            logger.info(f"Model classes: {result.names}")
            logger.info(
                f"Total detections: {len(result.boxes) if result.boxes is not None else 0}")

            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    original_label = result.names[class_id]

                    logger.info(
                        f"Detection {i}: class_id={class_id}, label='{original_label}', confidence={confidence:.3f}")

                    if len(result.names) == 3 and all(cls in ['FireExtinguisher', 'ToolBox', 'OxygenTank'] for cls in result.names.values()):
                        mapping = {'FireExtinguisher': 'fire_extinguisher',
                                   'ToolBox': 'toolbox', 'OxygenTank': 'oxygen_tank'}
                        class_name = mapping.get(
                            original_label, original_label.lower())
                    else:
                        class_name = class_mapping.get(
                            class_id, f'class_{class_id}')

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    detection = {
                        'id': str(i),
                        'label': class_name,
                        'confidence': round(confidence, 3),
                        'bbox': {
                            'x': round(x1, 2),
                            'y': round(y1, 2),
                            'width': round(x2 - x1, 2),
                            'height': round(y2 - y1, 2)
                        },
                        'risk_level': 'high' if confidence < 0.7 else 'medium' if confidence < 0.9 else 'low',
                        'original_detection': original_label
                    }
                    detections.append(detection)

            logger.info(
                f"Filtered to {len(detections)} VISTA-S relevant detections")
            return jsonify({
                'success': True,
                'detections': detections,
                'message': f'Found {len(detections)} VISTA-S objects: {", ".join(set([d["label"] for d in detections]))}',
                'model_info': {
                    'type': 'HackByte_Trained',
                    'classes': ['fire_extinguisher', 'toolbox', 'oxygen_tank'],
                    'confidence_threshold': 0.25,
                    'image_size': 320
                }
            })
        else:
            logger.warning("No detection results")
            return jsonify({
                'success': True,
                'detections': [],
                'message': 'No objects detected'
            })

    except Exception as e:
        logger.error(f"Error in API detection: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@routes.route('/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory('models/logs/detect', filename)


@routes.route('/api/images/<path:filename>')
def api_images(filename):
    return send_from_directory('../models/logs/detect', filename)


@routes.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })
