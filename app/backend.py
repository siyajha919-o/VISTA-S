from flask import Flask
from flask_cors import CORS
import logging
from routes import routes
import os
from pathlib import Path
from ultralytics import YOLO
import yaml

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=[
     '*', 'https://your-friends-frontend-domain.com', 'http://localhost:3000'])

app.register_blueprint(routes)

model_path = Path("../models/weights/best.pt")
if model_path.exists():
    model = YOLO(str(model_path))
    print("✅ Using trained HackByte model")

    config_path = Path("../models/weights/class_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            class_config = yaml.safe_load(f)
        class_mapping = class_config.get('class_mapping', {})
        print(f"✅ Class mapping loaded: {class_mapping}")
    else:
        class_mapping = {
            0: 'fire_extinguisher',
            1: 'toolbox',
            2: 'oxygen_tank'
        }
else:
    print("⚠️ No trained model found, using default YOLOv8n")
    model = YOLO('yolov8n.pt')
    class_mapping = {}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Observo backend server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
