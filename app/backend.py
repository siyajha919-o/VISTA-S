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
# Allow requests from any origin during development, or from specific frontend in production
CORS(app, origins=['*', 'https://your-friends-frontend-domain.com', 'http://localhost:3000'])

# Register blueprint
app.register_blueprint(routes)

# Update model loading section
model_path = Path("../models/weights/best.pt")
if model_path.exists():
    model = YOLO(str(model_path))
    print("✅ Using trained HackByte model")
    
    # Load class configuration
    config_path = Path("../models/weights/class_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            class_config = yaml.safe_load(f)
        class_mapping = class_config.get('class_mapping', {})
        print(f"✅ Class mapping loaded: {class_mapping}")
    else:
        # Default mapping for HackByte dataset
        class_mapping = {
            0: 'fire_extinguisher',  # FireExtinguisher
            1: 'toolbox',            # ToolBox  
            2: 'oxygen_tank'         # OxygenTank
        }
else:
    print("⚠️ No trained model found, using default YOLOv8n")
    model = YOLO('yolov8n.pt')
    class_mapping = {}

# This makes the app work both locally and on Render
if __name__=='__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Observo backend server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)