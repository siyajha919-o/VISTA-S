import os, cv2
from ultralytics import YOLO

# Custom class names for VISTA-S
VISTA_CLASSES = ['fire_extinguisher', 'oxygen_tank', 'toolbox']

def detect(image_path, model_path='models/weights/best.pt', save_dir=None):
    # Default save directory if not provided
    if save_dir is None:
        save_dir = os.environ.get('SAVE_DIR', 'models/logs/detect')
    
    # Make sure directories exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the model - try custom model first, then fallback
    if not os.path.exists(model_path):
        print(f"Custom model not found at {model_path}")
        # Try alternative locations
        alt_paths = [
            'config/best.pt',
            'models/best.pt', 
            'models/logs/yolov8_observo/weights/best.pt',
            'yolov8n.pt'  # Generic fallback
        ]
        
        model_found = False
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                print(f"Using model from: {alt_path}")
                model_found = True
                break
        
        if not model_found:
            print("No custom model found, using default YOLOv8n")
            print("⚠️  WARNING: Generic model active!")
            print("🎯  VISTA-S should detect: fire_extinguisher, oxygen_tank, toolbox")
            print("❌  Currently detecting: generic objects (oven, refrigerator, etc.)")
            model_path = 'yolov8n.pt'
    
    model = YOLO(model_path)
    
    # Run inference - use smaller image size for our trained model
    # Our model was trained on 320x320 images, so use that for better detection
    results = model.predict(source=image_path, save=True, save_dir=save_dir, imgsz=320, conf=0.25)
    
    # If using generic model, filter/remap results to VISTA classes
    if 'yolov8n.pt' in model_path:
        print("🔄 Filtering results for VISTA-S classes...")
        # You can add logic here to map generic detections to your classes
        # For now, we'll just warn the user
    
    print(f"Detections saved to {save_dir}")
    return results

if __name__ == '__main__':
    import sys
    img = sys.argv[1] if len(sys.argv)>1 else '../data/raw/test/images/sample.jpg'
    detect(img)