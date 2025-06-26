## ✨ Features

- **Ultra-Fast Detection**: Optimized YOLOv8n model for real-time CPU inference
- **Space-Optimized**: Trained specifically on space station equipment  
- **Modern Web Interface**: React-based PWA with real-time detection
- **Mobile Ready**: React Native mobile application
- **Professional UI**: Space-themed interface with modern design
- **API-First**: RESTful API for easy integration

## 🏗️ Architecture

```
VISTA_S/
├── app/                    # Flask backend
│   ├── routes.py          # API endpoints and detection logic
│   └── templates/         # HTML templates
├── src/                   # Core detection modules
│   ├── detect.py         # Detection implementation  
│   └── train.py          # Training scripts
├── models/               # AI models and weights
│   └── weights/
│       └── best.pt       # Trained YOLOv8 model
├── Web_App_frontend_new/ # React PWA frontend
├── mobile/              # React Native mobile app
└── docs/               # Documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### Backend Setup
```bash
cd VISTA_S
pip install -r requirements.txt
python app/routes.py
```

### Frontend Setup  
```bash
cd Web_App_frontend_new
npm install
npm run dev
```

### Access the Application
- **Web App**: http://localhost:5173
- **API**: http://localhost:5000/api/detect

## 📸 Frontend Interface

### Modern Space-Themed UI
The VISTA-S web interface features a professional space-themed design with:
- Real-time object detection dashboard
- Interactive upload interface
- Professional metrics and statistics
- Responsive design for all devices

### Detection Results
The system provides detailed detection results with:
- Bounding box visualization
- Confidence scores
- Object classification
- Risk level assessment

## 🎯 Detection API

### POST /api/detect
Upload an image to detect space objects.

**Request:**
```bash
curl -X POST \
  -F "image=@your_image.jpg" \
  http://localhost:5000/api/detect
```

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "id": "0", 
      "label": "oxygen_tank",
      "confidence": 0.858,
      "bbox": {
        "x": 150.5,
        "y": 200.3,
        "width": 100.2,
        "height": 150.7
      },
      "risk_level": "low",
      "original_detection": "OxygenTank"
    }
  ],
  "message": "Found 1 VISTA-S objects: oxygen_tank",
  "model_info": {
    "type": "HackByte_Trained",
    "classes": ["fire_extinguisher", "toolbox", "oxygen_tank"],
    "confidence_threshold": 0.25,
    "image_size": 320
  }
}
```

## 🧠 Model Details

- **Base Model**: YOLOv8n (nano - optimized for speed)
- **Training Dataset**: HackByte_Dataset (1,400+ images)
- **Classes**: 3 (FireExtinguisher, ToolBox, OxygenTank)
- **Image Size**: 320x320 pixels
- **Inference**: CPU-optimized for real-time detection
- **Confidence Threshold**: 0.25

## 🌟 Frontend Features

- **Modern React Interface** with TypeScript
- **Progressive Web App (PWA)** capabilities
- **Real-time Detection** with image upload
- **Space-themed UI** with animations
- **Responsive Design** for all devices
- **Professional Metrics Dashboard**

## 📱 Mobile App

React Native application with:
- Camera integration
- Real-time detection
- Offline capabilities
- Cross-platform support (iOS/Android)

## 📊 Performance Metrics

- **Detection Speed**: ~50ms per image (CPU)
- **Model Size**: 6.2MB
- **Accuracy**: 94%+ on validation set
- **Memory Usage**: <500MB RAM

## 🚀 Deployment

### Local Development
Both backend and frontend can run locally for development and testing.

### Production Deployment
- Backend: Flask/Gunicorn
- Frontend: Static hosting (Vercel, Netlify)
- Mobile: App Store/Play Store

## 🎯 Demo

The system is designed for live demonstration with:
1. Real-time object detection
2. Professional UI showcase
3. API functionality demo
4. Mobile app preview

---

**Built with ❤️ for Space Station Safety Operations**
