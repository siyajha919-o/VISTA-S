# VISTA-S: Visual Interface System for Target Assessment - Space

![VISTA-S Logo](https://img.shields.io/badge/VISTA--S-Space%20Object%20Detection-blue?style=for-the-badge&logo=satellite)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![React](https://img.shields.io/badge/React-18%2B-blue?style=for-the-badge&logo=react)
![YOLOv8](https://img.shields.io/badge/YOLOv8-AI%20Model-green?style=for-the-badge)

## 🚀 Overview

VISTA-S is an advanced AI-powered object detection system specifically designed for space station safety operations. It uses a custom-trained YOLOv8 model to detect three critical objects in space environments:

- 🧯 **Fire Extinguisher**
- 🫁 **Oxygen Tank** 
- 🧰 **Toolbox/Toolkit**

---

## ✨ Features & Highlights

- **High-Precision Object Detection:** Leverages state-of-the-art YOLOv8 for superior accuracy.
- **Optimized Performance:** Engineered for efficiency, delivering rapid inference.
- **User-Friendly Demo Application:** Easily visualize and interact with the model's capabilities.
- **Comprehensive Data Handling:** Streamlined data preparation for seamless integration.

---

## 📸 Glimpse of VISTA in Action

![image](https://github.com/user-attachments/assets/dd193d48-df47-4687-8eda-10ce44a7e0d4)


<br>

![image](https://github.com/user-attachments/assets/da6f8f5e-2834-4007-9003-6ee1eaab3798)



---

## ⚡️ Quick Setup Guide

Get VISTA up and running in a few simple steps!

### 1. Create Environment

```bash
conda env create -f environment.yaml
```

### 2. Activate Environment

```bash
conda activate VISTA
```

---

### 📦 Data Preparation

To begin, you'll need the **Falcon Dataset**.

- **Download the Falcon Dataset:**  
  Access the dataset from [here](#). <!-- Add actual link if available -->

- **Unzip & Place:**  
  Unzip the downloaded dataset and place its contents into the following directory structure:

  ```
  data/raw/
  ```

> **Note:** The dataset is not included in this repository due to its size. Please download it manually using the provided link.

---

### 🏋️‍♂️ Training the Model

Train Observo on your local machine with a single, straightforward command:

```bash
python src/train.py
```

---

### 🔍 Running Inference

Perform object detection on any sample image effortlessly:

```bash
python src/detect.py data/raw/test/images/sample.jpg
```

---

### 🖥️ Demo Application

Experience VISTA-S through its interactive web application:

1. **Navigate to App Directory:**

    ```bash
    cd app
    ```

2. **Install Requirements:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Start Backend Server:**

    ```bash
    python backend.py
    ```

---

## 📊 Performance Metrics

Observo delivers exceptional performance on the Falcon dataset:

- **Precision:** ~0.97967  
- **Recall:** ~0.90879
- **mAP@0.5:** ~0.94157 
- **mAP@0.5:0.95:** ~0.88426

These impressive results were achieved using the YOLOv8 architecture.  
For an in-depth analysis, including detailed logs and visualizations, please refer to the `models/logs/yolov8_observo/` directory.

---

## 📝 Project Structure

The repository is thoughtfully organized for clarity and ease of navigation:

```
├── app/                     # Flask backend application
│   ├── backend.py           # Main Flask application entry point
│   ├── routes.py            # API routes and endpoints
│   ├── simple_backend.py    # Simplified backend for deployment testing
│   ├── requirements.txt     # Python dependencies for the app
│   └── templates/           # HTML templates for web interface
│       └── index.html       # Main web interface template
├── config/                  # Configuration files
│   └── observo.yaml         # YOLO training configuration
├── data/                    # Dataset storage (excluded from git)
│   └── raw/                 # Raw training data and parameters
│       ├── classes.txt      # Class definitions
│       ├── predict.py       # Prediction utilities
│       ├── train.py         # Training utilities
│       ├── visualize.py     # Visualization utilities
│       ├── yolo_params.yaml # YOLO model parameters
│       └── data/            # Training/validation/test datasets
├── docs/                    # Project documentation
│   └── report_outline.md    # Project report structure
├── mobile/                  # React Native mobile application
│   ├── src/                 # Mobile app source code
│   │   ├── App.js           # Main mobile app component
│   │   ├── api/             # API integration for mobile
│   │   └── screens/         # Mobile app screens
│   ├── package.json         # Mobile app dependencies
│   ├── README.md            # Mobile app setup instructions
│   └── SETUP.md             # Detailed mobile setup guide
├── models/                  # Model storage (excluded from git)
│   ├── weights/             # Trained model weights (.pt files)
│   └── logs/                # Training logs and results
├── notebooks/               # Jupyter notebooks (excluded from git)
│   ├── EDA.ipynb            # Exploratory data analysis
│   └── train_yolov8.ipynb   # YOLOv8 training notebook
├── src/                     # Core detection and training code
│   ├── detect.py            # Object detection implementation
│   ├── train.py             # Model training script
│   ├── utils.py             # Utility functions
│   └── constraints.txt      # Package version constraints
├── uploads/                 # User uploaded images (excluded from git)
├── Web_App_frontend/        # React/TypeScript web frontend
│   ├── src/                 # Frontend source code
│   │   ├── components/      # Reusable UI components
│   │   ├── pages/           # Application pages
│   │   └── hooks/           # Custom React hooks
│   ├── package.json         # Frontend dependencies
│   └── vite.config.ts       # Build configuration
├── environment.yaml         # Conda environment definition
├── requirements.txt         # Python dependencies (root level)
├── requirements_minimal.txt # Minimal dependencies for deployment testing
├── render.yaml              # Render deployment configuration
├── gunicorn_config.py       # Production server configuration
├── Procfile                 # Alternative deployment configuration
├── .gitignore               # Git ignore patterns
├── .gitattributes           # Git file handling rules
├── DEPLOYMENT.md            # Deployment instructions and guides
└── README.md                # Project overview and setup instructions
```

---

## ⚠️ Important: Do NOT Commit Sensitive or Large Files

- **Model weights, logs, uploads, and raw data are excluded from version control via `.gitignore`.**
- **Never commit files in `models/weights/`, `models/logs/`, `uploads/`, or `data/raw/` to GitHub.**
- **Notebooks and environment folders are also excluded.**

---

## 🤝 Contributing

We welcome contributions from the community! If you have suggestions, bug reports, or would like to contribute code:

1. **Fork** the repository.
2. **Create a new branch** for your feature or bug fix.
3. **Submit a pull request** with a clear description of your changes.

For major changes or new features, please open an issue first to discuss the proposed modifications.

---

## 📄 License

This project is open-source and licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## 👥 Team

- **@siyajha919-o** - Project Lead & Core Developer
- **@Rishikesh-Jadhav** - Collaborator

## 🔗 Links

- **Repository**: https://github.com/siyajha919-o/VISTA-S
- **Demo Video**: Coming Soon
- **Documentation**: See `/docs` folder
- **Contributors**: See `CONTRIBUTORS.md`

---

<p align="center">
  ✨ Enjoy exploring the universe of possibilities with Observo! <br>
  Feel free to ⭐️ the repository, open issues, or contribute to its growth.<br>
  Your feedback and contributions are highly valued.
</p>

