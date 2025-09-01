<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-1.9%2B-orange?style=for-the-badge&logo=pytorch" alt="PyTorch Version">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<h1 align="center">VISTA-S: Visual Inference System for Target Assessment  </h1>
<h2 align="center">DualityAI Space Station Model </h2>

<p align="center">
  <b>Welcome to <strong>VISTA</strong></b> — an advanced, high-performance AI model designed for <b>object detection</b> with unparalleled precision.
  Inspired by the vast complexity and endless wonder of space exploration, VISTA brings cutting-edge computer vision capabilities to your fingertips. 🌌
</p>

---

## ✨ Features & Highlights

- **High-Precision Object Detection:** Leverages state-of-the-art YOLOv8 for superior accuracy.
- **Optimized Performance:** Engineered for efficiency, delivering rapid inference.
- **User-Friendly Demo Application:** Easily visualize and interact with the model's capabilities.
- **Comprehensive Data Handling:** Streamlined data preparation for seamless integration.

---

## 📸 Glimpse of VISTA in Action

![Screenshot 2025-06-26 111639](https://github.com/user-attachments/assets/ef5defcd-19df-4515-be84-acdd09346f24)

<br>

![Screenshot 2025-06-26 111656](https://github.com/user-attachments/assets/bd32a41b-ecf9-48e3-b233-11f638d9783c)

<br>

![Screenshot 2025-06-26 111707](https://github.com/user-attachments/assets/ad7b5e0a-3c0a-47b6-8202-efc10d108cd2)

<br>

![Screenshot 2025-06-26 111724](https://github.com/user-attachments/assets/1b65551c-454b-4799-b8dc-136a10ea9b26)

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
- **mAP@0.5:** ~0.98000
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

## 👥 Team

- **@siyajha919-o** - Project Lead & Core Developer
- **@abhijeetjha** - Collaborator

---

## 📄 License

This project is open-source and licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

<p align="center">
  ✨ Enjoy exploring the universe of possibilities with VISTA! <br>
  Feel free to ⭐️ the repository, open issues, or contribute to its growth.<br>
  Your feedback and contributions are highly valued.
</p>
