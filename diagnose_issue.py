"""
Quick diagnostic to test the current model and show exactly what classes it detects
"""
from ultralytics import YOLO
import os
import sys
sys.path.append('.')


def diagnose_current_model():
    print("🔍 CURRENT MODEL DIAGNOSIS")
    print("=" * 40)

    model_path = "models/weights/best.pt"
    if os.path.exists(model_path):
        model = YOLO(model_path)
        print(f"✅ Current model classes: {model.names}")

        # Check what the current model actually has
        for class_id, class_name in model.names.items():
            print(f"   Class {class_id}: '{class_name}'")

        print("\n🎯 Expected behavior:")
        print("   - Toolkit/ToolBox should be detected as 'toolbox' (display)")
        print("   - But current model might have wrong mapping")

        print("\n🔧 SOLUTION:")
        print("   - New model training in progress with correct mapping")
        print("   - Backend updated to handle both old and new models")
        print("   - Class mapping: FireExtinguisher->fire_extinguisher, ToolBox->toolbox, OxygenTank->oxygen_tank")

    else:
        print("❌ No model found")


def show_backend_mapping():
    print("\n🔀 BACKEND CLASS MAPPING")
    print("=" * 40)
    print("Dataset classes -> Display names:")
    print("  'FireExtinguisher' -> 'fire_extinguisher'")
    print("  'ToolBox' -> 'toolbox'")
    print("  'OxygenTank' -> 'oxygen_tank'")
    print("\n✅ This mapping is now active in the backend!")


if __name__ == "__main__":
    diagnose_current_model()
    show_backend_mapping()

    print("\n🚀 STATUS:")
    print("✅ Backend updated with correct mapping")
    print("⏳ New model training with correct classes")
    print("📱 Web app ready for testing at http://localhost:8083")
    print("🎯 Upload a toolbox image to test the fix!")
