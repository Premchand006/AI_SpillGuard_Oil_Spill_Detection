import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.health_service import HealthService
from app import OilSpillDetector

def verify():
    print("=== System Verification ===")
    
    # Check GPU
    print("\nChecking GPU...")
    gpu = HealthService.check_gpu()
    print(gpu)
    
    # Check Storage
    print("\nChecking Storage...")
    storage = HealthService.check_storage(["checkpoints", "detection_history"])
    print(storage)
    
    # Check Model Loading
    print("\nChecking Model Loading...")
    try:
        detector = OilSpillDetector("best_model.pth")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Model loading failed: {e}")
        
    print("\nVerification Complete.")

if __name__ == "__main__":
    verify()
