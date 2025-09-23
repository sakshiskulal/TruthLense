#!/usr/bin/env python3
"""
ML Classifier Training Script for Image Authenticity Detection

This script trains a lightweight machine learning classifier using the features
extracted by the forensic analysis engine. It can replace or complement the
rule-based detection system.

Usage:
    python scripts/train_ml_classifier.py [--data-dir /path/to/labeled/images]
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.models.image_model import LightweightImageDetector

class MLClassifierTrainer:
    def __init__(self):
        self.detector = LightweightImageDetector()
        self.scaler = StandardScaler()
        self.model = None
        
    def extract_features_from_image(self, image_path):
        """Extract all forensic features from an image."""
        try:
            # Get the full analysis result
            result = self.detector.detect(image_path)
            features = result.get('features', {})
            
            # Convert to feature vector
            feature_names = [
                'edge_density', 'texture_variance', 'color_uniformity', 'freq_variance',
                'avg_compression', 'compression_var', 'ela_score', 'exif_present',
                'dct_anomaly_ratio', 'noise_gaussian_deviation', 'noise_spatial_consistency',
                'gray_entropy', 'gradient_skewness', 'camera_metadata_completeness',
                'bytes_per_pixel', 'suspicious_score_raw', 'green_score'
            ]
            
            feature_vector = []
            for name in feature_names:
                value = features.get(name, 0)
                if isinstance(value, bool):
                    value = float(value)
                feature_vector.append(value)
            
            return np.array(feature_vector), feature_names
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None, None

    def generate_synthetic_training_data(self, num_samples=1000):
        """Generate synthetic training data with known labels."""
        print(f"Generating {num_samples} synthetic training samples...")
        
        # Generate synthetic feature vectors
        np.random.seed(42)
        
        # Real image characteristics (based on typical camera photos)
        real_features = {
            'edge_density': np.random.normal(0.15, 0.05, num_samples//2),
            'texture_variance': np.random.lognormal(5.5, 0.8, num_samples//2),
            'color_uniformity': np.random.normal(35, 10, num_samples//2),
            'freq_variance': np.random.normal(8, 3, num_samples//2),
            'avg_compression': np.random.normal(0.08, 0.03, num_samples//2),
            'compression_var': np.random.gamma(2, 0.01, num_samples//2),
            'ela_score': np.random.gamma(2, 2, num_samples//2),
            'exif_present': np.random.choice([0, 1], num_samples//2, p=[0.3, 0.7]),
            'dct_anomaly_ratio': np.random.beta(1, 5, num_samples//2),
            'noise_gaussian_deviation': np.random.normal(1.0, 0.3, num_samples//2),
            'noise_spatial_consistency': np.random.beta(2, 8, num_samples//2),
            'gray_entropy': np.random.normal(7.2, 0.5, num_samples//2),
            'gradient_skewness': np.random.normal(0.1, 0.3, num_samples//2),
            'camera_metadata_completeness': np.random.beta(3, 2, num_samples//2),
            'bytes_per_pixel': np.random.normal(1.5, 0.5, num_samples//2),
        }
        
        # Fake/AI image characteristics (more suspicious patterns)
        fake_features = {
            'edge_density': np.random.normal(0.05, 0.03, num_samples//2),  # Lower edges
            'texture_variance': np.random.lognormal(4.0, 1.2, num_samples//2),  # Lower variance
            'color_uniformity': np.random.normal(20, 8, num_samples//2),  # More uniform
            'freq_variance': np.random.normal(3, 2, num_samples//2),  # Less variation
            'avg_compression': np.random.normal(0.03, 0.02, num_samples//2),  # Different compression
            'compression_var': np.random.gamma(1, 0.02, num_samples//2),
            'ela_score': np.random.gamma(1, 1, num_samples//2),
            'exif_present': np.random.choice([0, 1], num_samples//2, p=[0.8, 0.2]),  # Less EXIF
            'dct_anomaly_ratio': np.random.beta(2, 3, num_samples//2),  # More anomalies
            'noise_gaussian_deviation': np.random.normal(2.5, 0.8, num_samples//2),  # Non-Gaussian
            'noise_spatial_consistency': np.random.beta(5, 5, num_samples//2),  # Inconsistent
            'gray_entropy': np.random.normal(6.0, 0.8, num_samples//2),  # Lower entropy
            'gradient_skewness': np.random.normal(0.8, 0.6, num_samples//2),  # Higher skew
            'camera_metadata_completeness': np.random.beta(1, 4, num_samples//2),  # Less metadata
            'bytes_per_pixel': np.random.normal(0.8, 0.3, num_samples//2),  # Different compression
        }
        
        # Combine features and labels
        feature_names = list(real_features.keys())
        
        # Real samples (label = 0)
        real_matrix = np.column_stack([real_features[name] for name in feature_names])
        real_labels = np.zeros(num_samples//2)
        
        # Fake samples (label = 1)
        fake_matrix = np.column_stack([fake_features[name] for name in feature_names])
        fake_labels = np.ones(num_samples//2)
        
        # Add derived features
        real_suspicious = np.random.beta(1, 4, num_samples//2)  # Low suspicion
        real_green = np.random.beta(4, 2, num_samples//2)       # High green score
        
        fake_suspicious = np.random.beta(3, 2, num_samples//2)  # High suspicion
        fake_green = np.random.beta(1, 3, num_samples//2)       # Low green score
        
        real_matrix = np.column_stack([real_matrix, real_suspicious, real_green])
        fake_matrix = np.column_stack([fake_matrix, fake_suspicious, fake_green])
        
        feature_names.extend(['suspicious_score_raw', 'green_score'])
        
        # Combine all data
        X = np.vstack([real_matrix, fake_matrix])
        y = np.hstack([real_labels, fake_labels])
        
        return X, y, feature_names

    def load_real_data(self, data_dir):
        """Load real labeled data from directory structure."""
        data_dir = Path(data_dir)
        real_dir = data_dir / "real"
        fake_dir = data_dir / "fake"
        
        if not (real_dir.exists() and fake_dir.exists()):
            print(f"Expected directory structure: {data_dir}/{{real,fake}}/")
            return None, None, None
        
        X_list = []
        y_list = []
        feature_names = None
        
        # Process real images
        print("Processing real images...")
        for img_path in real_dir.glob("*"):
            if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                features, names = self.extract_features_from_image(str(img_path))
                if features is not None:
                    X_list.append(features)
                    y_list.append(0)  # Real = 0
                    if feature_names is None:
                        feature_names = names
        
        # Process fake images
        print("Processing fake images...")
        for img_path in fake_dir.glob("*"):
            if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                features, names = self.extract_features_from_image(str(img_path))
                if features is not None:
                    X_list.append(features)
                    y_list.append(1)  # Fake = 1
        
        if not X_list:
            print("No valid images found!")
            return None, None, None
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"Loaded {len(X)} samples ({np.sum(y==0)} real, {np.sum(y==1)} fake)")
        return X, y, feature_names

    def train_models(self, X, y, feature_names):
        """Train multiple ML models and select the best one."""
        print("Training ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train different models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        best_score = 0
        best_model = None
        best_name = None
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
            print(f"CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Train final model
            model.fit(X_train_scaled, y_train)
            
            # Test set evaluation
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            test_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"Test AUC: {test_auc:.3f}")
            print(f"Classification Report:")
            print(classification_report(y_test, y_pred))
            
            if test_auc > best_score:
                best_score = test_auc
                best_model = model
                best_name = name
        
        print(f"\nBest model: {best_name} (AUC: {best_score:.3f})")
        self.model = best_model
        
        # Feature importance (if available)
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(importance_df.head(10).to_string(index=False))
        
        return best_model, best_score

    def save_model(self, output_dir="backend/models"):
        """Save the trained model and scaler."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        model_path = output_dir / "image_classifier.joblib"
        scaler_path = output_dir / "feature_scaler.joblib"
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
        
        # Save feature names for reference
        with open(output_dir / "feature_names.json", 'w') as f:
            json.dump(self.feature_names, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Train ML classifier for image authenticity")
    parser.add_argument('--data-dir', type=str, help='Directory with real/fake subdirectories')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic training data')
    parser.add_argument('--samples', type=int, default=1000, help='Number of synthetic samples')
    
    args = parser.parse_args()
    
    trainer = MLClassifierTrainer()
    
    if args.data_dir and not args.synthetic:
        # Use real data
        X, y, feature_names = trainer.load_real_data(args.data_dir)
        if X is None:
            print("Failed to load real data, falling back to synthetic data")
            X, y, feature_names = trainer.generate_synthetic_training_data(args.samples)
    else:
        # Use synthetic data
        X, y, feature_names = trainer.generate_synthetic_training_data(args.samples)
    
    trainer.feature_names = feature_names
    
    # Train models
    model, score = trainer.train_models(X, y, feature_names)
    
    # Save model
    trainer.save_model()
    
    print(f"\nTraining complete! Best model AUC: {score:.3f}")
    print("To use the ML classifier in production, set ML_CLASSIFIER_ENABLED=true in your environment")

if __name__ == "__main__":
    main()