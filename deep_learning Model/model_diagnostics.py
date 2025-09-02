# Model Diagnostics Script
# model_diagnostics.py

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib

def check_model_files():
    """Check if all required model files exist"""
    print("=" * 60)
    print("MODEL FILES DIAGNOSTIC")
    print("=" * 60)
    
    files_to_check = [
        'disease_prediction_model.h5',
        'disease_prediction_scaler.pkl', 
        'feature_names.pkl'
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file}: EXISTS ({size:,} bytes)")
        else:
            print(f"❌ {file}: MISSING")
    
    return all(os.path.exists(f) for f in files_to_check)

def analyze_model():
    """Analyze the trained model"""
    print("\n" + "=" * 60)
    print("MODEL ANALYSIS")
    print("=" * 60)
    
    try:
        # Load model
        model = keras.models.load_model('disease_prediction_model.h5')
        print("✅ Model loaded successfully")
        
        # Model summary
        print("\nModel Architecture:")
        model.summary()
        
        # Check model weights
        total_params = model.count_params()
        print(f"\nTotal parameters: {total_params:,}")
        
        # Check if model weights are initialized properly
        first_layer_weights = model.layers[0].get_weights()[0]
        print(f"First layer weight stats:")
        print(f"  Mean: {np.mean(first_layer_weights):.6f}")
        print(f"  Std: {np.std(first_layer_weights):.6f}")
        print(f"  Min: {np.min(first_layer_weights):.6f}")
        print(f"  Max: {np.max(first_layer_weights):.6f}")
        
        # Test prediction with random data
        print(f"\nTesting model prediction with random input...")
        test_input = np.random.random((1, first_layer_weights.shape[0]))
        test_pred = model.predict(test_input, verbose=0)
        print(f"Random input prediction: {test_pred[0][0]:.6f}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def analyze_scaler():
    """Analyze the scaler"""
    print("\n" + "=" * 60)
    print("SCALER ANALYSIS")
    print("=" * 60)
    
    try:
        scaler = joblib.load('disease_prediction_scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        print("✅ Scaler and features loaded successfully")
        print(f"Number of features: {len(feature_names)}")
        print(f"Feature names: {feature_names}")
        
        print(f"\nScaler statistics:")
        print(f"  Mean values: {scaler.mean_}")
        print(f"  Scale values: {scaler.scale_}")
        
        return scaler, feature_names
        
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
        return None, None

def test_predictions():
    """Test predictions with known patterns across all 5 risk categories"""
    print("\n" + "=" * 60)
    print("PREDICTION TESTING - 5 RISK CATEGORIES")
    print("=" * 60)
    
    try:
        model = keras.models.load_model('disease_prediction_model.h5')
        scaler = joblib.load('disease_prediction_scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        # Define 5 test patients targeting each risk category
        test_patients = {
            "Very High Risk (Target: 90%+)": {
                'age': 72, 'sex': 1, 'cp': 1, 'trestbps': 185, 'chol': 350,
                'fbs': 1, 'restecg': 2, 'thalach': 95, 'exang': 1,
                'oldpeak': 5.0, 'slope': 2, 'ca': 3, 'thal': 7
            },
            "High Risk (Target: 80-89%)": {
                'age': 64, 'sex': 1, 'cp': 1, 'trestbps': 170, 'chol': 300,
                'fbs': 1, 'restecg': 1, 'thalach': 115, 'exang': 0,
                'oldpeak': 3.5, 'slope': 2, 'ca': 2, 'thal': 6
            },
            "Moderate Risk (Target: 60-79%)": {
                'age': 58, 'sex': 1, 'cp': 2, 'trestbps': 155, 'chol': 270,
                'fbs': 1, 'restecg': 1, 'thalach': 130, 'exang': 0,
                'oldpeak': 2.5, 'slope': 2, 'ca': 1, 'thal': 5
            },
            "Low Moderate Risk (Target: 35-59%)":{
                'age': 55, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 240,
                'fbs': 0, 'restecg': 1, 'thalach': 140, 'exang': 0,
                'oldpeak': 1.5, 'slope': 2, 'ca': 1, 'thal': 6
            },
            "Low Risk (Target: <35%)": {
                'age': 32, 'sex': 0, 'cp': 4, 'trestbps': 105, 'chol': 170,
                'fbs': 0, 'restecg': 0, 'thalach': 185, 'exang': 0,
                'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 3
            }
        }
        
        def categorize_risk(prob):
            """Risk categorization matching the prediction system"""
            if prob >= 0.9:
                return "Very High Risk"
            elif prob >= 0.8:
                return "High Risk"
            elif prob >= 0.6:
                return "Moderate Risk"
            elif prob >= 0.35:
                return "Low-Moderate Risk"
            else:
                return "Low Risk"
        
        print("Testing all 5 risk categories:")
        print("-" * 80)
        print(f"{'Category':<25} {'Actual %':<12} {'Predicted Category':<20} {'Status':<10}")
        print("-" * 80)
        
        for category_name, patient_data in test_patients.items():
            patient_df = pd.DataFrame([patient_data])
            patient_df = patient_df[feature_names]
            
            # Scale and predict
            patient_scaled = scaler.transform(patient_df)
            prediction = model.predict(patient_scaled, verbose=0)[0][0]
            predicted_category = categorize_risk(prediction)
            
            # Check if prediction matches intended category
            target_category = category_name.split(" (")[0]
            status = "✅ Match" if target_category in predicted_category else "❌ Miss"
            
            print(f"{target_category:<25} {prediction*100:<12.1f} {predicted_category:<20} {status:<10}")
        
        print("-" * 80)
        
        # Additional boundary testing
        print("\nBoundary Testing (edge cases):")
        print("-" * 60)
        
        boundary_tests = {
            "Minimal Risk": {'age': 25, 'sex': 0, 'cp': 4, 'trestbps': 90, 'chol': 120, 'fbs': 0, 'restecg': 0, 'thalach': 200, 'exang': 0, 'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 3},
            "Maximum Age": {'age': 77, 'sex': 0, 'cp': 4, 'trestbps': 120, 'chol': 200, 'fbs': 0, 'restecg': 0, 'thalach': 160, 'exang': 0, 'oldpeak': 0.5, 'slope': 1, 'ca': 0, 'thal': 3},
            "High Cholesterol Only": {'age': 45, 'sex': 0, 'cp': 4, 'trestbps': 120, 'chol': 400, 'fbs': 0, 'restecg': 0, 'thalach': 170, 'exang': 0, 'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 3},
            "Exercise Angina Only": {'age': 45, 'sex': 0, 'cp': 4, 'trestbps': 120, 'chol': 200, 'fbs': 0, 'restecg': 0, 'thalach': 160, 'exang': 1, 'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 3}
        }
        
        for test_name, patient_data in boundary_tests.items():
            patient_df = pd.DataFrame([patient_data])
            patient_df = patient_df[feature_names]
            patient_scaled = scaler.transform(patient_df)
            prediction = model.predict(patient_scaled, verbose=0)[0][0]
            predicted_category = categorize_risk(prediction)
            
            print(f"{test_name:<20}: {prediction*100:>6.1f}% ({predicted_category})")
        
    except Exception as e:
        print(f"❌ Error in prediction testing: {e}")

def feature_impact_analysis():
    """Analyze impact of individual features"""
    print("\n" + "=" * 60)
    print("FEATURE IMPACT ANALYSIS")
    print("=" * 60)
    
    try:
        model = keras.models.load_model('disease_prediction_model.h5')
        scaler = joblib.load('disease_prediction_scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        # Baseline patient (neutral values)
        baseline = {
            'age': 50, 'sex': 0, 'cp': 3, 'trestbps': 130, 'chol': 200,
            'fbs': 0, 'restecg': 0, 'thalach': 150, 'exang': 0,
            'oldpeak': 1.0, 'slope': 1, 'ca': 0, 'thal': 3
        }
        
        baseline_df = pd.DataFrame([baseline])[feature_names]
        baseline_scaled = scaler.transform(baseline_df)
        baseline_pred = model.predict(baseline_scaled, verbose=0)[0][0]
        
        print(f"Baseline patient risk: {baseline_pred*100:.1f}%")
        print("\nFeature impact when changed to high-risk values:")
        print("-" * 50)
        
        # Test impact of changing each feature
        high_risk_values = {
            'age': 70, 'sex': 1, 'cp': 1, 'trestbps': 180, 'chol': 350,
            'fbs': 1, 'restecg': 2, 'thalach': 100, 'exang': 1,
            'oldpeak': 4.0, 'slope': 2, 'ca': 3, 'thal': 7
        }
        
        for feature in feature_names:
            # Create modified patient
            modified = baseline.copy()
            modified[feature] = high_risk_values[feature]
            
            modified_df = pd.DataFrame([modified])[feature_names]
            modified_scaled = scaler.transform(modified_df)
            modified_pred = model.predict(modified_scaled, verbose=0)[0][0]
            
            impact = modified_pred - baseline_pred
            print(f"{feature:<12}: {modified_pred*100:>6.1f}% (impact: {impact*100:+5.1f}%)")
            
    except Exception as e:
        print(f"❌ Error in feature impact analysis: {e}")

def check_training_data():
    """Check if training data was processed correctly"""
    print("\n" + "=" * 60)
    print("TRAINING DATA CHECK")
    print("=" * 60)
    
    try:
        # Try to reload and process the data the same way as training
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        
        df = pd.read_csv(url, names=column_names)
        df = df.replace('?', np.nan)
        df = df.dropna()
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Binary target
        df['target'] = (df['target'] > 0).astype(int)
        
        print(f"Training data shape: {df.shape}")
        print(f"Target distribution:")
        print(df['target'].value_counts())
        print(f"Target mean: {df['target'].mean():.3f}")
        
        # Check feature ranges
        X = df.drop('target', axis=1)
        print(f"\nFeature ranges:")
        for col in X.columns:
            print(f"  {col}: {X[col].min():.2f} to {X[col].max():.2f}")
            
    except Exception as e:
        print(f"❌ Error checking training data: {e}")

def main():
    """Run all diagnostics"""
    print("DISEASE RISK PREDICTION - MODEL DIAGNOSTICS")
    print("=" * 60)
    
    # Check files
    files_exist = check_model_files()
    
    if not files_exist:
        print("\n❌ Some model files are missing. Please run train_model.py first.")
        return
    
    # Analyze model
    model = analyze_model()
    
    # Analyze scaler
    scaler, features = analyze_scaler()
    
    # Check training data
    check_training_data()
    
    # Test predictions
    test_predictions()
    
    # Feature impact analysis
    feature_impact_analysis()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    if model is not None and scaler is not None:
        print("✅ Model and scaler loaded successfully")
        print("🔍 Check the prediction test results above")
        print("\n💡 If all predictions are very low, the model may need retraining")
        print("   with different hyperparameters or data preprocessing.")
    else:
        print("❌ Issues detected with model or scaler")

if __name__ == "__main__":
    main()