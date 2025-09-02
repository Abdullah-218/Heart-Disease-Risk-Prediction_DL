# Balanced Model Diagnostics Script - 5 Risk Categories
# balanced_model_diagnostics.py

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score

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
            print(f"Model file {file}: EXISTS ({size:,} bytes)")
        else:
            print(f"Model file {file}: MISSING")
    
    return all(os.path.exists(f) for f in files_to_check)

def analyze_model():
    """Analyze the trained model architecture and weights"""
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE ANALYSIS")
    print("=" * 60)
    
    try:
        model = keras.models.load_model('disease_prediction_model.h5')
        print("Model loaded successfully")
        
        print("\nModel Architecture:")
        model.summary()
        
        total_params = model.count_params()
        print(f"\nTotal parameters: {total_params:,}")
        
        # Analyze weights distribution
        layer_stats = []
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'get_weights') and layer.get_weights():
                weights = layer.get_weights()[0]
                layer_stats.append({
                    'layer': f"Layer {i} ({layer.name})",
                    'shape': weights.shape,
                    'mean': np.mean(weights),
                    'std': np.std(weights),
                    'min': np.min(weights),
                    'max': np.max(weights)
                })
        
        print("\nWeight Statistics by Layer:")
        for stats in layer_stats:
            print(f"{stats['layer']}: shape={stats['shape']}, mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def test_five_risk_categories():
    """Comprehensive testing of all 5 risk categories with multiple patients per category"""
    print("\n" + "=" * 60)
    print("5-CATEGORY RISK TESTING")
    print("=" * 60)
    
    try:
        model = keras.models.load_model('disease_prediction_model.h5')
        scaler = joblib.load('disease_prediction_scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        # Define multiple test cases for each risk category
        risk_test_cases = {
            "Very High Risk (90%+)": [
                {'age': 75, 'sex': 1, 'cp': 1, 'trestbps': 190, 'chol': 350, 'fbs': 1, 'restecg': 2, 'thalach': 90, 'exang': 1, 'oldpeak': 5.0, 'slope': 2, 'ca': 3, 'thal': 7},
                {'age': 68, 'sex': 1, 'cp': 1, 'trestbps': 180, 'chol': 320, 'fbs': 1, 'restecg': 2, 'thalach': 95, 'exang': 1, 'oldpeak': 4.5, 'slope': 2, 'ca': 3, 'thal': 6},
                {'age': 72, 'sex': 1, 'cp': 2, 'trestbps': 185, 'chol': 340, 'fbs': 1, 'restecg': 2, 'thalach': 85, 'exang': 1, 'oldpeak': 5.5, 'slope': 2, 'ca': 3, 'thal': 7}
            ],
            "High Risk (80-89%)": [
                {'age': 62, 'sex': 1, 'cp': 1, 'trestbps': 165, 'chol': 290, 'fbs': 1, 'restecg': 1, 'thalach': 115, 'exang': 0, 'oldpeak': 3.0, 'slope': 2, 'ca': 2, 'thal': 6},
                {'age': 60, 'sex': 1, 'cp': 2, 'trestbps': 170, 'chol': 300, 'fbs': 0, 'restecg': 1, 'thalach': 120, 'exang': 1, 'oldpeak': 2.5, 'slope': 2, 'ca': 2, 'thal': 5},
                {'age': 65, 'sex': 0, 'cp': 1, 'trestbps': 160, 'chol': 280, 'fbs': 1, 'restecg': 1, 'thalach': 110, 'exang': 0, 'oldpeak': 3.2, 'slope': 2, 'ca': 2, 'thal': 6}
            ],
            "Moderate Risk (60-79%)": [
                {'age': 55, 'sex': 1, 'cp': 2, 'trestbps': 150, 'chol': 260, 'fbs': 0, 'restecg': 1, 'thalach': 140, 'exang': 0, 'oldpeak': 2.0, 'slope': 1, 'ca': 1, 'thal': 5},
                {'age': 52, 'sex': 0, 'cp': 2, 'trestbps': 145, 'chol': 250, 'fbs': 1, 'restecg': 1, 'thalach': 135, 'exang': 0, 'oldpeak': 1.8, 'slope': 2, 'ca': 1, 'thal': 4},
                {'age': 58, 'sex': 1, 'cp': 3, 'trestbps': 155, 'chol': 270, 'fbs': 0, 'restecg': 0, 'thalach': 145, 'exang': 0, 'oldpeak': 1.5, 'slope': 1, 'ca': 0, 'thal': 5}
            ],
            "Low-Moderate Risk (35-59%)": [
                {'age': 48, 'sex': 0, 'cp': 3, 'trestbps': 135, 'chol': 210, 'fbs': 0, 'restecg': 0, 'thalach': 155, 'exang': 0, 'oldpeak': 1.0, 'slope': 1, 'ca': 0, 'thal': 4},
                {'age': 45, 'sex': 1, 'cp': 3, 'trestbps': 140, 'chol': 220, 'fbs': 0, 'restecg': 1, 'thalach': 150, 'exang': 0, 'oldpeak': 1.2, 'slope': 1, 'ca': 0, 'thal': 3},
                {'age': 50, 'sex': 0, 'cp': 2, 'trestbps': 130, 'chol': 200, 'fbs': 0, 'restecg': 0, 'thalach': 160, 'exang': 0, 'oldpeak': 0.8, 'slope': 1, 'ca': 0, 'thal': 4}
            ],
            "Low Risk (<35%)": [
                {'age': 32, 'sex': 0, 'cp': 4, 'trestbps': 110, 'chol': 180, 'fbs': 0, 'restecg': 0, 'thalach': 180, 'exang': 0, 'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 3},
                {'age': 28, 'sex': 0, 'cp': 4, 'trestbps': 105, 'chol': 170, 'fbs': 0, 'restecg': 0, 'thalach': 190, 'exang': 0, 'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 3},
                {'age': 40, 'sex': 1, 'cp': 4, 'trestbps': 120, 'chol': 185, 'fbs': 0, 'restecg': 0, 'thalach': 175, 'exang': 0, 'oldpeak': 0.2, 'slope': 1, 'ca': 0, 'thal': 3}
            ]
        }
        
        def categorize_risk(prob):
            """Risk categorization matching the 5-tier system"""
            if prob >= 0.9:
                return "Very High Risk", "red"
            elif prob >= 0.8:
                return "High Risk", "orange-red"
            elif prob >= 0.6:
                return "Moderate Risk", "yellow"
            elif prob >= 0.35:
                return "Low-Moderate Risk", "orange"
            else:
                return "Low Risk", "green"
        
        print("Testing multiple patients per risk category:")
        print("=" * 80)
        
        category_results = {}
        
        for category_name, patients in risk_test_cases.items():
            print(f"\n{category_name}:")
            print("-" * 50)
            
            predictions = []
            for i, patient_data in enumerate(patients, 1):
                patient_df = pd.DataFrame([patient_data])[feature_names]
                patient_scaled = scaler.transform(patient_df)
                prediction = model.predict(patient_scaled, verbose=0)[0][0]
                predicted_category, color = categorize_risk(prediction)
                
                predictions.append(prediction)
                
                status = "Match" if category_name.split(" (")[0] in predicted_category else "Miss"
                print(f"  Patient {i}: {prediction*100:>6.1f}% -> {predicted_category:<18} ({status})")
            
            avg_prediction = np.mean(predictions)
            category_results[category_name] = {
                'predictions': predictions,
                'average': avg_prediction,
                'min': min(predictions),
                'max': max(predictions)
            }
            
            print(f"  Average: {avg_prediction*100:>6.1f}% (Range: {min(predictions)*100:.1f}%-{max(predictions)*100:.1f}%)")
        
        # Summary analysis
        print("\n" + "=" * 60)
        print("CATEGORY PERFORMANCE SUMMARY")
        print("=" * 60)
        
        target_ranges = {
            "Very High Risk (90%+)": (0.9, 1.0),
            "High Risk (80-89%)": (0.8, 0.89),
            "Moderate Risk (60-79%)": (0.6, 0.79),
            "Low-Moderate Risk (35-59%)": (0.35, 0.59),
            "Low Risk (<35%)": (0.0, 0.34)
        }
        
        for category, results in category_results.items():
            target_min, target_max = target_ranges[category]
            avg = results['average']
            
            if target_min <= avg <= target_max:
                status = "PASS"
            else:
                status = "FAIL"
            
            print(f"{category:<25}: Avg={avg*100:>6.1f}% (Target: {target_min*100:.0f}-{target_max*100:.0f}%) [{status}]")
        
        return category_results
        
    except Exception as e:
        print(f"Error in 5-category testing: {e}")
        return None

def test_feature_sensitivity():
    """Test how sensitive the model is to individual feature changes"""
    print("\n" + "=" * 60)
    print("FEATURE SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    try:
        model = keras.models.load_model('disease_prediction_model.h5')
        scaler = joblib.load('disease_prediction_scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        # Define a neutral baseline patient
        baseline = {
            'age': 50, 'sex': 0, 'cp': 3, 'trestbps': 130, 'chol': 200,
            'fbs': 0, 'restecg': 0, 'thalach': 150, 'exang': 0,
            'oldpeak': 1.0, 'slope': 1, 'ca': 0, 'thal': 3
        }
        
        baseline_df = pd.DataFrame([baseline])[feature_names]
        baseline_scaled = scaler.transform(baseline_df)
        baseline_pred = model.predict(baseline_scaled, verbose=0)[0][0]
        
        print(f"Baseline patient risk: {baseline_pred*100:.2f}%")
        
        # Test each feature at different levels
        feature_tests = {
            'age': [30, 40, 50, 60, 70],
            'sex': [0, 1],
            'cp': [1, 2, 3, 4],
            'trestbps': [100, 120, 140, 160, 180],
            'chol': [150, 200, 250, 300, 350],
            'fbs': [0, 1],
            'restecg': [0, 1, 2],
            'thalach': [100, 125, 150, 175, 200],
            'exang': [0, 1],
            'oldpeak': [0.0, 1.0, 2.0, 3.0, 4.0],
            'slope': [1, 2, 3],
            'ca': [0, 1, 2, 3],
            'thal': [3, 4, 5, 6, 7]
        }
        
        print("\nFeature Impact Analysis:")
        print("-" * 70)
        print(f"{'Feature':<12} {'Values Tested':<25} {'Risk Range':<15} {'Max Impact':<12}")
        print("-" * 70)
        
        for feature, test_values in feature_tests.items():
            predictions = []
            
            for value in test_values:
                modified = baseline.copy()
                modified[feature] = value
                
                modified_df = pd.DataFrame([modified])[feature_names]
                modified_scaled = scaler.transform(modified_df)
                pred = model.predict(modified_scaled, verbose=0)[0][0]
                predictions.append(pred)
            
            min_risk = min(predictions) * 100
            max_risk = max(predictions) * 100
            max_impact = (max(predictions) - min(predictions)) * 100
            
            print(f"{feature:<12} {str(test_values):<25} {min_risk:>5.1f}-{max_risk:<5.1f}% {max_impact:>8.1f}%")
        
    except Exception as e:
        print(f"Error in feature sensitivity analysis: {e}")

def test_risk_progression():
    """Test risk progression across patient age groups"""
    print("\n" + "=" * 60)
    print("RISK PROGRESSION BY AGE ANALYSIS")
    print("=" * 60)
    
    try:
        model = keras.models.load_model('disease_prediction_model.h5')
        scaler = joblib.load('disease_prediction_scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        # Test risk progression with age for different patient profiles
        base_profiles = {
            "Healthy Female": {'sex': 0, 'cp': 4, 'trestbps': 120, 'chol': 200, 'fbs': 0, 'restecg': 0, 'thalach': 160, 'exang': 0, 'oldpeak': 0.5, 'slope': 1, 'ca': 0, 'thal': 3},
            "Healthy Male": {'sex': 1, 'cp': 4, 'trestbps': 130, 'chol': 210, 'fbs': 0, 'restecg': 0, 'thalach': 150, 'exang': 0, 'oldpeak': 0.8, 'slope': 1, 'ca': 0, 'thal': 3},
            "At-Risk Male": {'sex': 1, 'cp': 2, 'trestbps': 150, 'chol': 250, 'fbs': 1, 'restecg': 1, 'thalach': 130, 'exang': 0, 'oldpeak': 1.5, 'slope': 2, 'ca': 1, 'thal': 5}
        }
        
        ages_to_test = [30, 40, 50, 60, 70]
        
        print("Risk progression by age:")
        print("-" * 60)
        print(f"{'Profile':<15} {'Age 30':<8} {'Age 40':<8} {'Age 50':<8} {'Age 60':<8} {'Age 70':<8}")
        print("-" * 60)
        
        for profile_name, base_profile in base_profiles.items():
            risk_by_age = []
            
            for age in ages_to_test:
                patient = base_profile.copy()
                patient['age'] = age
                
                patient_df = pd.DataFrame([patient])[feature_names]
                patient_scaled = scaler.transform(patient_df)
                prediction = model.predict(patient_scaled, verbose=0)[0][0]
                risk_by_age.append(prediction * 100)
            
            print(f"{profile_name:<15} {risk_by_age[0]:<8.1f} {risk_by_age[1]:<8.1f} {risk_by_age[2]:<8.1f} {risk_by_age[3]:<8.1f} {risk_by_age[4]:<8.1f}")
        
    except Exception as e:
        print(f"Error in risk progression analysis: {e}")

def validate_risk_distribution():
    """Validate that the model can produce predictions across all risk ranges"""
    print("\n" + "=" * 60)
    print("RISK DISTRIBUTION VALIDATION")
    print("=" * 60)
    
    try:
        model = keras.models.load_model('disease_prediction_model.h5')
        scaler = joblib.load('disease_prediction_scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        # Generate random patients and check distribution
        print("Generating 100 random patients to check risk distribution...")
        
        random_patients = []
        for i in range(100):
            patient = {
                'age': np.random.randint(30, 75),
                'sex': np.random.randint(0, 2),
                'cp': np.random.randint(1, 5),
                'trestbps': np.random.randint(100, 180),
                'chol': np.random.randint(150, 350),
                'fbs': np.random.randint(0, 2),
                'restecg': np.random.randint(0, 3),
                'thalach': np.random.randint(100, 190),
                'exang': np.random.randint(0, 2),
                'oldpeak': np.random.uniform(0, 4),
                'slope': np.random.randint(1, 4),
                'ca': np.random.randint(0, 4),
                'thal': np.random.randint(3, 8)
            }
            random_patients.append(patient)
        
        # Make predictions
        patients_df = pd.DataFrame(random_patients)[feature_names]
        patients_scaled = scaler.transform(patients_df)
        predictions = model.predict(patients_scaled, verbose=0).flatten()
        
        # Categorize predictions
        very_high = np.sum(predictions >= 0.9)
        high = np.sum((predictions >= 0.8) & (predictions < 0.9))
        moderate = np.sum((predictions >= 0.6) & (predictions < 0.8))
        low_moderate = np.sum((predictions >= 0.35) & (predictions < 0.6))
        low = np.sum(predictions < 0.35)
        
        print(f"\nRisk distribution across 100 random patients:")
        print(f"Very High Risk (90%+):     {very_high:>3d} patients ({very_high}%)")
        print(f"High Risk (80-89%):        {high:>3d} patients ({high}%)")
        print(f"Moderate Risk (60-79%):    {moderate:>3d} patients ({moderate}%)")
        print(f"Low-Moderate Risk (35-59%): {low_moderate:>3d} patients ({low_moderate}%)")
        print(f"Low Risk (<35%):           {low:>3d} patients ({low}%)")
        
        print(f"\nPrediction statistics:")
        print(f"Min: {predictions.min()*100:.1f}%")
        print(f"Max: {predictions.max()*100:.1f}%")
        print(f"Mean: {predictions.mean()*100:.1f}%")
        print(f"Median: {np.median(predictions)*100:.1f}%")
        print(f"Std: {predictions.std()*100:.1f}%")
        
    except Exception as e:
        print(f"Error in risk distribution validation: {e}")

def model_performance_metrics():
    """Calculate and display comprehensive model performance metrics"""
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    
    try:
        model = keras.models.load_model('disease_prediction_model.h5')
        scaler = joblib.load('disease_prediction_scaler.pkl')
        
        # Test with the original dataset if available
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
            
            df = pd.read_csv(url, names=column_names)
            df = df.replace('?', np.nan).dropna()
            
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            df['target'] = (df['target'] > 0).astype(int)
            
            X = df.drop('target', axis=1)
            y = df['target']
            
            X_scaled = scaler.transform(X)
            y_pred_prob = model.predict(X_scaled, verbose=0).flatten()
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            accuracy = accuracy_score(y, y_pred)
            
            print(f"Performance on original UCI dataset:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"AUC: {roc_auc_score(y, y_pred_prob):.4f}")
            
            # Check prediction distribution on real data
            very_high = np.sum(y_pred_prob >= 0.9)
            high = np.sum((y_pred_prob >= 0.8) & (y_pred_prob < 0.9))
            moderate = np.sum((y_pred_prob >= 0.6) & (y_pred_prob < 0.8))
            low_moderate = np.sum((y_pred_prob >= 0.35) & (y_pred_prob < 0.6))
            low = np.sum(y_pred_prob < 0.35)
            
            total = len(y_pred_prob)
            print(f"\nPrediction distribution on real data ({total} patients):")
            print(f"Very High Risk: {very_high} ({very_high/total*100:.1f}%)")
            print(f"High Risk: {high} ({high/total*100:.1f}%)")
            print(f"Moderate Risk: {moderate} ({moderate/total*100:.1f}%)")
            print(f"Low-Moderate Risk: {low_moderate} ({low_moderate/total*100:.1f}%)")
            print(f"Low Risk: {low} ({low/total*100:.1f}%)")
            
        except Exception as e:
            print(f"Could not test on original dataset: {e}")
    
    except Exception as e:
        print(f"Error in performance metrics: {e}")

def main():
    """Run comprehensive diagnostics for the balanced model"""
    print("BALANCED MODEL DIAGNOSTICS - 5 RISK CATEGORIES")
    print("=" * 60)
    
    # Check files
    files_exist = check_model_files()
    if not files_exist:
        print("\nSome model files are missing. Please run retrain_balanced.py first.")
        return
    
    # Analyze model architecture
    model = analyze_model()
    if model is None:
        return
    
    # Test 5 risk categories with multiple patients each
    category_results = test_five_risk_categories()
    
    # Feature sensitivity analysis
    test_feature_sensitivity()
    
    # Risk progression analysis
    test_risk_progression()
    
    # Risk distribution validation
    validate_risk_distribution()
    
    # Performance metrics
    model_performance_metrics()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    if category_results:
        print("Model diagnostic completed successfully.")
        print("Check the category performance summary above to see")
        print("if the model produces appropriate predictions for each risk level.")
        
        # Count successful categories
        target_ranges = {
            "Very High Risk (90%+)": (0.9, 1.0),
            "High Risk (80-89%)": (0.8, 0.89),
            "Moderate Risk (60-79%)": (0.6, 0.79),
            "Low-Moderate Risk (35-59%)": (0.35, 0.59),
            "Low Risk (<35%)": (0.0, 0.34)
        }
        
        successful_categories = 0
        for category, results in category_results.items():
            target_min, target_max = target_ranges[category]
            if target_min <= results['average'] <= target_max:
                successful_categories += 1
        
        print(f"\nSuccessful risk categories: {successful_categories}/5")
        
        if successful_categories >= 4:
            print("Model appears to be working well across risk categories.")
        elif successful_categories >= 2:
            print("Model shows partial success - may need fine-tuning.")
        else:
            print("Model may need retraining with different approach.")
    else:
        print("Issues detected with model diagnostics.")

if __name__ == "__main__":
    main()