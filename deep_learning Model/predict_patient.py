# Disease Risk Prediction - Patient Prediction
# predict_patient.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import warnings
warnings.filterwarnings('ignore')

class DiseaseRiskPredictor:
    def __init__(self, model_path='disease_prediction_model.h5', 
                 scaler_path='disease_prediction_scaler.pkl',
                 features_path='feature_names.pkl'):
        """Initialize the predictor with saved model and scaler"""
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # Load the trained model and scaler
        self.load_model_and_scaler(model_path, scaler_path, features_path)
    
    def load_model_and_scaler(self, model_path, scaler_path, features_path):
        """Load the trained model, scaler, and feature names"""
        try:
            # Load model
            print(f"Loading model from {model_path}...")
            self.model = keras.models.load_model(model_path)
            print("Model loaded successfully!")
            
            # Load scaler
            print(f"Loading scaler from {scaler_path}...")
            self.scaler = joblib.load(scaler_path)
            print("Scaler loaded successfully!")
            
            # Load feature names
            print(f"Loading feature names from {features_path}...")
            self.feature_names = joblib.load(features_path)
            print("Feature names loaded successfully!")
            
            print(f"Model expects {len(self.feature_names)} features: {self.feature_names}")
            
        except FileNotFoundError as e:
            print(f"Error: Model files not found. Please run train_model.py first.")
            print(f"Missing file: {e}")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict_single_patient(self, patient_data):
        """Predict disease risk for a single patient"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded. Please ensure model files exist.")
        
        # Convert to DataFrame if dictionary
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data.copy()
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(patient_df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Ensure correct feature order
        patient_df = patient_df[self.feature_names]
        
        # Scale features
        patient_scaled = self.scaler.transform(patient_df)
        
        # Predict
        risk_prob = self.model.predict(patient_scaled, verbose=0)[0][0]
        
    def _categorize_risk(self, risk_prob):
        """Centralized risk categorization logic with 5-tier system"""
        if risk_prob >= 0.9:
            return "Very High Risk", "Urgent cardiology consultation - potential emergency", "🟥"
        elif risk_prob >= 0.8:
            return "High Risk", "Immediate consultation with cardiologist recommended", "🔴"
        elif risk_prob >= 0.6:
            return "Moderate Risk", "Regular monitoring and lifestyle changes advised", "🟡"
        elif risk_prob >= 0.35:
            return "Low-Moderate Risk", "Consider preventive measures and regular check-ups", "🟠"
        else:
            return "Low Risk", "Continue healthy lifestyle practices", "🟢"
    
    def predict_single_patient(self, patient_data):
        """Predict disease risk for a single patient"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded. Please ensure model files exist.")
        
        # Convert to DataFrame if dictionary
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data.copy()
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(patient_df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Ensure correct feature order
        patient_df = patient_df[self.feature_names]
        
        # Scale features
        patient_scaled = self.scaler.transform(patient_df)
        
        # Predict
        risk_prob = self.model.predict(patient_scaled, verbose=0)[0][0]
        
        # Use centralized risk categorization
        risk_level, recommendation, color_code = self._categorize_risk(risk_prob)
        
        return {
            'risk_probability': float(risk_prob),
            'risk_percentage': float(risk_prob * 100),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'color_code': color_code
        }
    
    def predict_batch_patients(self, patients_df):
        """Predict disease risk for multiple patients"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded. Please ensure model files exist.")
        
        # Ensure correct feature order
        patients_df = patients_df[self.feature_names]
        
        # Scale features
        patients_scaled = self.scaler.transform(patients_df)
        
        # Predict
        risk_probs = self.model.predict(patients_scaled, verbose=0).flatten()
        
        # Create results dataframe
        results = pd.DataFrame({
            'patient_id': range(1, len(patients_df) + 1),
            'risk_probability': risk_probs,
            'risk_percentage': risk_probs * 100
        })
        
        # Add risk levels using the same centralized logic
        results['risk_level'] = results['risk_probability'].apply(
            lambda x: self._categorize_risk(x)[0]  # Get just the risk level
        )
        
        return results

def interactive_prediction(predictor):
    """Interactive function for making predictions"""
    print("\n" + "="*60)
    print("🏥 INTERACTIVE DISEASE RISK PREDICTION 🏥")
    print("="*60)
    
    print("Enter patient information (press Enter for default values):")
    print("Note: All values should be numeric")
    
    # Feature descriptions and prompts
    feature_info = {
        'age': {
            'prompt': "Age in years (29-77)",
            'default': 50,
            'description': "Patient's age"
        },
        'sex': {
            'prompt': "Sex (1=male, 0=female)",
            'default': 1,
            'description': "Biological sex"
        },
        'cp': {
            'prompt': "Chest pain type (0=typical angina, 1=atypical angina, 2=non-anginal, 3=asymptomatic)",
            'default': 0,
            'description': "Type of chest pain experienced"
        },
        'trestbps': {
            'prompt': "Resting blood pressure in mm Hg (94-200)",
            'default': 120,
            'description': "Blood pressure when at rest"
        },
        'chol': {
            'prompt': "Serum cholesterol in mg/dl (126-564)",
            'default': 200,
            'description': "Total cholesterol level"
        },
        'fbs': {
            'prompt': "Fasting blood sugar > 120 mg/dl (1=true, 0=false)",
            'default': 0,
            'description': "Whether fasting blood sugar > 120 mg/dl"
        },
        'restecg': {
            'prompt': "Resting ECG results (0=normal, 1=ST-T abnormality, 2=LV hypertrophy)",
            'default': 0,
            'description': "Resting electrocardiographic results"
        },
        'thalach': {
            'prompt': "Maximum heart rate achieved (71-202)",
            'default': 150,
            'description': "Maximum heart rate during exercise test"
        },
        'exang': {
            'prompt': "Exercise induced angina (1=yes, 0=no)",
            'default': 0,
            'description': "Whether exercise induces chest pain"
        },
        'oldpeak': {
            'prompt': "ST depression induced by exercise (0.0-6.2)",
            'default': 1.0,
            'description': "ST segment depression during exercise"
        },
        'slope': {
            'prompt': "Slope of peak exercise ST segment (0=upsloping, 1=flat, 2=downsloping)",
            'default': 1,
            'description': "Slope of ST segment during peak exercise"
        },
        'ca': {
            'prompt': "Number of major vessels colored by fluoroscopy (0-3)",
            'default': 0,
            'description': "Number of major blood vessels visible in fluoroscopy"
        },
        'thal': {
            'prompt': "Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)",
            'default': 2,
            'description': "Blood disorder thalassemia status"
        }
    }
    
    # Collect patient data
    patient_data = {}
    
    print("\n📋 Please provide the following information:")
    print("-" * 50)
    
    for feature, info in feature_info.items():
        while True:
            try:
                user_input = input(f"\n{info['description']}\n{info['prompt']} [default: {info['default']}]: ").strip()
                
                if user_input == "":
                    patient_data[feature] = info['default']
                    break
                else:
                    value = float(user_input)
                    patient_data[feature] = value
                    break
                    
            except ValueError:
                print("❌ Invalid input! Please enter a numeric value.")
                continue
    
    print("\n" + "="*50)
    print("📊 PATIENT DATA SUMMARY")
    print("="*50)
    
    for feature, value in patient_data.items():
        desc = feature_info[feature]['description']
        print(f"{desc}: {value}")
    
    # Make prediction
    print("\n🔮 Making prediction...")
    try:
        result = predictor.predict_single_patient(patient_data)
        
        print("\n" + "="*50)
        print("📈 PREDICTION RESULTS")
        print("="*50)
        
        print(f"\n{result['color_code']} Risk Assessment:")
        print(f"   • Risk Probability: {result['risk_probability']:.3f}")
        print(f"   • Risk Percentage: {result['risk_percentage']:.1f}%")
        print(f"   • Risk Level: {result['risk_level']}")
        print(f"\n💡 Recommendation:")
        print(f"   {result['recommendation']}")
        
        # Additional risk interpretation
        print(f"\n📋 Risk Interpretation:")
        if result['risk_probability'] >= 0.7:
            print("   ⚠️  This indicates a HIGH probability of heart disease.")
            print("   🏥 Immediate medical attention is strongly recommended.")
            print("   📞 Consider scheduling an urgent appointment with a cardiologist.")
        elif result['risk_probability'] >= 0.4:
            print("   ⚡ This indicates a MODERATE risk of heart disease.")
            print("   🔍 Regular health monitoring is advised.")
            print("   🏃‍♂️ Consider lifestyle modifications and follow-up with your doctor.")
        else:
            print("   ✅ This indicates a LOW risk of heart disease.")
            print("   🌟 Continue maintaining a healthy lifestyle.")
            print("   📅 Regular check-ups are still recommended for overall health.")
            
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        
def load_sample_patients():
    """Load sample patient data for demonstration"""
    sample_patients = pd.DataFrame([
        {
            # HIGH RISK: Older male with multiple risk factors
            'age': 65, 'sex': 1, 'cp': 1, 'trestbps': 160, 'chol': 280,
            'fbs': 1, 'restecg': 1, 'thalach': 120, 'exang': 1,
            'oldpeak': 3.0, 'slope': 2, 'ca': 2, 'thal': 6,
            'description': 'High-risk patient (65-year-old male with multiple risk factors)'
        },
        {
            # MODERATE RISK: Profile based on diagnostic insights
            'age': 62, 'sex': 1, 'cp': 2, 'trestbps': 155, 'chol': 265,
            'fbs': 1, 'restecg': 1, 'thalach': 130, 'exang': 0,
            'oldpeak': 2.2, 'slope': 2, 'ca': 1, 'thal': 5,
            'description': 'Moderate-risk patient (62-year-old male with several risk factors)'
        },
        {
            # LOW RISK: Young female with minimal risk factors
            'age': 35, 'sex': 0, 'cp': 4, 'trestbps': 110, 'chol': 180,
            'fbs': 0, 'restecg': 0, 'thalach': 180, 'exang': 0,
            'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 3,
            'description': 'Low-risk patient (35-year-old female with minimal risk factors)'
        }
    ])
    
    return sample_patients

def batch_prediction_demo(predictor):
    """Demonstrate batch predictions with sample patients"""
    print("\n" + "="*60)
    print("🔬 BATCH PREDICTION DEMONSTRATION")
    print("="*60)
    
    # Load sample patients
    sample_patients = load_sample_patients()
    
    print("Sample patients loaded:")
    for idx, row in sample_patients.iterrows():
        print(f"  {idx + 1}. {row['description']}")
    
    # Remove description column for prediction
    prediction_data = sample_patients.drop('description', axis=1)
    
    # Make batch predictions
    try:
        results = predictor.predict_batch_patients(prediction_data)
        
        print("\n📊 Batch Prediction Results:")
        print("-" * 80)
        print(f"{'Patient':<10} {'Description':<35} {'Risk %':<10} {'Risk Level':<15}")
        print("-" * 80)
        
        for idx, (_, result_row) in enumerate(results.iterrows()):
            desc = sample_patients.iloc[idx]['description']
            risk_pct = result_row['risk_percentage']
            risk_level = result_row['risk_level']
            
            # Get color coding using the centralized function
            _, _, color = predictor._categorize_risk(result_row['risk_probability'])
                
            print(f"{color} Patient {idx+1:<5} {desc:<55} {risk_pct:<10.1f} {risk_level:<15}")
        
        print("-" * 80)
        
    except Exception as e:
        print(f"❌ Error in batch prediction: {e}")

def show_model_info(predictor):
    """Display information about the loaded model"""
    print("\n" + "="*60)
    print("🤖 MODEL INFORMATION")
    print("="*60)
    
    if predictor.model is not None:
        print(f"✅ Model Status: Loaded and Ready")
        print(f"📊 Model Type: Deep Neural Network")
        print(f"🎯 Purpose: Heart Disease Risk Prediction")
        print(f"📐 Input Features: {len(predictor.feature_names)}")
        print(f"🔧 Features: {', '.join(predictor.feature_names)}")
        
        # Model architecture summary
        print(f"\n🏗️ Model Architecture:")
        predictor.model.summary(print_fn=lambda x: print(f"   {x}"))
        
    else:
        print("❌ Model Status: Not Loaded")

def main():
    """Main function for patient prediction"""
    print("="*70)
    print("🏥 DISEASE RISK PREDICTION - PATIENT PREDICTION SYSTEM 🏥")
    print("="*70)
    
    try:
        # Initialize predictor (will load model, scaler, and feature names)
        print("🚀 Initializing prediction system...")
        predictor = DiseaseRiskPredictor()
        print("✅ System initialized successfully!\n")
        
        while True:
            print("\n" + "="*50)
            print("📋 MAIN MENU")
            print("="*50)
            print("1. 🔮 Make Single Patient Prediction")
            print("2. 📊 Batch Prediction Demo")
            print("3. ℹ️  Show Model Information")
            print("4. ❓ Show Feature Descriptions")
            print("5. 🚪 Exit")
            print("-" * 50)
            
            choice = input("Please select an option (1-5): ").strip()
            
            if choice == '1':
                interactive_prediction(predictor)
                
            elif choice == '2':
                batch_prediction_demo(predictor)
                
            elif choice == '3':
                show_model_info(predictor)
                
            elif choice == '4':
                show_feature_descriptions()
                
            elif choice == '5':
                print("\n👋 Thank you for using the Disease Risk Prediction System!")
                print("💙 Stay healthy and take care!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-5.")
                
            # Ask if user wants to continue
            if choice in ['1', '2', '3', '4']:
                input("\n⏸️  Press Enter to return to main menu...")
                
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        print("💡 Please ensure you have run train_model.py first to create the necessary model files.")

def show_feature_descriptions():
    """Show detailed descriptions of all features"""
    print("\n" + "="*60)
    print("📚 FEATURE DESCRIPTIONS")
    print("="*60)
    
    descriptions = {
        'age': "Age of the patient in years (typically 29-77)",
        'sex': "Biological sex (1 = male, 0 = female)",
        'cp': "Chest pain type:\n     • 0: Typical angina\n     • 1: Atypical angina\n     • 2: Non-anginal pain\n     • 3: Asymptomatic",
        'trestbps': "Resting blood pressure in mm Hg (typically 94-200)",
        'chol': "Serum cholesterol in mg/dl (typically 126-564)",
        'fbs': "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
        'restecg': "Resting electrocardiographic results:\n     • 0: Normal\n     • 1: Having ST-T wave abnormality\n     • 2: Left ventricular hypertrophy",
        'thalach': "Maximum heart rate achieved during exercise (typically 71-202)",
        'exang': "Exercise induced angina (1 = yes, 0 = no)",
        'oldpeak': "ST depression induced by exercise relative to rest (typically 0.0-6.2)",
        'slope': "Slope of the peak exercise ST segment:\n     • 0: Upsloping\n     • 1: Flat\n     • 2: Downsloping",
        'ca': "Number of major vessels colored by fluoroscopy (0-3)",
        'thal': "Thalassemia blood disorder:\n     • 1: Normal\n     • 2: Fixed defect\n     • 3: Reversible defect"
    }
    
    for i, (feature, description) in enumerate(descriptions.items(), 1):
        print(f"\n{i:2d}. {feature.upper()}:")
        print(f"    {description}")
    
    print("\n" + "="*60)
    print("💡 These features are used by the AI model to assess heart disease risk.")
    print("🏥 Always consult with medical professionals for proper diagnosis.")

if __name__ == "__main__":
    main()