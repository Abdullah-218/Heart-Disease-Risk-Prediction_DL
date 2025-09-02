# Balanced Disease Risk Prediction Model
# retrain_balanced.py

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib backend
plt.switch_backend('Agg')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

class BalancedDiseaseRiskTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def create_balanced_dataset(self, n_samples=1000):
        """Create a balanced synthetic dataset with diverse risk profiles"""
        print("Creating balanced synthetic dataset...")
        
        np.random.seed(42)
        
        # Define risk categories and their proportions
        risk_categories = {
            'very_high': {'n': 150, 'target_prob': 0.95},  # 90%+ risk
            'high': {'n': 200, 'target_prob': 0.85},       # 80-90% risk
            'moderate': {'n': 250, 'target_prob': 0.70},   # 60-80% risk
            'low_moderate': {'n': 250, 'target_prob': 0.45}, # 35-60% risk
            'low': {'n': 150, 'target_prob': 0.15}         # <35% risk
        }
        
        all_data = []
        all_targets = []
        
        for category, config in risk_categories.items():
            n = config['n']
            target_prob = config['target_prob']
            
            if category == 'very_high':
                # Very high risk: Elderly males with severe abnormalities
                data = self._generate_risk_group(
                    n=n,
                    age_range=(65, 77), age_mean=70,
                    sex_prob=0.8,  # 80% male
                    cp_values=[1, 2], cp_probs=[0.7, 0.3],
                    bp_range=(160, 200), bp_mean=175,
                    chol_range=(280, 400), chol_mean=320,
                    fbs_prob=0.6,
                    restecg_values=[1, 2], restecg_probs=[0.6, 0.4],
                    thalach_range=(80, 130), thalach_mean=105,
                    exang_prob=0.7,
                    oldpeak_range=(3.0, 6.0), oldpeak_mean=4.0,
                    slope_values=[2, 3], slope_probs=[0.8, 0.2],
                    ca_values=[2, 3], ca_probs=[0.6, 0.4],
                    thal_values=[6, 7], thal_probs=[0.6, 0.4]
                )
                
            elif category == 'high':
                # High risk: Older adults with multiple risk factors
                data = self._generate_risk_group(
                    n=n,
                    age_range=(55, 70), age_mean=62,
                    sex_prob=0.7,
                    cp_values=[1, 2, 3], cp_probs=[0.4, 0.4, 0.2],
                    bp_range=(145, 180), bp_mean=160,
                    chol_range=(250, 320), chol_mean=280,
                    fbs_prob=0.4,
                    restecg_values=[0, 1, 2], restecg_probs=[0.3, 0.6, 0.1],
                    thalach_range=(100, 140), thalach_mean=120,
                    exang_prob=0.4,
                    oldpeak_range=(2.0, 4.0), oldpeak_mean=2.8,
                    slope_values=[1, 2], slope_probs=[0.3, 0.7],
                    ca_values=[1, 2], ca_probs=[0.6, 0.4],
                    thal_values=[5, 6], thal_probs=[0.6, 0.4]
                )
                
            elif category == 'moderate':
                # Moderate risk: Middle-aged with some risk factors
                data = self._generate_risk_group(
                    n=n,
                    age_range=(45, 65), age_mean=55,
                    sex_prob=0.6,
                    cp_values=[2, 3, 4], cp_probs=[0.4, 0.4, 0.2],
                    bp_range=(130, 160), bp_mean=145,
                    chol_range=(220, 280), chol_mean=250,
                    fbs_prob=0.3,
                    restecg_values=[0, 1], restecg_probs=[0.6, 0.4],
                    thalach_range=(120, 160), thalach_mean=140,
                    exang_prob=0.2,
                    oldpeak_range=(1.0, 3.0), oldpeak_mean=1.8,
                    slope_values=[1, 2], slope_probs=[0.5, 0.5],
                    ca_values=[0, 1], ca_probs=[0.7, 0.3],
                    thal_values=[4, 5], thal_probs=[0.6, 0.4]
                )
                
            elif category == 'low_moderate':
                # Low-moderate risk: Mixed demographics with mild factors
                data = self._generate_risk_group(
                    n=n,
                    age_range=(35, 60), age_mean=47,
                    sex_prob=0.4,
                    cp_values=[3, 4], cp_probs=[0.6, 0.4],
                    bp_range=(120, 145), bp_mean=130,
                    chol_range=(190, 240), chol_mean=215,
                    fbs_prob=0.2,
                    restecg_values=[0, 1], restecg_probs=[0.8, 0.2],
                    thalach_range=(140, 180), thalach_mean=160,
                    exang_prob=0.1,
                    oldpeak_range=(0.5, 2.0), oldpeak_mean=1.0,
                    slope_values=[1, 2], slope_probs=[0.7, 0.3],
                    ca_values=[0], ca_probs=[1.0],
                    thal_values=[3, 4], thal_probs=[0.7, 0.3]
                )
                
            else:  # low risk
                # Low risk: Young, healthy individuals
                data = self._generate_risk_group(
                    n=n,
                    age_range=(25, 50), age_mean=37,
                    sex_prob=0.3,
                    cp_values=[4], cp_probs=[1.0],
                    bp_range=(90, 130), bp_mean=115,
                    chol_range=(140, 200), chol_mean=170,
                    fbs_prob=0.05,
                    restecg_values=[0], restecg_probs=[1.0],
                    thalach_range=(160, 200), thalach_mean=180,
                    exang_prob=0.05,
                    oldpeak_range=(0.0, 1.0), oldpeak_mean=0.2,
                    slope_values=[1], slope_probs=[1.0],
                    ca_values=[0], ca_probs=[1.0],
                    thal_values=[3], thal_probs=[1.0]
                )
            
            # Generate targets with some noise around target probability
            targets = np.random.binomial(1, target_prob, n)
            
            all_data.extend(data)
            all_targets.extend(targets)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df['target'] = all_targets
        
        print(f"Generated dataset shape: {df.shape}")
        print(f"Target distribution: {df['target'].value_counts().to_dict()}")
        print(f"Target percentage: {df['target'].mean()*100:.1f}%")
        
        return df
    
    def _generate_risk_group(self, n, age_range, age_mean, sex_prob, cp_values, cp_probs,
                            bp_range, bp_mean, chol_range, chol_mean, fbs_prob,
                            restecg_values, restecg_probs, thalach_range, thalach_mean,
                            exang_prob, oldpeak_range, oldpeak_mean, slope_values, slope_probs,
                            ca_values, ca_probs, thal_values, thal_probs):
        """Generate a group of patients with specific risk characteristics"""
        
        data = []
        for i in range(n):
            patient = {
                'age': int(np.clip(np.random.normal(age_mean, 5), age_range[0], age_range[1])),
                'sex': np.random.binomial(1, sex_prob),
                'cp': np.random.choice(cp_values, p=cp_probs),
                'trestbps': int(np.clip(np.random.normal(bp_mean, 10), bp_range[0], bp_range[1])),
                'chol': int(np.clip(np.random.normal(chol_mean, 25), chol_range[0], chol_range[1])),
                'fbs': np.random.binomial(1, fbs_prob),
                'restecg': np.random.choice(restecg_values, p=restecg_probs),
                'thalach': int(np.clip(np.random.normal(thalach_mean, 15), thalach_range[0], thalach_range[1])),
                'exang': np.random.binomial(1, exang_prob),
                'oldpeak': np.clip(np.random.normal(oldpeak_mean, 0.5), oldpeak_range[0], oldpeak_range[1]),
                'slope': np.random.choice(slope_values, p=slope_probs),
                'ca': np.random.choice(ca_values, p=ca_probs),
                'thal': np.random.choice(thal_values, p=thal_probs)
            }
            data.append(patient)
        
        return data
    
    def create_balanced_model(self, input_dim):
        """Create a simpler, more balanced model architecture"""
        model = keras.Sequential([
            # Simpler architecture with less regularization
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dropout(0.1),  # Much lower dropout
            
            Dense(32, activation='relu'),
            Dropout(0.1),
            
            Dense(16, activation='relu'),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        # Use a higher learning rate for better convergence
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),  # Higher learning rate
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_balanced_model(self):
        """Train the model with balanced dataset"""
        print("="*60)
        print("TRAINING BALANCED DISEASE RISK PREDICTION MODEL")
        print("="*60)
        
        # Create balanced dataset
        df = self.create_balanced_dataset()
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Training target distribution: {y_train.value_counts().to_dict()}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create model
        self.model = self.create_balanced_model(X_train_scaled.shape[1])
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Train model with fewer restrictions
        print("\nStarting training...")
        history = self.model.fit(
            X_train_scaled, y_train,
            batch_size=32,
            epochs=100,  # More epochs
            validation_split=0.2,
            callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
            verbose=1
        )
        
        # Evaluate model
        print("\nEvaluating model...")
        y_pred_prob = self.model.predict(X_test_scaled)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_prob)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        print(f"Training completed in {len(history.history['loss'])} epochs")
        
        # Test probability distribution
        print("\nTesting probability distribution:")
        test_probs = y_pred_prob.flatten()
        print(f"Min prediction: {test_probs.min():.3f}")
        print(f"Max prediction: {test_probs.max():.3f}")
        print(f"Mean prediction: {test_probs.mean():.3f}")
        print(f"Std prediction: {test_probs.std():.3f}")
        
        # Show distribution across risk categories
        very_high = np.sum(test_probs >= 0.9)
        high = np.sum((test_probs >= 0.8) & (test_probs < 0.9))
        moderate = np.sum((test_probs >= 0.6) & (test_probs < 0.8))
        low_mod = np.sum((test_probs >= 0.35) & (test_probs < 0.6))
        low = np.sum(test_probs < 0.35)
        
        print(f"\nPrediction distribution on test set:")
        print(f"Very High Risk (90%+): {very_high} patients")
        print(f"High Risk (80-89%): {high} patients")
        print(f"Moderate Risk (60-79%): {moderate} patients")
        print(f"Low-Moderate Risk (35-59%): {low_mod} patients")
        print(f"Low Risk (<35%): {low} patients")
        
        # Save model and scaler
        self.save_model_and_scaler()
        
        return history
    
    def save_model_and_scaler(self):
        """Save the trained model and scaler"""
        try:
            self.model.save('disease_prediction_model.h5')
            joblib.dump(self.scaler, 'disease_prediction_scaler.pkl')
            joblib.dump(self.feature_names, 'feature_names.pkl')
            print("\nModel, scaler, and feature names saved successfully!")
        except Exception as e:
            print(f"Error saving: {e}")
    
    def test_sample_patients(self):
        """Test the model with sample patients across all risk categories"""
        print("\n" + "="*60)
        print("TESTING SAMPLE PATIENTS")
        print("="*60)
        
        test_patients = {
            "Very High Risk": {
                'age': 70, 'sex': 1, 'cp': 1, 'trestbps': 180, 'chol': 320,
                'fbs': 1, 'restecg': 2, 'thalach': 100, 'exang': 1,
                'oldpeak': 4.5, 'slope': 2, 'ca': 3, 'thal': 7
            },
            "High Risk": {
                'age': 63, 'sex': 1, 'cp': 1, 'trestbps': 165, 'chol': 290,
                'fbs': 1, 'restecg': 1, 'thalach': 125, 'exang': 0,
                'oldpeak': 2.8, 'slope': 2, 'ca': 2, 'thal': 6
            },
            "Moderate Risk": {
                'age': 55, 'sex': 1, 'cp': 2, 'trestbps': 150, 'chol': 260,
                'fbs': 0, 'restecg': 1, 'thalach': 140, 'exang': 0,
                'oldpeak': 2.0, 'slope': 2, 'ca': 1, 'thal': 5
            },
            "Low-Moderate Risk": {
                'age': 45, 'sex': 0, 'cp': 3, 'trestbps': 135, 'chol': 210,
                'fbs': 0, 'restecg': 0, 'thalach': 155, 'exang': 0,
                'oldpeak': 1.0, 'slope': 1, 'ca': 0, 'thal': 4
            },
            "Low Risk": {
                'age': 35, 'sex': 0, 'cp': 4, 'trestbps': 110, 'chol': 180,
                'fbs': 0, 'restecg': 0, 'thalach': 180, 'exang': 0,
                'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 3
            }
        }
        
        print("Sample patient predictions:")
        print("-" * 60)
        
        for category, patient_data in test_patients.items():
            patient_df = pd.DataFrame([patient_data])
            patient_df = patient_df[self.feature_names]
            
            patient_scaled = self.scaler.transform(patient_df)
            prediction = self.model.predict(patient_scaled, verbose=0)[0][0]
            
            print(f"{category:<20}: {prediction*100:>6.1f}%")

def main():
    """Main training function"""
    trainer = BalancedDiseaseRiskTrainer()
    
    # Train the balanced model
    history = trainer.train_balanced_model()
    
    # Test with sample patients
    trainer.test_sample_patients()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("The new model should provide better balanced predictions")
    print("across all 5 risk categories.")
    print("\nYou can now run predict_patient.py to test the new model.")

if __name__ == "__main__":
    main()