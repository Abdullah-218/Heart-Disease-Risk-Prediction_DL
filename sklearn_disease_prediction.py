# Disease Risk Prediction using Scikit-Learn
# No TensorFlow - Clean and Fast Implementation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
np.random.seed(42)

class DiseaseRiskPredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.model_scores = {}
        
    def create_synthetic_dataset(self, n_samples=1000):
        """Create realistic synthetic heart disease dataset"""
        print("🏥 Creating synthetic heart disease dataset...")
        
        np.random.seed(42)
        
        # Generate realistic patient data
        data = {}
        
        # Age: Normal distribution around 54 years
        data['age'] = np.random.normal(54, 12, n_samples).clip(25, 80).astype(int)
        
        # Sex: 65% male (higher heart disease risk)
        data['sex'] = np.random.binomial(1, 0.65, n_samples)
        
        # Chest pain type (0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic)
        data['cp'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.2, 0.3, 0.2])
        
        # Resting blood pressure
        data['trestbps'] = np.random.normal(130, 20, n_samples).clip(90, 200).astype(int)
        
        # Cholesterol
        data['chol'] = np.random.normal(240, 60, n_samples).clip(120, 500).astype(int)
        
        # Fasting blood sugar > 120 mg/dl
        data['fbs'] = np.random.binomial(1, 0.15, n_samples)
        
        # Resting ECG results
        data['restecg'] = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.45, 0.05])
        
        # Maximum heart rate achieved
        data['thalach'] = np.random.normal(150, 25, n_samples).clip(70, 210).astype(int)
        
        # Exercise induced angina
        data['exang'] = np.random.binomial(1, 0.35, n_samples)
        
        # ST depression induced by exercise
        data['oldpeak'] = np.random.gamma(1.5, 0.7, n_samples).clip(0, 6)
        
        # Slope of peak exercise ST segment
        data['slope'] = np.random.choice([0, 1, 2], n_samples, p=[0.2, 0.5, 0.3])
        
        # Number of major vessels colored by fluoroscopy
        data['ca'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.2, 0.15, 0.05])
        
        # Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)
        data['thal'] = np.random.choice([1, 2, 3], n_samples, p=[0.1, 0.5, 0.4])
        
        # Create realistic target based on risk factors
        target = np.zeros(n_samples)
        
        for i in range(n_samples):
            risk_score = 0
            
            # Age risk
            if data['age'][i] > 65: risk_score += 3
            elif data['age'][i] > 55: risk_score += 2
            elif data['age'][i] > 45: risk_score += 1
            
            # Gender risk (males higher)
            if data['sex'][i] == 1: risk_score += 1.5
            
            # Chest pain type
            if data['cp'][i] == 0: risk_score += 2  # Typical angina
            elif data['cp'][i] == 1: risk_score += 1  # Atypical
            
            # Blood pressure
            if data['trestbps'][i] > 160: risk_score += 2
            elif data['trestbps'][i] > 140: risk_score += 1
            
            # Cholesterol
            if data['chol'][i] > 280: risk_score += 2
            elif data['chol'][i] > 240: risk_score += 1
            
            # Exercise factors
            if data['exang'][i] == 1: risk_score += 1.5  # Exercise angina
            if data['oldpeak'][i] > 2: risk_score += 1.5  # ST depression
            
            # Heart rate
            if data['thalach'][i] < 120: risk_score += 1
            
            # Major vessels
            risk_score += data['ca'][i] * 0.7
            
            # Thalassemia
            if data['thal'][i] == 3: risk_score += 1.5  # Reversible defect
            
            # Fasting blood sugar
            if data['fbs'][i] == 1: risk_score += 0.5
            
            # Convert to probability with some randomness
            probability = min(risk_score / 12, 0.95)  # Max 95% risk
            target[i] = 1 if np.random.random() < probability else 0
        
        data['target'] = target.astype(int)
        df = pd.DataFrame(data)
        
        print(f"✅ Dataset created: {n_samples} patients")
        print(f"📊 Disease prevalence: {df['target'].mean():.1%}")
        
        return df
    
    def explore_data(self, df):
        """Comprehensive data exploration"""
        print("\n📊 EXPLORATORY DATA ANALYSIS")
        print("=" * 50)
        
        # Basic info
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {df.shape[1] - 1}")
        print(f"Total patients: {df.shape[0]}")
        
        # Target distribution
        disease_count = df['target'].sum()
        healthy_count = len(df) - disease_count
        print(f"\n🔴 Disease cases: {disease_count} ({disease_count/len(df):.1%})")
        print(f"🟢 Healthy cases: {healthy_count} ({healthy_count/len(df):.1%})")
        
        # Key statistics by disease status
        print(f"\n📈 STATISTICS BY DISEASE STATUS")
        print("-" * 40)
        
        key_features = ['age', 'trestbps', 'chol', 'thalach']
        for feature in key_features:
            disease_avg = df[df['target']==1][feature].mean()
            healthy_avg = df[df['target']==0][feature].mean()
            print(f"{feature.upper():12} | Disease: {disease_avg:6.1f} | Healthy: {healthy_avg:6.1f}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Age distribution
        df[df['target']==0]['age'].hist(alpha=0.7, bins=25, ax=axes[0,0], label='Healthy', color='green')
        df[df['target']==1]['age'].hist(alpha=0.7, bins=25, ax=axes[0,0], label='Disease', color='red')
        axes[0,0].set_title('Age Distribution by Health Status')
        axes[0,0].legend()
        axes[0,0].set_xlabel('Age (years)')
        
        # Cholesterol vs Blood Pressure
        healthy = df[df['target']==0]
        disease = df[df['target']==1]
        axes[0,1].scatter(healthy['chol'], healthy['trestbps'], alpha=0.6, label='Healthy', color='green')
        axes[0,1].scatter(disease['chol'], disease['trestbps'], alpha=0.6, label='Disease', color='red')
        axes[0,1].set_xlabel('Cholesterol')
        axes[0,1].set_ylabel('Blood Pressure')
        axes[0,1].set_title('Cholesterol vs Blood Pressure')
        axes[0,1].legend()
        
        # Feature correlations with target
        correlations = df.corr()['target'].abs().sort_values(ascending=False)[1:]
        correlations.plot(kind='barh', ax=axes[1,0], color='skyblue')
        axes[1,0].set_title('Feature Correlation with Disease')
        axes[1,0].set_xlabel('Absolute Correlation')
        
        # Target distribution
        target_counts = df['target'].value_counts()
        axes[1,1].pie(target_counts.values, labels=['Healthy', 'Disease'], 
                     autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        axes[1,1].set_title('Overall Disease Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def prepare_models(self):
        """Initialize multiple ML models for comparison"""
        print("\n🤖 INITIALIZING MACHINE LEARNING MODELS")
        print("=" * 50)
        
        self.models = {
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=32,
                learning_rate_init=0.001,
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            ),
            'Support Vector Machine': SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }
        
        print(f"✅ Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"   • {name}")
    
    def train_and_compare_models(self, X_train, y_train, X_val, y_val):
        """Train all models and compare performance"""
        print("\n🚀 TRAINING AND COMPARING MODELS")
        print("=" * 50)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            
            # Store results
            results[name] = {
                'model': model,
                'train_accuracy': train_score,
                'val_accuracy': val_score,
                'overfitting': train_score - val_score
            }
            
            print(f"   ✅ Train Accuracy: {train_score:.3f}")
            print(f"   📊 Val Accuracy: {val_score:.3f}")
            print(f"   📈 Overfitting: {results[name]['overfitting']:.3f}")
        
        # Find best model
        best_name = max(results.keys(), key=lambda x: results[x]['val_accuracy'])
        self.best_model = results[best_name]['model']
        self.best_model_name = best_name
        self.model_scores = results
        
        print(f"\n🏆 BEST MODEL: {best_name}")
        print(f"   📊 Validation Accuracy: {results[best_name]['val_accuracy']:.3f}")
        
        return results
    
    def comprehensive_evaluation(self, X_test, y_test):
        """Detailed evaluation of the best model"""
        print(f"\n📊 COMPREHENSIVE EVALUATION - {self.best_model_name}")
        print("=" * 60)
        
        # Predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_prob = self.best_model.predict_proba(X_test)[:, 1]
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"🎯 Test Accuracy: {accuracy:.3f}")
        
        # Detailed report
        report = classification_report(y_test, y_pred, output_dict=True)
        print(f"🔍 Precision: {report['1']['precision']:.3f}")
        print(f"📊 Recall: {report['1']['recall']:.3f}")
        print(f"⚖️  F1-Score: {report['1']['f1-score']:.3f}")
        
        # ROC AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        print(f"📈 ROC AUC: {roc_auc:.3f}")
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
                   xticklabels=['Healthy', 'Disease'],
                   yticklabels=['Healthy', 'Disease'])
        axes[0,0].set_title('Confusion Matrix')
        
        # ROC Curve
        axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, 
                      label=f'ROC (AUC = {roc_auc:.3f})')
        axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        pr_auc = auc(recall, precision)
        axes[1,0].plot(recall, precision, color='green', lw=2,
                      label=f'PR (AUC = {pr_auc:.3f})')
        axes[1,0].set_xlabel('Recall')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].set_title('Precision-Recall Curve')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Model Comparison
        model_names = list(self.model_scores.keys())
        val_accuracies = [self.model_scores[name]['val_accuracy'] for name in model_names]
        
        bars = axes[1,1].bar(range(len(model_names)), val_accuracies, 
                            color=['gold' if name == self.best_model_name else 'skyblue' 
                                  for name in model_names])
        axes[1,1].set_xticks(range(len(model_names)))
        axes[1,1].set_xticklabels([name.replace(' ', '\n') for name in model_names], rotation=0)
        axes[1,1].set_ylabel('Validation Accuracy')
        axes[1,1].set_title('Model Performance Comparison')
        axes[1,1].grid(True, alpha=0.3)
        
        # Highlight best model
        for i, (bar, name) in enumerate(zip(bars, model_names)):
            if name == self.best_model_name:
                bar.set_color('gold')
                axes[1,1].text(i, val_accuracies[i] + 0.01, '🏆', 
                              ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        return accuracy, roc_auc
    
    def feature_importance_analysis(self, X_test, y_test):
        """Analyze feature importance"""
        print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            self.best_model, X_test, y_test, 
            n_repeats=10, random_state=42, scoring='accuracy'
        )
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=False)
        
        print("🔝 Top 10 Most Important Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"   {row['feature']:12} | {row['importance']:.4f} ± {row['std']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_10 = importance_df.head(10)
        bars = plt.barh(range(len(top_10)), top_10['importance'], 
                       color='lightcoral', alpha=0.8)
        plt.yticks(range(len(top_10)), top_10['feature'])
        plt.xlabel('Permutation Importance')
        plt.title(f'Top 10 Feature Importance - {self.best_model_name}')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, top_10['importance'])):
            plt.text(importance + 0.001, i, f'{importance:.3f}', 
                    va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def cross_validation_analysis(self, X, y):
        """Perform cross-validation for all models"""
        print(f"\n🔄 CROSS-VALIDATION ANALYSIS")
        print("=" * 50)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = {}
        
        for name, model in self.models.items():
            print(f"🔄 Cross-validating {name}...")
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            cv_results[name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            print(f"   📊 Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        
        return cv_results
    
    def predict_patient_risk(self, patient_data):
        """Predict disease risk for a patient"""
        if self.best_model is None:
            raise ValueError("No model trained yet!")
        
        # Convert to DataFrame
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data
        
        # Ensure correct feature order
        patient_df = patient_df[self.feature_names]
        
        # Scale features
        patient_scaled = self.scaler.transform(patient_df)
        
        # Get prediction and probability
        prediction = self.best_model.predict(patient_scaled)[0]
        risk_prob = self.best_model.predict_proba(patient_scaled)[0][1]
        
        # Risk categorization
        if risk_prob >= 0.7:
            risk_level = "🔴 HIGH RISK"
            recommendation = "Immediate cardiology consultation recommended"
        elif risk_prob >= 0.4:
            risk_level = "🟡 MODERATE RISK"
            recommendation = "Regular monitoring and lifestyle changes needed"
        else:
            risk_level = "🟢 LOW RISK"
            recommendation = "Continue healthy lifestyle practices"
        
        return {
            'prediction': int(prediction),
            'probability': float(risk_prob),
            'percentage': f"{risk_prob*100:.1f}%",
            'level': risk_level,
            'recommendation': recommendation
        }

def create_sample_patients():
    """Create sample patients for demonstration"""
    patients = {
        'High Risk Patient': {
            'age': 68, 'sex': 1, 'cp': 0, 'trestbps': 170, 'chol': 320,
            'fbs': 1, 'restecg': 1, 'thalach': 100, 'exang': 1,
            'oldpeak': 4.0, 'slope': 2, 'ca': 3, 'thal': 3
        },
        'Moderate Risk Patient': {
            'age': 55, 'sex': 1, 'cp': 1, 'trestbps': 145, 'chol': 250,
            'fbs': 0, 'restecg': 0, 'thalach': 130, 'exang': 1,
            'oldpeak': 2.0, 'slope': 1, 'ca': 1, 'thal': 2
        },
        'Low Risk Patient': {
            'age': 32, 'sex': 0, 'cp': 3, 'trestbps': 110, 'chol': 180,
            'fbs': 0, 'restecg': 0, 'thalach': 185, 'exang': 0,
            'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 2
        }
    }
    return patients

def main():
    """Main execution function"""
    print("🏥 DISEASE RISK PREDICTION SYSTEM")
    print("Using Advanced Machine Learning (No TensorFlow)")
    print("=" * 60)
    
    # Initialize predictor
    predictor = DiseaseRiskPredictor()
    
    # Step 1: Create dataset
    df = predictor.create_synthetic_dataset(n_samples=1200)
    
    # Step 2: Explore data
    predictor.explore_data(df)
    
    # Step 3: Prepare data
    X = df.drop('target', axis=1)
    y = df['target']
    predictor.feature_names = X.columns.tolist()
    
    # Step 4: Split data
    print("\n📂 SPLITTING DATASET")
    print("=" * 50)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"📚 Training: {X_train.shape[0]} samples")
    print(f"📖 Validation: {X_val.shape[0]} samples") 
    print(f"📄 Testing: {X_test.shape[0]} samples")
    
    # Step 5: Scale features
    X_train_scaled = predictor.scaler.fit_transform(X_train)
    X_val_scaled = predictor.scaler.transform(X_val)
    X_test_scaled = predictor.scaler.transform(X_test)
    
    # Step 6: Initialize and train models
    predictor.prepare_models()
    results = predictor.train_and_compare_models(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Step 7: Comprehensive evaluation
    accuracy, auc_score = predictor.comprehensive_evaluation(X_test_scaled, y_test)
    
    # Step 8: Feature importance
    importance_df = predictor.feature_importance_analysis(X_test_scaled, y_test)
    
    # Step 9: Cross-validation
    cv_results = predictor.cross_validation_analysis(X_train_scaled, y_train)
    
    # Step 10: Demo predictions
    print(f"\n🧪 DEMONSTRATION PREDICTIONS")
    print("=" * 50)
    
    sample_patients = create_sample_patients()
    
    for patient_type, patient_data in sample_patients.items():
        result = predictor.predict_patient_risk(patient_data)
        print(f"\n👤 {patient_type}:")
        print(f"   🎯 Risk: {result['percentage']} - {result['level']}")
        print(f"   💡 Advice: {result['recommendation']}")
    
    # Final summary
    print(f"\n🎉 PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"🏆 Best Model: {predictor.best_model_name}")
    print(f"📊 Final Accuracy: {accuracy:.1%}")
    print(f"📈 AUC Score: {auc_score:.3f}")
    cv_mean = cv_results[predictor.best_model_name]['mean']
    cv_std = cv_results[predictor.best_model_name]['std']
    print(f"🔄 CV Accuracy: {cv_mean:.3f} ± {cv_std:.3f}")
    print(f"🔝 Top Feature: {importance_df.iloc[0]['feature']}")
    
    return predictor

def interactive_patient_input():
    """Simple interactive patient data input"""
    print(f"\n🔮 INTERACTIVE RISK ASSESSMENT")
    print("=" * 50)
    print("Enter patient details (press Enter for defaults):")
    
    defaults = {
        'age': 50, 'sex': 1, 'cp': 0, 'trestbps': 120, 'chol': 200,
        'fbs': 0, 'restecg': 0, 'thalach': 150, 'exang': 0,
        'oldpeak': 1.0, 'slope': 1, 'ca': 0, 'thal': 2
    }
    
    questions = {
        'age': '👤 Age (25-80)',
        'sex': '⚥ Sex (1=male, 0=female)',
        'trestbps': '💓 Blood pressure (90-200)',
        'chol': '🩸 Cholesterol (120-400)',
        'thalach': '💗 Max heart rate (70-200)'
    }
    
    patient_data = defaults.copy()
    
    for feature, question in questions.items():
        try:
            user_input = input(f"{question} [{defaults[feature]}]: ").strip()
            if user_input:
                patient_data[feature] = float(user_input)
        except:
            pass  # Use default
    
    return patient_data

if __name__ == "__main__":
    try:
        # Run complete analysis
        predictor = main()
        
        # Interactive prediction
        print("\n" + "🔮" * 25)
        while True:
            choice = input("\nTry prediction for custom patient? (y/n): ").lower().strip()
            if choice == 'y':
                custom_patient = interactive_patient_input()
                result = predictor.predict_patient_risk(custom_patient)
                
                print(f"\n🎯 PREDICTION RESULT:")
                print(f"   Risk Level: {result['level']}")
                print(f"   Risk Probability: {result['percentage']}")
                print(f"   Recommendation: {result['recommendation']}")
            else:
                break
        
        print(f"\n✨ Thank you for using the Disease Risk Prediction System!")
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Program interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please ensure all required libraries are installed:")