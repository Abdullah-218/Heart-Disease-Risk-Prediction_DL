# Disease Risk Prediction - Model Training
# train_model.py
import os
# Set environment variables BEFORE importing TensorFlow
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONHASHSEED"] = "0"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance

# Configure matplotlib to use non-interactive backend
plt.switch_backend('Agg')

import tensorflow as tf
# Configure TensorFlow for macOS
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Disable GPU if available to avoid additional complications
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

from tensorflow import keras
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DiseaseRiskTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.history = None
        
    def load_data(self):
        """Load and prepare the heart disease dataset"""
        print("Loading dataset...")
        
        try:
            # Try to load the UCI heart disease dataset
            print("Loading dataset... from url")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
            
            df = pd.read_csv(url, names=column_names)
            
            # Clean the data immediately after loading
            df = df.replace('?', np.nan)
            
            # Check if we have too many missing values
            missing_percentage = df.isnull().sum() / len(df) * 100
            if missing_percentage.max() > 50:
                print("Too many missing values in real dataset, using synthetic data...")
                return self.create_synthetic_data()
            
            print("Real UCI dataset loaded successfully!")
            return df
            
        except Exception as e:
            print(f"Could not load real dataset ({e}), using synthetic data...")
            return self.create_synthetic_data()
    
    def create_synthetic_data(self, n_samples=300):
        """Create synthetic heart disease data for demonstration"""
        print("Loading dataset... from synthetic data")
        np.random.seed(42)
        
        data = {
            'age': np.random.normal(54, 9, n_samples).clip(29, 77).astype(int),
            'sex': np.random.binomial(1, 0.68, n_samples),  # More males
            'cp': np.random.choice([0, 1, 2, 3], n_samples, p=[0.47, 0.17, 0.29, 0.07]),
            'trestbps': np.random.normal(131, 17, n_samples).clip(94, 200).astype(int),
            'chol': np.random.normal(246, 52, n_samples).clip(126, 564).astype(int),
            'fbs': np.random.binomial(1, 0.15, n_samples),
            'restecg': np.random.choice([0, 1, 2], n_samples, p=[0.48, 0.48, 0.04]),
            'thalach': np.random.normal(149, 23, n_samples).clip(71, 202).astype(int),
            'exang': np.random.binomial(1, 0.33, n_samples),
            'oldpeak': np.random.gamma(1, 1, n_samples).clip(0, 6.2),
            'slope': np.random.choice([0, 1, 2], n_samples, p=[0.21, 0.49, 0.30]),
            'ca': np.random.choice([0, 1, 2, 3], n_samples, p=[0.54, 0.18, 0.12, 0.16]),
            'thal': np.random.choice([1, 2, 3], n_samples, p=[0.02, 0.55, 0.43]),
        }
        
        # Create target based on some logical rules
        target = np.zeros(n_samples)
        for i in range(n_samples):
            risk_score = 0
            risk_score += 1 if data['age'][i] > 55 else 0
            risk_score += 1 if data['sex'][i] == 1 else 0  # Male
            risk_score += 1 if data['cp'][i] in [0, 1] else 0  # Chest pain
            risk_score += 1 if data['chol'][i] > 240 else 0  # High cholesterol
            risk_score += 1 if data['trestbps'][i] > 140 else 0  # High BP
            risk_score += 1 if data['exang'][i] == 1 else 0  # Exercise angina
            
            # Add some noise
            target[i] = 1 if (risk_score >= 3 or np.random.random() < 0.1) else 0
        
        data['target'] = target.astype(int)
        return pd.DataFrame(data)
    
    def preprocess_data(self, df):
        """Clean and preprocess the dataset"""
        print("Preprocessing data...")
        
        # Handle missing values
        df = df.replace('?', np.nan)
        df = df.dropna()
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Binary target (0: no disease, 1: disease)
        df['target'] = (df['target'] > 0).astype(int)
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def explore_data(self, df):
        """Perform exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        print(f"Dataset shape: {df.shape}")
        print(f"\nMissing values before cleaning:\n{df.isnull().sum()}")
        
        # Clean data first for analysis
        df_clean = df.replace('?', np.nan)
        print(f"\nMissing values after replacing '?' with NaN:")
        missing_counts = df_clean.isnull().sum()
        print(missing_counts[missing_counts > 0])
        
        # Drop rows with missing values for analysis
        df_clean = df_clean.dropna()
        print(f"\nDataset shape after removing missing values: {df_clean.shape}")
        
        # Convert to numeric
        for col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Binary target for analysis
        df_clean['target'] = (df_clean['target'] > 0).astype(int)
        
        print(f"\nTarget distribution:\n{df_clean['target'].value_counts()}")
        print(f"\nTarget percentage:\n{df_clean['target'].value_counts(normalize=True) * 100}")
        
        # Basic statistics
        print(f"\nBasic Statistics:\n{df_clean.describe()}")
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Target distribution
        df_clean['target'].value_counts().plot(kind='bar', ax=axes[0,0], title='Disease Distribution')
        axes[0,0].set_xlabel('Disease (0=No, 1=Yes)')
        
        # Age distribution by target
        for target_val in [0, 1]:
            subset = df_clean[df_clean['target'] == target_val]['age']
            axes[0,1].hist(subset, alpha=0.7, bins=15, label=f'Target {target_val}')
        axes[0,1].set_title('Age Distribution by Disease Status')
        axes[0,1].set_xlabel('Age')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend(['No Disease', 'Disease'])
        
        # Correlation heatmap
        correlation = df_clean.corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', ax=axes[1,0], cmap='coolwarm')
        axes[1,0].set_title('Feature Correlation Matrix')
        
        # Feature importance (correlation with target)
        target_corr = correlation['target'].abs().sort_values(ascending=False)[1:]
        target_corr.plot(kind='barh', ax=axes[1,1], title='Feature Correlation with Target')
        
        plt.tight_layout()
        # Save plot instead of showing to avoid display issues
        plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print("Data exploration plots saved as 'data_exploration.png'")
    
    def create_model(self, input_dim):
        """Create the deep learning model architecture"""
        model = keras.Sequential([
            # Input layer with batch normalization
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dropout(0.2),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train the deep learning model"""
        print("\n" + "="*50)
        print("TRAINING MODEL")
        print("="*50)
        
        # Create model
        self.model = self.create_model(X_train.shape[1])
        
        # Display model architecture
        print("\nModel Architecture:")
        self.model.summary()
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=10,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Train model
        print("\nStarting training...")
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=200,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        
    def plot_training_history(self):
        """Plot training metrics"""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy', color='blue')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[0, 0].set_title('Model Accuracy Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss', color='blue')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss', color='red')
        axes[0, 1].set_title('Model Loss Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision', color='blue')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision', color='red')
        axes[1, 0].set_title('Model Precision Over Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall', color='blue')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall', color='red')
        axes[1, 1].set_title('Model Recall Over Time')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        # Save plot instead of showing
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📈 Training history plots saved as 'training_history.png'")
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Confusion matrix saved as 'confusion_matrix.png'")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📈 ROC curve saved as 'roc_curve.png'")
        
        return accuracy, roc_auc
    
    def cross_validate(self, X, y, cv_folds=5):
        """Perform cross-validation"""
        print(f"\nPerforming {cv_folds}-fold cross-validation...")
        
        kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            print(f"Training fold {fold + 1}/{cv_folds}")
            
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Scale data
            scaler_fold = StandardScaler()
            X_train_scaled = scaler_fold.fit_transform(X_train_fold)
            X_val_scaled = scaler_fold.transform(X_val_fold)
            
            # Create and train model
            model_fold = self.create_model(X_train_scaled.shape[1])
            
            model_fold.fit(
                X_train_scaled, y_train_fold,
                epochs=100,
                batch_size=32,
                validation_data=(X_val_scaled, y_val_fold),
                verbose=0,
                callbacks=[EarlyStopping(patience=15, restore_best_weights=True)]
            )
            
            # Evaluate
            score = model_fold.evaluate(X_val_scaled, y_val_fold, verbose=0)
            cv_scores.append(score[1])  # accuracy
        
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        
        print(f"\nCross-validation Results:")
        print(f"Mean Accuracy: {mean_cv_score:.4f} (+/- {std_cv_score * 2:.4f})")
        
        return cv_scores

    def feature_importance_analysis(self, X_test_scaled, y_test):
        """Calculate and plot feature importance using model weights"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)

        # Get weights from the first hidden layer
        try:
            weights = self.model.layers[0].get_weights()[0]  # shape: (n_features, n_units)
            importance = np.mean(np.abs(weights), axis=1)    # average absolute weight per feature

            # Create DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)

            print("Top 10 Most Important Features (by input layer weights):")
            print(feature_importance_df.head(10))

            # Plot top 10 features
            plt.figure(figsize=(10, 8))
            top_features = feature_importance_df.head(10)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Average Absolute Weight (Input Layer)')
            plt.title('Top 10 Feature Importance for Disease Prediction')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("📊 Feature importance plot saved as 'feature_importance.png'")

            return feature_importance_df
        except Exception as e:
            print(f"⚠️ Could not compute feature importance: {e}")
            return pd.DataFrame({"feature": self.feature_names, "importance": [0]*len(self.feature_names)})
        

    def save_model_and_scaler(self, model_path='disease_prediction_model.h5', 
                              scaler_path='disease_prediction_scaler.pkl',
                              features_path='feature_names.pkl'):
        """Save trained model, scaler, and feature names"""
        try:
            # Save the model
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
            
            # Save the scaler
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")
            
            # Save feature names
            joblib.dump(self.feature_names, features_path)
            print(f"Feature names saved to {features_path}")
            
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

def main():
    """Main training function"""
    print("="*60)
    print("DISEASE RISK PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Initialize trainer
    trainer = DiseaseRiskTrainer()
    
    # Step 1: Load and explore data
    df = trainer.load_data()
    print(f"Dataset loaded successfully! Shape: {df.shape}")
    
    # Explore data
    trainer.explore_data(df)
    
    # Step 2: Preprocess data
    X, y = trainer.preprocess_data(df)
    
    # Step 3: Split data
    print("\nSplitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Step 4: Scale features
    X_train_scaled = trainer.scaler.fit_transform(X_train)
    X_val_scaled = trainer.scaler.transform(X_val)
    X_test_scaled = trainer.scaler.transform(X_test)
    
    # Convert back to DataFrames for compatibility
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Step 5: Train model
    trainer.train_model(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Step 6: Plot training history
    trainer.plot_training_history()
    
    # Step 7: Evaluate model
    accuracy, auc_score = trainer.evaluate_model(X_test_scaled, y_test)
    
    # Step 8: Cross-validation
    cv_scores = trainer.cross_validate(X, y)
    
    # Step 9: Feature importance analysis
    feature_importance_df = trainer.feature_importance_analysis(X_test_scaled, y_test)
    
    # Step 10: Model performance summary
    print("\n" + "="*50)
    print("FINAL MODEL PERFORMANCE SUMMARY")
    print("="*50)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"Cross-validation Mean Accuracy: {np.mean(cv_scores):.4f}")
    print(f"Cross-validation Std: {np.std(cv_scores):.4f}")
    print(f"Training Epochs: {len(trainer.history.history['loss'])}")
    print(f"Best Validation Loss: {min(trainer.history.history['val_loss']):.4f}")
    
    # Step 11: Save model and scaler
    trainer.save_model_and_scaler()
    
    print("\nTraining completed successfully!")
    print("Model and scaler have been saved for use in predict_patient.py")
    
    return trainer

if __name__ == "__main__":
    trainer = main()