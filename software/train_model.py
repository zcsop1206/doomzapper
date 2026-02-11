"""
sEMG Feature Extraction and Random Forest Training Pipeline
Processes raw sEMG data and trains a classifier for scroll detection
"""

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import glob

class sEMGFeatureExtractor:
    """Extract time-domain features from sEMG signals"""
    
    @staticmethod
    def extract_rms(window):
        """Root Mean Square - overall signal power"""
        return np.sqrt(np.mean(window ** 2))
    
    @staticmethod
    def extract_mav(window):
        """Mean Absolute Value - average muscle activity"""
        return np.mean(np.abs(window))
    
    @staticmethod
    def extract_wl(window):
        """Waveform Length - signal complexity/variation"""
        return np.sum(np.abs(np.diff(window)))
    
    @staticmethod
    def extract_zc(window, threshold=10):
        """
        Zero Crossings - frequency content indicator
        Note: For envelope data, this counts near-zero crossings
        threshold: minimum difference between consecutive samples
        """
        # Shift to mean-centered for zero crossing detection
        window_centered = window - np.mean(window)
        
        zc_count = 0
        for i in range(len(window_centered) - 1):
            if abs(window_centered[i] - window_centered[i+1]) >= threshold:
                if window_centered[i] * window_centered[i+1] < 0:
                    zc_count += 1
        return zc_count
    
    @staticmethod
    def extract_ssc(window, threshold=10):
        """
        Slope Sign Changes - frequency/texture information
        Counts changes in signal slope direction
        """
        ssc_count = 0
        for i in range(1, len(window) - 1):
            diff_prev = window[i] - window[i-1]
            diff_next = window[i] - window[i+1]
            
            if abs(diff_prev) >= threshold or abs(diff_next) >= threshold:
                if diff_prev * diff_next > 0:  # Same sign = slope change
                    ssc_count += 1
        return ssc_count
    
    @staticmethod
    def extract_all_features(window):
        """Extract all features from a single window"""
        return {
            'RMS': sEMGFeatureExtractor.extract_rms(window),
            'MAV': sEMGFeatureExtractor.extract_mav(window),
            'WL': sEMGFeatureExtractor.extract_wl(window),
            'ZC': sEMGFeatureExtractor.extract_zc(window),
            'SSC': sEMGFeatureExtractor.extract_ssc(window)
        }

def load_and_process_data(filepath):
    """
    Load CSV data from Arduino and extract features
    
    Returns:
        features: numpy array of shape (n_samples, n_features)
        labels: numpy array of shape (n_samples,)
    """
    print(f"Loading data from {filepath}")
    
    # Read CSV, skipping comment lines that start with #
    df = pd.read_csv(filepath, header=None, comment='#')
    
    # Last column is label, everything else except first column (timestamp) is signal
    timestamps = df.iloc[:, 0]
    labels = df.iloc[:, -1]
    signal_data = df.iloc[:, 1:-1]  # All columns except timestamp and label
    
    print(f"  Loaded {len(df)} windows")
    print(f"  Window size: {signal_data.shape[1]} samples")
    print(f"  Label distribution: {labels.value_counts().to_dict()}")
    
    # Extract features for each window
    feature_list = []
    for idx, row in signal_data.iterrows():
        window = row.values.astype(float)
        features = sEMGFeatureExtractor.extract_all_features(window)
        feature_list.append(list(features.values()))
    
    # Convert to numpy arrays
    X = np.array(feature_list)
    y = labels.values
    
    # Encode labels (scroll=1, rest=0)
    y_encoded = np.array([1 if label == 'scroll' else 0 for label in y])
    
    return X, y_encoded

def save_processed_features(X, y, filepath):
    """Save processed features to CSV for later analysis"""
    feature_names = ['RMS', 'MAV', 'WL', 'ZC', 'SSC']
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    df.to_csv(filepath, index=False)
    print(f"  Saved processed features to {filepath}")

def train_random_forest(X_train, y_train, tune_parameters=True):
    """
    Train Random Forest classifier with optional hyperparameter tuning
    
    Returns:
        trained model
        best parameters (if tuned)
    """
    if tune_parameters:
        print("\nPerforming hyperparameter tuning with GridSearchCV...")
        
        # Parameter grid for tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Create base model
        rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    else:
        print("\nTraining Random Forest with default parameters...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        rf.fit(X_train, y_train)
        return rf, None

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test set"""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\n[[TN  FP]")
    print(" [FN  TP]]")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Rest', 'Scroll']))
    
    # Feature Importance
    feature_names = ['RMS', 'MAV', 'WL', 'ZC', 'SSC']
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature Importance:")
    for i in range(len(feature_names)):
        print(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    return accuracy, y_pred

def main():
    print("="*60)
    print("sEMG SCROLL DETECTION - FEATURE EXTRACTION & ML TRAINING")
    print("="*60)
    
    # Find data files
    data_dir = 'data'
    training_files = glob.glob(os.path.join(data_dir, 'training_*.csv'))
    testing_files = glob.glob(os.path.join(data_dir, 'testing_*.csv'))
    
    if not training_files:
        print(f"\nError: No training data found in {data_dir}/")
        print("Please run collect_data.py first to collect training data")
        return
    
    # Use most recent training file
    training_file = max(training_files, key=os.path.getctime)
    
    # Load and process training data
    print(f"\n{'='*60}")
    print("LOADING TRAINING DATA")
    print(f"{'='*60}")
    X_train, y_train = load_and_process_data(training_file)
    
    # Save processed training features
    processed_train_file = training_file.replace('.csv', '_features.csv')
    save_processed_features(X_train, y_train, processed_train_file)
    
    # Load and process testing data (if available)
    if testing_files:
        testing_file = max(testing_files, key=os.path.getctime)
        print(f"\n{'='*60}")
        print("LOADING TESTING DATA")
        print(f"{'='*60}")
        X_test, y_test = load_and_process_data(testing_file)
        
        # Save processed testing features
        processed_test_file = testing_file.replace('.csv', '_features.csv')
        save_processed_features(X_test, y_test, processed_test_file)
    else:
        print("\nWarning: No testing data found. Using train/test split from training data.")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
        )
    
    # Train model
    print(f"\n{'='*60}")
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print(f"{'='*60}")
    
    model, best_params = train_random_forest(X_train, y_train, tune_parameters=True)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\nCross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Evaluate on test set
    accuracy, y_pred = evaluate_model(model, X_test, y_test)
    
    # Save model
    model_filename = 'models/scroll_detector_rf.pkl'
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_filename)
    print(f"\n✓ Model saved to {model_filename}")
    
    # Save best parameters
    if best_params:
        params_filename = 'models/best_params.txt'
        with open(params_filename, 'w') as f:
            f.write(f"Best Parameters:\n")
            for param, value in best_params.items():
                f.write(f"  {param}: {value}\n")
            f.write(f"\nTest Accuracy: {accuracy:.4f}\n")
        print(f"✓ Best parameters saved to {params_filename}")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    main()
