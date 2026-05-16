# -*- coding: utf-8 -*-
"""
Advanced Ensemble Model for Football Predictions
Combines multiple algorithms for higher accuracy (Target: 60%+)

@author: Enhanced ML Pipeline
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')


class AdvancedEnsemblePredictor:
    """
    Ensemble model combining:
    - Random Forest (base model)
    - Gradient Boosting (XGBoost-like performance)
    - K-Nearest Neighbors (local patterns)
    
    Improvements over single model:
    - Better handling of class imbalance
    - Reduced overfitting
    - More reliable probability estimates
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = RobustScaler()  # Better for outliers than StandardScaler
        self.ensemble_model = None
        self.feature_names = None
        self.classes = None
        
    def build_ensemble(self, X_train, y_train, cv_folds=5):
        """
        Build ensemble model with optimized hyperparameters
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels (0=Away Win, 1=Draw, 2=Home Win)
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict : Model performance metrics
        """
        
        print("=" * 60)
        print("BUILDING ADVANCED ENSEMBLE MODEL")
        print("=" * 60)
        
        # Scale features
        print("\n[1/5] Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Define individual models with optimized hyperparameters
        print("[2/5] Creating base learners...")
        
        # Random Forest - optimized for balance and generalization
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',  # Handle class imbalance
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Gradient Boosting - captures complex patterns
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=self.random_state
        )
        
        # K-Nearest Neighbors - local patterns
        knn_model = KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            metric='minkowski',
            p=2,
            n_jobs=-1
        )
        
        # Create voting ensemble with soft voting (probability averaging)
        print("[3/5] Creating voting ensemble...")
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('gb', gb_model),
                ('knn', knn_model)
            ],
            voting='soft',  # Use probability averaging
            n_jobs=-1
        )
        
        # Train ensemble
        print("[4/5] Training ensemble model...")
        self.ensemble_model.fit(X_train_scaled, y_train)
        
        # Cross-validation evaluation
        print("[5/5] Evaluating with cross-validation...")
        cv_scores = cross_val_score(
            self.ensemble_model, 
            X_train_scaled, 
            y_train, 
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
            scoring='accuracy'
        )
        
        # Store classes
        self.classes = self.ensemble_model.classes_
        
        # Print results
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION RESULTS")
        print("=" * 60)
        print(f"CV Scores: {cv_scores}")
        print(f"Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return {
            'cv_scores': cv_scores,
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std()
        }
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set with detailed metrics
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
            
        Returns:
        --------
        dict : Detailed evaluation metrics
        """
        
        if self.ensemble_model is None:
            raise ValueError("Model not trained yet")
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.ensemble_model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print("\n" + "=" * 60)
        print("TEST SET EVALUATION")
        print("=" * 60)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"\nConfusion Matrix:\n{conf_matrix}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }
    
    def predict_probabilities(self, X):
        """
        Get prediction probabilities for matches
        
        Parameters:
        -----------
        X : array-like
            Features for prediction
            
        Returns:
        --------
        array : Probabilities for each class
        """
        
        if self.ensemble_model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        probabilities = self.ensemble_model.predict_proba(X_scaled)
        
        return probabilities
    
    def predict(self, X):
        """
        Get class predictions
        
        Parameters:
        -----------
        X : array-like
            Features for prediction
            
        Returns:
        --------
        array : Predicted classes
        """
        
        if self.ensemble_model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        return self.ensemble_model.predict(X_scaled)
    
    def save_model(self, filepath):
        """Save trained model and scaler to file"""
        model_dict = {
            'ensemble': self.ensemble_model,
            'scaler': self.scaler,
            'classes': self.classes
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_dict, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model and scaler from file"""
        with open(filepath, 'rb') as f:
            model_dict = pickle.load(f)
        self.ensemble_model = model_dict['ensemble']
        self.scaler = model_dict['scaler']
        self.classes = model_dict['classes']
        print(f"Model loaded from {filepath}")


class FeatureEngineeringEnhanced:
    """
    Enhanced feature engineering with:
    - Interaction features
    - Polynomial features for non-linear relationships
    - Normalization for better scaling
    - Outlier handling
    """
    
    @staticmethod
    def create_interaction_features(df):
        """
        Create interaction features to capture team dynamics
        
        Parameters:
        -----------
        df : DataFrame
            Features dataframe
            
        Returns:
        --------
        DataFrame : Enhanced features with interactions
        """
        
        df = df.copy()
        
        # Interaction: Shot creation ability (Shots * Accuracy)
        df['Team_Shot_Power'] = df['Av Shots Diff'] * (df['Av Pass Accuracy Diff'] / 100)
        df['Opponent_Shot_Power'] = -df['Team_Shot_Power']
        
        # Interaction: Defensive capability (Fouls + Corners allowed)
        df['Team_Defensive_Activity'] = df['Av Fouls Diff'] + df['Av Corners Diff']
        
        # Interaction: Possession efficiency
        df['Team_Possession_Efficiency'] = df['Av Possession Diff'] * (df['Av Shots Diff'] / 10)
        
        # Polynomial features for non-linear relationships
        df['Possession_Squared'] = df['Av Possession Diff'] ** 2
        df['Goal_Diff_Squared'] = df['Av Goal Difference'] ** 2
        
        # Ratio features
        df['Shot_Accuracy_Ratio'] = df['Av Shots Inside Box Diff'] / (df['Av Shots Diff'] + 0.1)
        
        return df
    
    @staticmethod
    def handle_outliers(df, columns=None, method='iqr', threshold=3):
        """
        Handle outliers using IQR or Z-score method
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        columns : list
            Columns to check for outliers
        method : str
            'iqr' or 'zscore'
        threshold : float
            Z-score threshold (default: 3)
            
        Returns:
        --------
        DataFrame : Cleaned dataframe
        """
        
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        if method == 'iqr':
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        elif method == 'zscore':
            from scipy import stats
            for col in columns:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                df.loc[z_scores > threshold, col] = df[col].median()
        
        return df


# Example usage function
def example_usage():
    """
    Example of how to use the AdvancedEnsemblePredictor
    """
    
    print("Advanced Ensemble Predictor Example")
    print("====================================\n")
    
    # Create synthetic data for demonstration
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(
        n_samples=500, 
        n_features=14, 
        n_informative=10,
        n_classes=3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    predictor = AdvancedEnsemblePredictor()
    predictor.build_ensemble(X_train, y_train, cv_folds=5)
    
    # Evaluate
    predictor.evaluate(X_test, y_test)
    
    # Make predictions
    probs = predictor.predict_probabilities(X_test[:5])
    print(f"\nSample Predictions:\n{probs}")
    
    return predictor


if __name__ == "__main__":
    example_usage()
