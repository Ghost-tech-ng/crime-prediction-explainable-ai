# main.py - Fixed Advanced Crime Prediction System Core
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import traceback
warnings.filterwarnings('ignore')

# Advanced ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import DBSCAN

# Explainable AI Libraries
import shap
import lime
from lime import lime_tabular

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

import joblib
import os
from datetime import datetime

class AdvancedCrimePredictionSystem:
    """
    Fixed Advanced Crime Prediction System with Explainable AI
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.feature_names = []
        self.best_model = None
        self.best_model_name = None
        self.shap_explainer = None
        # REMOVED: self.lime_explainer = None  # This causes pickling issues
        self.train_data = None
        self.categorical_encoders = {}  # Store encoders for categorical variables
        self.feature_stats = {}  # Store feature statistics for normalization
        self.X_sample_for_lime = None  # Store sample data for LIME recreation
        
    def load_and_preprocess_data(self, train_path, test_path=None):
        """Load and preprocess data with advanced feature engineering"""
        print("🔄 Loading and preprocessing data...")
        
        try:
            # Load data
            print(f"Loading training data from: {train_path}")
            self.train_data = pd.read_csv(train_path)
            
            if test_path and os.path.exists(test_path):
                print(f"Loading test data from: {test_path}")
                self.test_data = pd.read_csv(test_path)
                print(f"Test data shape: {self.test_data.shape}")
            
            print(f"Training data shape: {self.train_data.shape}")
            print(f"Columns: {list(self.train_data.columns)}")
            
            # Check for required columns
            required_cols = ['X', 'Y', 'Category']
            missing_cols = [col for col in required_cols if col not in self.train_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Clean data
            print("Cleaning data...")
            self.train_data = self._clean_data(self.train_data)
            print(f"Data shape after cleaning: {self.train_data.shape}")
            
            # Engineer features
            print("Engineering features...")
            self.train_data = self._engineer_features(self.train_data)
            print(f"Data shape after feature engineering: {self.train_data.shape}")
            
            print(f"✅ Data preprocessing complete. Final shape: {self.train_data.shape}")
            return self.train_data
            
        except Exception as e:
            print(f"❌ Error in load_and_preprocess_data: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _clean_data(self, df):
        """Clean and prepare the dataset"""
        try:
            df = df.copy()
            
            print(f"Original data shape: {df.shape}")
            
            # Handle missing values
            print("Checking for missing values in X, Y...")
            missing_before = df[['X', 'Y']].isnull().sum().sum()
            print(f"Missing X,Y values before cleaning: {missing_before}")
            
            df = df.dropna(subset=['X', 'Y'])
            print(f"Shape after dropping missing X,Y: {df.shape}")
            
            # Clean coordinates
            print("Converting coordinates to numeric...")
            df['X'] = pd.to_numeric(df['X'], errors='coerce')
            df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
            
            # Check for invalid coordinates after conversion
            invalid_coords = df[['X', 'Y']].isnull().sum().sum()
            if invalid_coords > 0:
                print(f"Found {invalid_coords} invalid coordinates after conversion")
                df = df.dropna(subset=['X', 'Y'])
                print(f"Shape after dropping invalid coordinates: {df.shape}")
            
            # Remove invalid coordinates (San Francisco bounds)
            print("Filtering coordinates to San Francisco bounds...")
            sf_bounds = {
                'min_lat': 37.7, 'max_lat': 37.8,
                'min_lon': -122.5, 'max_lon': -122.3
            }
            
            # Only filter if coordinates seem to be for San Francisco
            lat_mean = df['Y'].mean()
            lon_mean = df['X'].mean()
            
            if 37.5 < lat_mean < 38.0 and -123.0 < lon_mean < -122.0:
                before_filter = len(df)
                df = df[
                    (df['Y'] >= sf_bounds['min_lat']) & (df['Y'] <= sf_bounds['max_lat']) &
                    (df['X'] >= sf_bounds['min_lon']) & (df['X'] <= sf_bounds['max_lon'])
                ]
                after_filter = len(df)
                print(f"Filtered out {before_filter - after_filter} records outside SF bounds")
            else:
                print("Coordinates don't appear to be San Francisco, skipping geo-filtering")
            
            print(f"Final cleaned data shape: {df.shape}")
            
            if len(df) == 0:
                raise ValueError("No data remains after cleaning - check coordinate bounds")
            
            return df
            
        except Exception as e:
            print(f"❌ Error in _clean_data: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _engineer_features(self, df):
        """Advanced feature engineering"""
        try:
            df = df.copy()
            print("Starting feature engineering...")
            
            # Temporal features
            if 'Dates' in df.columns:
                print("Engineering temporal features...")
                df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce')
                
                # Check for invalid dates
                invalid_dates = df['Dates'].isnull().sum()
                if invalid_dates > 0:
                    print(f"Warning: {invalid_dates} invalid dates found")
                    # Fill invalid dates with a default or drop them
                    df = df.dropna(subset=['Dates'])
                
                df['Hour'] = df['Dates'].dt.hour
                df['DayOfWeek_num'] = df['Dates'].dt.dayofweek
                df['Month'] = df['Dates'].dt.month
                df['Year'] = df['Dates'].dt.year
                df['IsWeekend'] = (df['DayOfWeek_num'] >= 5).astype(int)
                df['Season'] = df['Month'].apply(self._get_season)
                df['TimeOfDay'] = df['Hour'].apply(self._categorize_time)
                df['IsNight'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
                print("✅ Temporal features created")
            else:
                # Create default temporal features if Dates column doesn't exist
                print("No Dates column found, creating default temporal features...")
                df['Hour'] = 12
                df['DayOfWeek_num'] = 1
                df['Month'] = 6
                df['Year'] = 2023
                df['IsWeekend'] = 0
                df['Season'] = 'Summer'
                df['TimeOfDay'] = 'Afternoon'
                df['IsNight'] = 0
            
            # Spatial features
            if 'X' in df.columns and 'Y' in df.columns:
                print("Engineering spatial features...")
                # Distance from city center (use data center if not SF)
                center_x = df['X'].mean()
                center_y = df['Y'].mean()
                
                df['DistanceFromCenter'] = np.sqrt(
                    (df['X'] - center_x)**2 + (df['Y'] - center_y)**2
                )
                
                # Spatial clustering (reduce data for clustering if too large)
                print("Performing spatial clustering...")
                coords = df[['X', 'Y']].values
                
                # If dataset is very large, sample for clustering
                if len(coords) > 10000:
                    print(f"Large dataset ({len(coords)} points), sampling for clustering...")
                    sample_indices = np.random.choice(len(coords), 10000, replace=False)
                    coords_sample = coords[sample_indices]
                    
                    # Determine appropriate eps based on coordinate range
                    coord_range = max(df['X'].max() - df['X'].min(), df['Y'].max() - df['Y'].min())
                    eps = coord_range * 0.01  # 1% of coordinate range
                    
                    spatial_clusters = DBSCAN(eps=eps, min_samples=5).fit(coords_sample)
                    
                    # Initialize all points as -1 (noise)
                    df['SpatialCluster'] = -1
                    # Assign cluster labels to sampled points
                    df.iloc[sample_indices, df.columns.get_loc('SpatialCluster')] = spatial_clusters.labels_
                else:
                    coord_range = max(df['X'].max() - df['X'].min(), df['Y'].max() - df['Y'].min())
                    eps = coord_range * 0.01
                    spatial_clusters = DBSCAN(eps=eps, min_samples=5).fit(coords)
                    df['SpatialCluster'] = spatial_clusters.labels_
                
                # Grid features
                df['GridX'] = (df['X'] * 100).round().astype(int)
                df['GridY'] = (df['Y'] * 100).round().astype(int)
                print("✅ Spatial features created")
            
            # District features
            if 'PdDistrict' in df.columns:
                print("Engineering district features...")
                district_counts = df['PdDistrict'].value_counts()
                df['DistrictCrimeFreq'] = df['PdDistrict'].map(district_counts)
                print("✅ District features created")
            else:
                print("No PdDistrict column found, creating default district features...")
                df['DistrictCrimeFreq'] = 100  # Default value
            
            # Crime severity (only for training data with Category column)
            if 'Category' in df.columns:
                print("Engineering crime severity features...")
                severity_mapping = {
                    'WARRANTS': 1, 'OTHER OFFENSES': 1, 'NON-CRIMINAL': 1,
                    'LARCENY/THEFT': 2, 'VEHICLE THEFT': 2, 'VANDALISM': 2,
                    'DRUG/NARCOTIC': 3, 'BURGLARY': 3, 'FRAUD': 3,
                    'ASSAULT': 4, 'ROBBERY': 4, 'SEX OFFENSES FORCIBLE': 5,
                    'KIDNAPPING': 5, 'ARSON': 5
                }
                df['CrimeSeverity'] = df['Category'].map(severity_mapping).fillna(2)
                print("✅ Crime severity features created")
            
            print(f"Feature engineering complete. Final shape: {df.shape}")
            return df
            
        except Exception as e:
            print(f"❌ Error in _engineer_features: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _get_season(self, month):
        """Map month to season"""
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        else: return 'Fall'
    
    def _categorize_time(self, hour):
        """Categorize hour into time periods"""
        if 6 <= hour < 12: return 'Morning'
        elif 12 <= hour < 18: return 'Afternoon'
        elif 18 <= hour < 22: return 'Evening'
        else: return 'Night'
    
    def prepare_features_and_target(self, target_column='Category'):
        """Prepare features and target for modeling"""
        try:
            print("Preparing features and target...")
            df = self.train_data.copy()
            
            # Basic numeric features
            numeric_features = [
                'Hour', 'DayOfWeek_num', 'Month', 'Year', 'IsWeekend', 'IsNight',
                'X', 'Y', 'DistanceFromCenter', 'SpatialCluster', 'GridX', 'GridY',
                'DistrictCrimeFreq'
            ]
            
            # Add CrimeSeverity only if it exists (training data)
            if 'CrimeSeverity' in df.columns:
                numeric_features.append('CrimeSeverity')
            
            # Check which numeric features actually exist
            existing_numeric = [col for col in numeric_features if col in df.columns]
            missing_numeric = [col for col in numeric_features if col not in df.columns]
            
            if missing_numeric:
                print(f"Warning: Missing numeric features: {missing_numeric}")
            
            print(f"Using {len(existing_numeric)} numeric features")
            
            # Categorical features to encode
            categorical_features = []
            if 'PdDistrict' in df.columns:
                categorical_features.append('PdDistrict')
            if 'DayOfWeek' in df.columns:
                categorical_features.append('DayOfWeek')
            if 'TimeOfDay' in df.columns:
                categorical_features.append('TimeOfDay')
            if 'Season' in df.columns:
                categorical_features.append('Season')
            
            print(f"Found {len(categorical_features)} categorical features: {categorical_features}")
            
            # Start with numeric features
            feature_columns = existing_numeric.copy()
            X = df[existing_numeric].copy()
            
            # One-hot encode categorical features
            for cat_feature in categorical_features:
                print(f"One-hot encoding {cat_feature}...")
                encoded = pd.get_dummies(df[cat_feature], prefix=cat_feature, dummy_na=True)
                
                # Store the encoder mapping for prediction
                self.categorical_encoders[cat_feature] = {
                    'categories': df[cat_feature].unique().tolist(),
                    'encoded_columns': encoded.columns.tolist()
                }
                
                X = pd.concat([X, encoded], axis=1)
                feature_columns.extend(encoded.columns)
                print(f"Added {len(encoded.columns)} encoded features for {cat_feature}")
            
            # Fill any missing values
            X = X.fillna(0)
            
            # Store feature statistics for prediction normalization
            self.feature_stats = {
                'numeric_means': X[existing_numeric].mean().to_dict(),
                'numeric_stds': X[existing_numeric].std().to_dict()
            }
            
            # Target variable
            y = df[target_column]
            
            print(f"Feature matrix shape: {X.shape}")
            print(f"Target vector shape: {y.shape}")
            print(f"Unique target classes: {len(y.unique())}")
            
            # Encode target
            print("Encoding target variable...")
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Store feature names
            self.feature_names = feature_columns
            
            print("✅ Features and target prepared successfully")
            return X, y_encoded, y
            
        except Exception as e:
            print(f"❌ Error in prepare_features_and_target: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise
    
    def train_models(self, X, y):
        """Train multiple ML models"""
        try:
            print("🤖 Training multiple models...")
            
            # Check if we have enough data diversity
            unique_classes = len(np.unique(y))
            print(f"Number of unique classes: {unique_classes}")
            
            if unique_classes < 2:
                raise ValueError("Need at least 2 different classes for classification")
            
            # Split data
            print("Splitting data into train/validation...")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
            
            # Subsample training data if too large
            max_train_samples = 50000
            if X_train.shape[0] > max_train_samples:
                print(f"Dataset too large ({X_train.shape[0]} rows), subsampling to {max_train_samples}...")
                from sklearn.utils import resample
                X_train, y_train = resample(X_train, y_train, n_samples=max_train_samples, 
                                          random_state=42, stratify=y_train)
                print(f"Subsampled training set: {X_train.shape}")
            
            # Scale features
            print("Scaling features...")
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Models configuration
            models_config = {
                'RandomForest': {
                    'model': RandomForestClassifier(
                        n_estimators=100,
                        max_depth=15,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1
                    ),
                    'use_scaling': False
                },
                'GradientBoosting': {
                    'model': GradientBoostingClassifier(
                        n_estimators=50,
                        learning_rate=0.1,
                        max_depth=6,
                        subsample=0.8,
                        random_state=42
                    ),
                    'use_scaling': False
                },
                'LogisticRegression': {
                    'model': LogisticRegression(
                        random_state=42,
                        max_iter=2000,
                        n_jobs=-1,
                        multi_class='ovr'
                    ),
                    'use_scaling': True
                }
            }
            
            results = {}
            
            for name, config in models_config.items():
                print(f"Training {name}...")
                
                model = config['model']
                X_train_model = X_train_scaled if config['use_scaling'] else X_train
                X_val_model = X_val_scaled if config['use_scaling'] else X_val
                
                # Train
                start_time = datetime.now()
                model.fit(X_train_model, y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Evaluate
                y_pred = model.predict(X_val_model)
                accuracy = accuracy_score(y_val, y_pred)
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'use_scaling': config['use_scaling'],
                    'training_time': training_time
                }
                
                self.models[name] = model
                print(f"{name} Accuracy: {accuracy:.4f}, Training time: {training_time:.2f} seconds")
            
            # Select best model
            self.best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
            self.best_model = results[self.best_model_name]['model']
            
            print(f"\n🏆 Best Model: {self.best_model_name} (Accuracy: {results[self.best_model_name]['accuracy']:.4f})")
            
            return results, X_val, y_val
            
        except Exception as e:
            print(f"❌ Error in train_models: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise
        
    def setup_explainable_ai(self, X):
        """Setup explainable AI tools"""
        try:
            print("🔍 Setting up Explainable AI...")
            
            # SHAP explainer
            if self.best_model_name in ['RandomForest', 'GradientBoosting']:
                self.shap_explainer = shap.TreeExplainer(self.best_model)
            else:
                X_sample = X.sample(min(50, len(X)))
                self.shap_explainer = shap.KernelExplainer(self.best_model.predict_proba, X_sample)
            
            # Store sample data for LIME recreation (instead of storing LIME explainer)
            self.X_sample_for_lime = X.sample(min(100, len(X))).values
            
            print("✅ Explainable AI setup complete!")
            
        except Exception as e:
            print(f"⚠️ Explainable AI setup warning: {str(e)}")
    
    def get_lime_explainer(self):
        """Create LIME explainer on demand (not stored to avoid pickling issues)"""
        if self.X_sample_for_lime is not None:
            return lime_tabular.LimeTabularExplainer(
                self.X_sample_for_lime,
                feature_names=self.feature_names,
                class_names=self.label_encoder.classes_,
                mode='classification'
            )
        return None
    
    def generate_lime_explanation(self, input_data):
        """Generate LIME explanation for a given input"""
        try:
            # Prepare the input data as a DataFrame
            df = pd.DataFrame([input_data])
            df['Hour'] = input_data.get('hour', 12)
            df['DayOfWeek_num'] = input_data.get('day_of_week', 1)
            df['Month'] = input_data.get('month', 6)
            df['Year'] = input_data.get('year', 2023)
            df['X'] = input_data.get('longitude', -122.4194)
            df['Y'] = input_data.get('latitude', 37.7749)
            df['IsWeekend'] = 1 if df['DayOfWeek_num'].iloc[0] >= 5 else 0
            df['IsNight'] = 1 if (df['Hour'].iloc[0] >= 22 or df['Hour'].iloc[0] <= 6) else 0
            
            # Calculate DistanceFromCenter using training data center
            if hasattr(self, 'feature_stats') and 'numeric_means' in self.feature_stats:
                center_x = self.feature_stats['numeric_means'].get('X', -122.4194)
                center_y = self.feature_stats['numeric_means'].get('Y', 37.7749)
            else:
                center_x = -122.4194
                center_y = 37.7749
            df['DistanceFromCenter'] = np.sqrt((df['X'] - center_x)**2 + (df['Y'] - center_y)**2)
            
            df['PdDistrict'] = input_data.get('district', 'CENTRAL')
            df['DistrictCrimeFreq'] = self.feature_stats['numeric_means'].get('DistrictCrimeFreq', 100)
            df['SpatialCluster'] = -1  # Default to noise
            df['GridX'] = (df['X'] * 100).round().astype(int)
            df['GridY'] = (df['Y'] * 100).round().astype(int)
            df['TimeOfDay'] = df['Hour'].apply(self._categorize_time)
            df['Season'] = df['Month'].apply(self._get_season)
            
            # Build feature matrix matching training features
            X = pd.DataFrame()
            numeric_features = [col for col in self.feature_names 
                              if not any(col.startswith(cat + '_') for cat in ['PdDistrict', 'TimeOfDay', 'Season', 'DayOfWeek'])]
            for feature in numeric_features:
                if feature in df.columns:
                    X[feature] = df[feature]
                else:
                    default_val = self.feature_stats['numeric_means'].get(feature, 0)
                    X[feature] = [default_val]
            
            for cat_feature, encoder_info in self.categorical_encoders.items():
                if cat_feature in df.columns:
                    value = df[cat_feature].iloc[0]
                    for encoded_col in encoder_info['encoded_columns']:
                        if encoded_col.endswith(f'_{value}'):
                            X[encoded_col] = [1]
                        else:
                            X[encoded_col] = [0]
                else:
                    for encoded_col in encoder_info['encoded_columns']:
                        X[encoded_col] = [0]
            
            for feature in self.feature_names:
                if feature not in X.columns:
                    X[feature] = [0]
            X = X[self.feature_names]
            
            # Scale features if needed
            use_scaling = False
            for name, config in [('RandomForest', False), ('GradientBoosting', False), ('LogisticRegression', True)]:
                if self.best_model_name == name:
                    use_scaling = config
                    break
            
            if use_scaling and self.scaler:
                X_processed = self.scaler.transform(X)
            else:
                X_processed = X.values
            
            # Get the LIME explainer
            explainer = self.get_lime_explainer()
            if explainer is None:
                raise ValueError("LIME explainer could not be created")
            
            # Generate explanation
            explanation = explainer.explain_instance(
                X_processed[0],
                self.best_model.predict_proba,
                num_features=6,  # Number of features to show
                num_samples=1000  # Number of samples for LIME
            )
            
            # Convert explanation to a dictionary for easy display
            exp_dict = {
                'feature_importance': [(feat, round(weight, 3)) for feat, weight in explanation.as_list()]
            }
            
            return exp_dict
            
        except Exception as e:
            print(f"❌ Error in generate_lime_explanation: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise Exception(f"Failed to generate LIME explanation: {str(e)}")
    
    def save_model(self, filepath='models/crime_prediction_system.pkl'):
        """Save the trained model"""
        try:
            print("Saving model...")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Create a copy without non-pickleable objects
            model_copy = AdvancedCrimePredictionSystem()
            model_copy.__dict__.update(self.__dict__)
            
            # Remove SHAP explainer if it's causing issues
            if hasattr(model_copy, 'shap_explainer'):
                model_copy.shap_explainer = None
                print("Removed SHAP explainer to avoid pickling issues")
            
            joblib.dump(model_copy, filepath)
            print(f"💾 Model saved to {filepath}")
            
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            
            # Try saving without explainable AI components
            try:
                print("Attempting to save model without explainable AI components...")
                model_minimal = AdvancedCrimePredictionSystem()
                
                # Copy essential components only
                essential_attrs = [
                    'models', 'scaler', 'label_encoder', 'feature_names', 
                    'best_model', 'best_model_name', 'train_data', 
                    'categorical_encoders', 'feature_stats', 'X_sample_for_lime'
                ]
                
                for attr in essential_attrs:
                    if hasattr(self, attr):
                        setattr(model_minimal, attr, getattr(self, attr))
                
                joblib.dump(model_minimal, filepath)
                print(f"💾 Model saved (minimal version) to {filepath}")
                
            except Exception as e2:
                print(f"❌ Even minimal save failed: {str(e2)}")
                raise e2
    
    def predict_crime(self, input_data):
        """Predict crime type based on input location and time data"""
        try:
            print(f"🔮 Making prediction for: {input_data}")
            
            # Create a DataFrame from input data
            df = pd.DataFrame([input_data])
            
            # Engineer features similar to training
            df['Hour'] = input_data.get('hour', 12)
            df['DayOfWeek_num'] = input_data.get('day_of_week', 1)
            df['Month'] = input_data.get('month', 6)
            df['Year'] = input_data.get('year', 2023)
            df['X'] = input_data.get('longitude', -122.4194)
            df['Y'] = input_data.get('latitude', 37.7749)
            df['IsWeekend'] = 1 if df['DayOfWeek_num'].iloc[0] >= 5 else 0
            df['IsNight'] = 1 if (df['Hour'].iloc[0] >= 22 or df['Hour'].iloc[0] <= 6) else 0
            
            # Calculate DistanceFromCenter using training data center
            if hasattr(self, 'feature_stats') and 'numeric_means' in self.feature_stats:
                center_x = self.feature_stats['numeric_means'].get('X', -122.4194)
                center_y = self.feature_stats['numeric_means'].get('Y', 37.7749)
            else:
                center_x = -122.4194
                center_y = 37.7749
                
            df['DistanceFromCenter'] = np.sqrt(
                (df['X'] - center_x)**2 + (df['Y'] - center_y)**2
            )
            
            # Add district info
            df['PdDistrict'] = input_data.get('district', 'CENTRAL')
            df['DistrictCrimeFreq'] = self.feature_stats['numeric_means'].get('DistrictCrimeFreq', 100)
            
            # Add spatial features
            df['SpatialCluster'] = -1  # Default to noise
            df['GridX'] = (df['X'] * 100).round().astype(int)
            df['GridY'] = (df['Y'] * 100).round().astype(int)
            
            # Add temporal categorical features
            df['TimeOfDay'] = df['Hour'].apply(self._categorize_time)
            df['Season'] = df['Month'].apply(self._get_season)
            
            # Build feature matrix matching training features
            X = pd.DataFrame()
            
            # Add numeric features
            numeric_features = [col for col in self.feature_names 
                              if not any(col.startswith(cat + '_') for cat in ['PdDistrict', 'TimeOfDay', 'Season', 'DayOfWeek'])]
            
            for feature in numeric_features:
                if feature in df.columns:
                    X[feature] = df[feature]
                else:
                    # Use mean from training data or default
                    default_val = self.feature_stats['numeric_means'].get(feature, 0)
                    X[feature] = [default_val]
            
            # Add categorical features (one-hot encoded)
            for cat_feature, encoder_info in self.categorical_encoders.items():
                if cat_feature in df.columns:
                    value = df[cat_feature].iloc[0]
                    for encoded_col in encoder_info['encoded_columns']:
                        if encoded_col.endswith(f'_{value}'):
                            X[encoded_col] = [1]
                        else:
                            X[encoded_col] = [0]
                else:
                    # Set all categorical features to 0 (unknown category)
                    for encoded_col in encoder_info['encoded_columns']:
                        X[encoded_col] = [0]
            
            # Ensure all training features are present
            for feature in self.feature_names:
                if feature not in X.columns:
                    X[feature] = [0]
            
            # Reorder columns to match training
            X = X[self.feature_names]
            
            print(f"Feature vector shape: {X.shape}")
            print(f"Feature vector sample: {X.iloc[0].head()}")
            
            # Scale features if needed
            use_scaling = False
            for name, config in [('RandomForest', False), ('GradientBoosting', False), ('LogisticRegression', True)]:
                if self.best_model_name == name:
                    use_scaling = config
                    break
            
            if use_scaling and self.scaler:
                X_processed = self.scaler.transform(X)
            else:
                X_processed = X.values
            
            # Make prediction
            if hasattr(self.best_model, 'predict_proba'):
                probabilities = self.best_model.predict_proba(X_processed)
                predicted_class = np.argmax(probabilities, axis=1)[0]
                confidence = np.max(probabilities, axis=1)[0]
                
                # Get top predictions
                top_indices = np.argsort(probabilities[0])[::-1][:3]
                top_predictions = []
                for idx in top_indices:
                    top_predictions.append({
                        'crime_type': self.label_encoder.inverse_transform([idx])[0],
                        'probability': float(probabilities[0][idx])
                    })
            else:
                prediction = self.best_model.predict(X_processed)
                predicted_class = prediction[0]
                confidence = 0.5  # Default confidence for non-probabilistic models
                top_predictions = [{
                    'crime_type': self.label_encoder.inverse_transform([predicted_class])[0],
                    'probability': confidence
                }]
            
            # Decode predicted class
            predicted_crime = self.label_encoder.inverse_transform([predicted_class])[0]
            
            result = {
                'predicted_crime': predicted_crime,
                'confidence': float(confidence),
                'top_predictions': top_predictions
            }
            
            print(f"✅ Prediction result: {result}")
            return result
            
        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            print(f"❌ {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
            raise Exception(error_msg)
    
    @classmethod
    def load_model(cls, filepath='models/crime_prediction_system.pkl'):
        """Load a trained model"""
        return joblib.load(filepath)

def train_system():
    """Train the complete system"""
    try:
        print("🚀 Starting Advanced Crime Prediction System Training")
        
        # Initialize system
        print("Initializing system...")
        system = AdvancedCrimePredictionSystem()
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        train_data = system.load_and_preprocess_data('data/train.csv', 'data/test.csv')
        
        # Prepare features and target
        print("Preparing features and target...")
        X, y_encoded, y_original = system.prepare_features_and_target()
        
        # Train models
        print("Training models...")
        results, X_val, y_val = system.train_models(X, y_encoded)
        
        # Setup explainable AI
        print("Setting up explainable AI...")
        system.setup_explainable_ai(X)
        
        # Save model
        print("Saving model...")
        system.save_model()
        
        print("\n✅ Training Complete!")
        print(f"Best Model: {system.best_model_name}")
        print(f"Features: {len(system.feature_names)}")
        print(f"Classes: {len(system.label_encoder.classes_)}")
        
        return system
        
    except Exception as e:
        print(f"❌ Fatal error in train_system: {str(e)}")
        print(f"Full traceback:")
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    system = train_system()