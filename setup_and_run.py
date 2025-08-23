# setup_and_run.py - Improved Setup and Training Script
import os
import sys
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback

def check_requirements():
    """Check if all required packages are installed"""
    print("🔍 Checking requirements...")
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn',
        'tensorflow', 'shap', 'lime', 'plotly', 'streamlit', 'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'scikit-learn':
                __import__('sklearn')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
    else:
        print("✅ All required packages are installed!")

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    directories = ['data', 'models', 'outputs', 'logs']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"   Created: {directory}/")
        else:
            print(f"   Exists: {directory}/")

def check_data_files():
    """Check if data files exist"""
    print("📊 Checking data files...")
    
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    
    if not os.path.exists(train_path):
        print(f"❌ Training data not found at {train_path}")
        print("Please ensure your train.csv file is in the data/ folder")
        return False
    
    if not os.path.exists(test_path):
        print(f"⚠️  Test data not found at {test_path} (optional)")
    
    # Check data format
    try:
        train_df = pd.read_csv(train_path, nrows=5)  # Read first 5 rows
        print(f"✅ Training data found: {train_path}")
        print(f"   Shape: {pd.read_csv(train_path).shape}")
        print(f"   Columns: {list(train_df.columns)}")
        
        # Check required columns
        required_columns = ['X', 'Y', 'Category']
        missing_cols = [col for col in required_columns if col not in train_df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        # Check data quality
        full_df = pd.read_csv(train_path)
        print(f"   Unique categories: {full_df['Category'].nunique()}")
        print(f"   Missing values: {full_df[required_columns].isnull().sum().sum()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading training data: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return False

def create_diverse_sample_data():
    """Create diverse sample data with better distribution"""
    print("📝 Creating diverse sample data for demonstration...")
    
    np.random.seed(42)
    n_samples = 5000  # Increased sample size
    
    # San Francisco coordinate bounds
    lat_min, lat_max = 37.70, 37.80
    lon_min, lon_max = -122.50, -122.30
    
    # Define crime types with realistic distributions
    crime_types = [
        ('LARCENY/THEFT', 0.25),
        ('OTHER OFFENSES', 0.15),
        ('NON-CRIMINAL', 0.10),
        ('ASSAULT', 0.08),
        ('DRUG/NARCOTIC', 0.07),
        ('VEHICLE THEFT', 0.06),
        ('VANDALISM', 0.05),
        ('WARRANTS', 0.05),
        ('BURGLARY', 0.04),
        ('SUSPICIOUS OCC', 0.04),
        ('DRUNKENNESS', 0.03),
        ('FRAUD', 0.03),
        ('ROBBERY', 0.02),
        ('FAMILY OFFENSES', 0.02),
        ('FORGERY/COUNTERFEITING', 0.01)
    ]
    
    # Create weighted crime type samples
    crime_names = [ct[0] for ct in crime_types]
    crime_weights = [ct[1] for ct in crime_types]
    categories = np.random.choice(crime_names, size=n_samples, p=crime_weights)
    
    # Define districts with realistic distributions
    districts = [
        ('SOUTHERN', 0.15),
        ('MISSION', 0.12),
        ('NORTHERN', 0.11),
        ('CENTRAL', 0.10),
        ('BAYVIEW', 0.09),
        ('INGLESIDE', 0.08),
        ('TARAVAL', 0.08),
        ('TENDERLOIN', 0.08),
        ('RICHMOND', 0.10),
        ('PARK', 0.09)
    ]
    
    district_names = [d[0] for d in districts]
    district_weights = [d[1] for d in districts]
    pd_districts = np.random.choice(district_names, size=n_samples, p=district_weights)
    
    # Create realistic temporal patterns
    # More crimes during certain hours (evening/night peak)
    hour_weights = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.03,  # 0-5 AM
                            0.04, 0.05, 0.06, 0.07, 0.08, 0.09,   # 6-11 AM
                            0.09, 0.08, 0.07, 0.06, 0.07, 0.08,   # 12-5 PM
                            0.09, 0.08, 0.07, 0.06, 0.04, 0.03])  # 6-11 PM
    
    hours = np.random.choice(24, size=n_samples, p=hour_weights)
    
    # Generate dates over the past year with some seasonal variation
    start_date = datetime.now() - timedelta(days=365)
    dates = []
    for i in range(n_samples):
        # Add some seasonal bias (more crimes in summer)
        days_offset = np.random.exponential(scale=100) % 365
        if np.random.random() < 0.3:  # 30% chance of summer bias
            days_offset = (days_offset + 150) % 365  # Bias toward summer
        
        sample_date = start_date + timedelta(days=days_offset)
        sample_date = sample_date.replace(hour=hours[i], 
                                        minute=np.random.randint(0, 60),
                                        second=np.random.randint(0, 60))
        dates.append(sample_date)
    
    # Generate coordinates with some clustering around district centers
    district_centers = {
        'SOUTHERN': (37.7749, -122.4194),
        'MISSION': (37.7599, -122.4148),
        'NORTHERN': (37.8021, -122.4340),
        'CENTRAL': (37.7879, -122.4075),
        'BAYVIEW': (37.7312, -122.3826),
        'INGLESIDE': (37.7249, -122.4657),
        'TARAVAL': (37.7431, -122.4660),
        'TENDERLOIN': (37.7835, -122.4134),
        'RICHMOND': (37.7806, -122.4644),
        'PARK': (37.7691, -122.4561)
    }
    
    latitudes = []
    longitudes = []
    
    for district in pd_districts:
        center_lat, center_lon = district_centers.get(district, (37.7749, -122.4194))
        
        # Add some random spread around district center
        lat = np.random.normal(center_lat, 0.01)  # ~1km spread
        lon = np.random.normal(center_lon, 0.01)
        
        # Ensure coordinates stay within SF bounds
        lat = np.clip(lat, lat_min, lat_max)
        lon = np.clip(lon, lon_min, lon_max)
        
        latitudes.append(lat)
        longitudes.append(lon)
    
    # Generate day of week
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    days_of_week = [date.strftime('%A') for date in dates]
    
    # Generate realistic addresses
    street_names = [
        'Market St', 'Mission St', 'Van Ness Ave', 'Geary Blvd', 'Valencia St',
        'Fillmore St', '16th St', '24th St', 'Castro St', 'Union St',
        'Polk St', 'Divisadero St', 'Irving St', 'Taraval St', 'Noriega St'
    ]
    
    addresses = []
    for i in range(n_samples):
        block = np.random.randint(100, 2000)
        street = np.random.choice(street_names)
        addresses.append(f"{block} {street}")
    
    # Create resolution with realistic distribution
    resolutions = np.random.choice(
        ['NONE', 'ARREST, BOOKED', 'CLEARED-CONTACT JUVENILE', 'COMPLAINANT REFUSES TO PROSECUTE'],
        size=n_samples,
        p=[0.6, 0.25, 0.1, 0.05]
    )
    
    # Create the sample data
    sample_data = {
        'Dates': dates,
        'Category': categories,
        'Descript': [f"Sample description for {cat}" for cat in categories],
        'DayOfWeek': days_of_week,
        'PdDistrict': pd_districts,
        'Resolution': resolutions,
        'Address': addresses,
        'X': longitudes,
        'Y': latitudes
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    # Sort by date for realism
    sample_df = sample_df.sort_values('Dates').reset_index(drop=True)
    
    # Save sample data
    train_size = int(0.8 * len(sample_df))
    train_df = sample_df[:train_size]
    test_df = sample_df[train_size:]
    
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print("✅ Diverse sample data created successfully!")
    print(f"   - train.csv: {len(train_df)} samples")
    print(f"   - test.csv: {len(test_df)} samples")
    print(f"   - Crime types: {len(train_df['Category'].unique())}")
    print(f"   - Districts: {len(train_df['PdDistrict'].unique())}")
    print(f"   - Date range: {train_df['Dates'].min()} to {train_df['Dates'].max()}")
    
    # Show sample statistics
    print(f"\n📊 Sample Data Statistics:")
    print(f"   - Most common crime: {train_df['Category'].value_counts().index[0]}")
    print(f"   - Most active district: {train_df['PdDistrict'].value_counts().index[0]}")
    print(f"   - Peak hour: {pd.to_datetime(train_df['Dates']).dt.hour.value_counts().index[0]}:00")

def train_model():
    """Train the crime prediction model"""
    print("\n🚀 Starting Model Training...")
    print("=" * 50)
    
    try:
        from main import train_system
        system = train_system()
        print("\n✅ Model training completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        print(f"Full traceback:")
        print(traceback.format_exc())
        return False

def launch_web_interface():
    """Launch the Streamlit web interface"""
    print("\n🌐 Launching Web Interface...")
    print("=" * 50)
    print("The web interface will open in your browser at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Web interface stopped by user")
    except Exception as e:
        print(f"❌ Failed to launch web interface: {e}")
        print(f"Full traceback: {traceback.format_exc()}")

def run_quick_test():
    """Run a quick test of the system"""
    print("\n🧪 Running Quick System Test...")
    print("=" * 40)
    
    try:
        from main import AdvancedCrimePredictionSystem
        
        # Load the trained model
        if not os.path.exists('models/crime_prediction_system.pkl'):
            print("❌ No trained model found. Please train the model first.")
            return False
        
        print("Loading trained model...")
        system = AdvancedCrimePredictionSystem.load_model('models/crime_prediction_system.pkl')
        
        # Test prediction
        print("Testing prediction functionality...")
        test_input = {
            'latitude': 37.7749,
            'longitude': -122.4194,
            'district': 'CENTRAL',
            'hour': 14,
            'day_of_week': 1,
            'month': 6,
            'year': 2023
        }
        
        result = system.predict_crime(test_input)
        
        print("✅ Quick test successful!")
        print(f"   Test prediction: {result['predicted_crime']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Top 3 predictions: {len(result.get('top_predictions', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {str(e)}")
        print(f"Full traceback:")
        print(traceback.format_exc())
        return False

def main():
    """Main setup and run function"""
    print("🚨 Advanced Crime Prediction System - Setup & Launch")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Step 1: Check requirements
        check_requirements()
        print()
        
        # Step 2: Setup directories
        setup_directories()
        print()
        
        # Step 3: Check data files
        data_exists = check_data_files()
        
        if not data_exists:
            response = input("\n❓ No data found. Create diverse sample data for demonstration? (y/n): ")
            if response.lower() == 'y':
                create_diverse_sample_data()
            else:
                print("❌ Cannot proceed without data. Please add your train.csv to the data/ folder.")
                return
        
        print()
        
        # Step 4: Train model
        training_success = train_model()
        
        if not training_success:
            print("❌ Cannot proceed without trained model.")
            return
        
        print()
        
        # Step 5: Run quick test
        test_success = run_quick_test()
        
        if not test_success:
            print("⚠️  System test failed, but you can still try the web interface.")
        
        print()
        
        # Step 6: Launch web interface
        response = input("🌐 Launch web interface now? (y/n): ")
        if response.lower() == 'y':
            launch_web_interface()
        else:
            print("✅ Setup complete! Run 'streamlit run app.py' to launch the web interface.")
            print("\n📋 Quick Start Commands:")
            print("   - Launch web app: streamlit run app.py")
            print("   - Retrain model: python main.py")
            print("   - Run setup again: python setup_and_run.py")
    
    except Exception as e:
        print(f"❌ Fatal error in main: {str(e)}")
        print(f"Full traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()