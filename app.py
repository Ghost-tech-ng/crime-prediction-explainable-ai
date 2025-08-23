# app.py - Fixed Web Interface for Crime Prediction
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time
import joblib
import os
import traceback
from main import AdvancedCrimePredictionSystem

# Configure page
st.set_page_config(
    page_title="🚨 Advanced Crime Prediction System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .debug-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.8rem;
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_model():
    """Load the trained model"""
    try:
        if os.path.exists('models/crime_prediction_system.pkl'):
            model = AdvancedCrimePredictionSystem.load_model('models/crime_prediction_system.pkl')
            
            # Check if model has required attributes
            required_attrs = ['best_model', 'best_model_name', 'label_encoder', 'feature_names']
            missing_attrs = [attr for attr in required_attrs if not hasattr(model, attr) or getattr(model, attr) is None]
            
            if missing_attrs:
                st.error(f"❌ Model is missing required attributes: {missing_attrs}")
                st.error("Please retrain the model by running: python main.py")
                return None
            
            st.success(f"✅ Model loaded successfully! Best model: {model.best_model_name}")
            return model
        else:
            st.error("❌ Model file not found at 'models/crime_prediction_system.pkl'")
            st.error("Please run 'python setup_and_run.py' first to train the model.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.error("Please retrain the model by running: python main.py")
        with st.expander("Error Details"):
            st.text(traceback.format_exc())
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🚨 Advanced Crime Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("### *Using Explainable Artificial Intelligence for Crime Prevention*")
    
    # Load model
    model = load_trained_model()
    
    if model is None:
        st.error("❌ Cannot proceed without a trained model.")
        st.info("Please run one of the following commands to train a model:")
        st.code("python setup_and_run.py")
        st.code("python main.py")
        st.stop()
    
    # Display model info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Model", model.best_model_name)
    with col2:
        st.metric("Features", len(model.feature_names) if model.feature_names else 0)
    with col3:
        st.metric("Crime Types", len(model.label_encoder.classes_) if model.label_encoder else 0)
    with col4:
        training_samples = len(model.train_data) if hasattr(model, 'train_data') and model.train_data is not None else "N/A"
        st.metric("Training Samples", training_samples)
    
    # Sidebar for navigation
    st.sidebar.title("🔧 Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["🎯 Crime Prediction", "📊 Model Analytics", "🗺️ Crime Hotspots", "🔧 Debug Info", "📋 About System"]
    )
    
    if page == "🎯 Crime Prediction":
        prediction_page(model)
    elif page == "📊 Model Analytics":
        analytics_page(model)
    elif page == "🗺️ Crime Hotspots":
        hotspots_page(model)
    elif page == "🔧 Debug Info":
        debug_page(model)
    else:
        about_page()

def prediction_page(model):
    """Crime prediction interface"""
    st.header("🎯 Crime Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📍 Location & Time Details")
        
        # Location inputs with adjustable step and wider range
        latitude = st.number_input(
            "Latitude", 
            min_value=34.0, max_value=42.0,
            value=37.7749, step=0.01,
            help="Latitude range: 34.0 - 42.0 (e.g., San Francisco area)"
        )
        
        longitude = st.number_input(
            "Longitude", 
            min_value=-124.0, max_value=-120.0,
            value=-122.4194, step=0.01,
            help="Longitude range: -124.0 - -120.0 (e.g., San Francisco area)"
        )
        
        # District with more options
        districts = [
            'SOUTHERN', 'NORTHERN', 'MISSION', 'CENTRAL', 'BAYVIEW',
            'RICHMOND', 'INGLESIDE', 'TARAVAL', 'PARK', 'TENDERLOIN'
        ]
        district = st.selectbox("Police District", districts, index=4)  # Default to BAYVIEW
        
        # Time inputs
        date_input = st.date_input("Date", datetime.now())
        time_input = st.time_input("Time", time(12, 0))
        
        # Combine date and time
        datetime_combined = datetime.combine(date_input, time_input)
        
        # Display debug info
        st.subheader("🔍 Input Summary")
        st.markdown(f"""
        <div class="debug-box">
        <strong>Coordinates:</strong> ({latitude:.4f}, {longitude:.4f})<br>
        <strong>District:</strong> {district}<br>
        <strong>DateTime:</strong> {datetime_combined}<br>
        <strong>Hour:</strong> {datetime_combined.hour}<br>
        <strong>Day of Week:</strong> {datetime_combined.weekday()} ({datetime_combined.strftime('%A')})<br>
        <strong>Month:</strong> {datetime_combined.month}<br>
        <strong>Year:</strong> {datetime_combined.year}
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.subheader("⚡ Prediction Results")
        
        if st.button("🔮 Predict Crime Type", type="primary"):
            with st.spinner("Making prediction..."):
                # Prepare input data
                input_data = {
                    'latitude': latitude,
                    'longitude': longitude,
                    'district': district,
                    'hour': datetime_combined.hour,
                    'day_of_week': datetime_combined.weekday(),
                    'month': datetime_combined.month,
                    'year': datetime_combined.year
                }
                
                try:
                    # Make prediction
                    result = model.predict_crime(input_data)
                    
                    # Display main prediction
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>🎯 Predicted Crime Type</h2>
                        <h1>{result['predicted_crime']}</h1>
                        <h3>Confidence: {result['confidence']:.1%}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display top predictions if available
                    if result.get('top_predictions'):
                        st.subheader("📈 Top 3 Predictions")
                        for i, pred in enumerate(result['top_predictions'][:3]):
                            confidence_color = "green" if pred['probability'] > 0.5 else "orange" if pred['probability'] > 0.3 else "red"
                            st.markdown(f"""
                            <div class="metric-box">
                                <strong>{i+1}. {pred['crime_type']}</strong><br>
                                <span style="color: {confidence_color}">Probability: {pred['probability']:.1%}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Risk assessment
                    confidence = result['confidence']
                    if confidence > 0.7:
                        risk_level = "🔴 HIGH CONFIDENCE"
                        risk_color = "red"
                        risk_desc = "Model is very confident in this prediction"
                    elif confidence > 0.4:
                        risk_level = "🟡 MEDIUM CONFIDENCE"
                        risk_color = "orange"
                        risk_desc = "Model has moderate confidence in this prediction"
                    else:
                        risk_level = "🟢 LOW CONFIDENCE"
                        risk_color = "green"
                        risk_desc = "Model has low confidence - multiple crime types are possible"
                    
                    st.markdown(f"### Confidence Level: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
                    st.caption(risk_desc)
                    
                    # Show prediction details in expander
                    with st.expander("🔍 Prediction Details", expanded=False):
                        st.json(result)
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
                    st.error("Please check the debug page for more information.")
                    with st.expander("Error Details"):
                        st.text(traceback.format_exc())
        
        # Quick prediction buttons
        st.subheader("🚀 Quick Predictions")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🌃 Downtown Night", help="Union Square area at 10 PM"):
                quick_predict(model, "downtown", "night")
        
        with col_b:
            if st.button("🏙️ Mission Day", help="Mission District at 2 PM"):
                quick_predict(model, "mission", "day")
        
        col_c, col_d = st.columns(2)
        
        with col_c:
            if st.button("🌅 Marina Morning", help="Marina District at 8 AM"):
                quick_predict(model, "marina", "morning")
                
        with col_d:
            if st.button("🌆 Tenderloin Evening", help="Tenderloin at 6 PM"):
                quick_predict(model, "tenderloin", "evening")
    
    # Show explainable AI section
    st.subheader("🧠 Model Explanation")
    try:
        # Try to create LIME explanation
        lime_explainer = model.get_lime_explainer()
        if lime_explainer is not None:
            st.success("✅ LIME explainer available for detailed explanations")
            if st.button("🔍 Generate LIME Explanation"):
                with st.spinner("Generating explanation..."):
                    # Use session state to store and display explanation
                    if 'lime_explanation' not in st.session_state:
                        st.session_state.lime_explanation = None
                    input_data = {
                        'latitude': latitude,
                        'longitude': longitude,
                        'district': district,
                        'hour': datetime_combined.hour,
                        'day_of_week': datetime_combined.weekday(),
                        'month': datetime_combined.month,
                        'year': datetime_combined.year
                    }
                    st.session_state.lime_explanation = model.generate_lime_explanation(input_data)
                    if st.session_state.lime_explanation:
                        st.subheader("📊 LIME Explanation")
                        lime_df = pd.DataFrame(
                            st.session_state.lime_explanation['feature_importance'],
                            columns=['Feature', 'Importance']
                        )
                        st.dataframe(lime_df, use_container_width=True)
                        fig = px.bar(
                            lime_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title="LIME Feature Importance",
                            color='Importance',
                            color_continuous_scale='RdBu'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("❌ Failed to generate LIME explanation.")
        else:
            st.info("LIME explainer not available - using feature importance instead")
            
        # Show feature importance if available
        if hasattr(model.best_model, 'feature_importances_') and model.feature_names:
            importance_df = pd.DataFrame({
                'Feature': model.feature_names,
                'Importance': model.best_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            st.subheader("📊 Top 10 Most Important Features")
            fig = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title="Feature Importance for Prediction",
                color='Importance',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as explain_error:
        st.warning(f"⚠️ Explainable AI features temporarily unavailable: {str(explain_error)}")

def quick_predict(model, area, time_period):
    """Quick prediction for common scenarios"""
    area_coords = {
        "downtown": (37.7879, -122.4075, "CENTRAL"),
        "mission": (37.7599, -122.4148, "MISSION"),
        "marina": (37.8021, -122.4340, "NORTHERN"),
        "tenderloin": (37.7835, -122.4134, "TENDERLOIN")
    }
    
    time_hours = {
        "night": 22,
        "day": 14,
        "morning": 8,
        "evening": 18
    }
    
    lat, lon, district = area_coords.get(area, (37.7749, -122.4194, "CENTRAL"))
    hour = time_hours.get(time_period, 12)
    
    input_data = {
        'latitude': lat,
        'longitude': lon,
        'district': district,
        'hour': hour,
        'day_of_week': 1,  # Tuesday
        'month': datetime.now().month,
        'year': datetime.now().year
    }
    
    try:
        result = model.predict_crime(input_data)
        st.success(f"🎯 **{area.title()} {time_period.title()}**: {result['predicted_crime']} ({result['confidence']:.1%} confidence)")
        
        # Show top 3 predictions
        if result.get('top_predictions'):
            top_3 = ", ".join([f"{pred['crime_type']} ({pred['probability']:.1%})" 
                              for pred in result['top_predictions'][:3]])
            st.info(f"Top predictions: {top_3}")
            
    except Exception as e:
        st.error(f"❌ Quick prediction failed: {e}")

def analytics_page(model):
    """Model analytics and performance"""
    st.header("📊 Model Analytics")
    
    # Model info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best Model", model.best_model_name if hasattr(model, 'best_model_name') else "Unknown")
    
    with col2:
        st.metric("Features", len(model.feature_names) if hasattr(model, 'feature_names') and model.feature_names else 0)
    
    with col3:
        st.metric("Crime Categories", len(model.label_encoder.classes_) if hasattr(model, 'label_encoder') and model.label_encoder else 0)
    
    with col4:
        if hasattr(model, 'best_model') and model.best_model and hasattr(model.best_model, 'n_estimators'):
            st.metric("Estimators", model.best_model.n_estimators)
        elif hasattr(model, 'best_model') and model.best_model:
            st.metric("Model Type", type(model.best_model).__name__)
        else:
            st.metric("Model Type", "Unknown")
    
    # Feature importance (if available)
    if (hasattr(model, 'best_model') and model.best_model and 
        hasattr(model.best_model, 'feature_importances_') and 
        hasattr(model, 'feature_names') and model.feature_names):
        
        st.subheader("🎯 Feature Importance")
        
        importance_df = pd.DataFrame({
            'Feature': model.feature_names,
            'Importance': model.best_model.feature_importances_
        }).sort_values('Importance', ascending=True).tail(15)
        
        fig = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature',
            title="Top 15 Most Important Features",
            color='Importance',
            color_continuous_scale='viridis',
            height=600
        )
        fig.update_layout(xaxis_title="Feature Importance", yaxis_title="Features")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show feature importance table
        with st.expander("📋 Feature Importance Table"):
            st.dataframe(importance_df.sort_values('Importance', ascending=False))
    else:
        st.warning("⚠️ Feature importance not available for this model type")
    
    # Crime categories
    if hasattr(model, 'label_encoder') and model.label_encoder:
        st.subheader("🏷️ Crime Categories")
        categories_df = pd.DataFrame({
            'Crime Type': model.label_encoder.classes_,
            'Encoded Index': range(len(model.label_encoder.classes_))
        })
        
        if hasattr(model, 'train_data') and model.train_data is not None and 'Category' in model.train_data.columns:
            # Add crime counts
            crime_counts = model.train_data['Category'].value_counts()
            categories_df['Count'] = categories_df['Crime Type'].map(crime_counts).fillna(0)
            categories_df = categories_df.sort_values('Count', ascending=False)
        
        st.dataframe(categories_df, use_container_width=True)
    
    # Model parameters
    if hasattr(model, 'best_model') and model.best_model:
        st.subheader("🔧 Model Parameters")
        try:
            if hasattr(model.best_model, 'get_params'):
                params = model.best_model.get_params()
                st.json(params)
        except Exception as e:
            st.error(f"Error retrieving model parameters: {e}")

def hotspots_page(model):
    """Crime hotspots visualization"""
    st.header("🗺️ Crime Hotspots Analysis")
    
    if hasattr(model, 'train_data') and model.train_data is not None:
        # Sample data for visualization (limit for performance)
        max_points = 2000
        if len(model.train_data) > max_points:
            sample_data = model.train_data.sample(max_points)
            st.info(f"Showing {max_points} random samples from {len(model.train_data)} total records")
        else:
            sample_data = model.train_data
        
        # Check if required columns exist
        required_cols = ['X', 'Y']
        if not all(col in sample_data.columns for col in required_cols):
            st.error("❌ Required columns (X, Y) not found in training data")
            return
        
        # Crime density map
        st.subheader("🔥 Crime Density Heatmap")
        
        try:
            if 'CrimeSeverity' in sample_data.columns:
                fig = px.density_mapbox(
                    sample_data,
                    lat='Y',
                    lon='X',
                    z='CrimeSeverity',
                    radius=10,
                    center=dict(lat=37.7749, lon=-122.4194),
                    zoom=11,
                    mapbox_style="open-street-map",
                    color_continuous_scale="Reds",
                    title="San Francisco Crime Density (by Severity)",
                    height=600
                )
            else:
                # Create a basic scatter plot if no severity data
                plot_data = sample_data.head(1000)  # Limit points for performance
                if 'Category' in plot_data.columns:
                    fig = px.scatter_mapbox(
                        plot_data,
                        lat='Y',
                        lon='X',
                        color='Category',
                        size_max=8,
                        zoom=11,
                        center=dict(lat=37.7749, lon=-122.4194),
                        mapbox_style="open-street-map",
                        title="Crime Locations by Type",
                        height=600
                    )
                else:
                    fig = px.scatter_mapbox(
                        plot_data,
                        lat='Y',
                        lon='X',
                        size_max=8,
                        zoom=11,
                        center=dict(lat=37.7749, lon=-122.4194),
                        mapbox_style="open-street-map",
                        title="Crime Locations",
                        height=600
                    )
            
            fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error creating map visualization: {e}")
        
        # District analysis
        if 'PdDistrict' in model.train_data.columns:
            st.subheader("📊 Crime by District")
            district_counts = model.train_data['PdDistrict'].value_counts().head(10)
            
            fig = px.bar(
                x=district_counts.values,
                y=district_counts.index,
                orientation='h',
                title="Crime Count by Police District",
                color=district_counts.values,
                color_continuous_scale='Blues',
                height=500
            )
            fig.update_layout(xaxis_title="Number of Crimes", yaxis_title="Police District")
            st.plotly_chart(fig, use_container_width=True)
        
        # Temporal patterns
        st.subheader("⏰ Temporal Crime Patterns")
        
        col1, col2 = st.columns(2)
        
        # Hourly pattern
        if 'Hour' in model.train_data.columns:
            with col1:
                hourly_crimes = model.train_data.groupby('Hour').size()
                fig = px.line(
                    x=hourly_crimes.index,
                    y=hourly_crimes.values,
                    title="Crime Patterns by Hour of Day",
                    labels={'x': 'Hour of Day', 'y': 'Number of Crimes'},
                    markers=True
                )
                fig.update_layout(xaxis=dict(dtick=2))
                st.plotly_chart(fig, use_container_width=True)
        
        # Day of week pattern
        if 'DayOfWeek_num' in model.train_data.columns:
            with col2:
                dow_crimes = model.train_data.groupby('DayOfWeek_num').size()
                days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                fig = px.bar(
                    x=days,
                    y=dow_crimes.values,
                    title="Crime Patterns by Day of Week",
                    color=dow_crimes.values,
                    color_continuous_scale='Oranges'
                )
                fig.update_layout(xaxis_title="Day of Week", yaxis_title="Number of Crimes")
                st.plotly_chart(fig, use_container_width=True)
        
        # Crime type distribution
        if 'Category' in model.train_data.columns:
            st.subheader("📈 Crime Type Distribution")
            crime_dist = model.train_data['Category'].value_counts().head(15)
            
            fig = px.pie(
                values=crime_dist.values,
                names=crime_dist.index,
                title="Top 15 Crime Types Distribution"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("⚠️ No training data available for visualization")
        st.info("Please retrain the model to enable hotspots analysis")

def debug_page(model):
    """Debug information page"""
    st.header("🔧 Debug Information")
    
    st.subheader("🔍 Model Information")
    
    # Basic model info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Details:**")
        st.write(f"- Best Model: {getattr(model, 'best_model_name', 'Unknown')}")
        st.write(f"- Model Type: {type(model.best_model).__name__ if hasattr(model, 'best_model') and model.best_model else 'Unknown'}")
        st.write(f"- Number of Features: {len(model.feature_names) if hasattr(model, 'feature_names') and model.feature_names else 0}")
        st.write(f"- Number of Classes: {len(model.label_encoder.classes_) if hasattr(model, 'label_encoder') and model.label_encoder else 0}")
        
        if hasattr(model, 'scaler') and model.scaler:
            st.write(f"- Feature Scaling: Enabled")
        else:
            st.write(f"- Feature Scaling: Disabled")
    
    with col2:
        st.markdown("**Training Data:**")
        if hasattr(model, 'train_data') and model.train_data is not None:
            st.write(f"- Training Samples: {len(model.train_data)}")
            st.write(f"- Data Columns: {len(model.train_data.columns)}")
            st.write(f"- Memory Usage: {model.train_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        else:
            st.write("- No training data available")
    
    # Feature information
    st.subheader("📊 Feature Information")
    
    if hasattr(model, 'feature_names') and model.feature_names:
        feature_df = pd.DataFrame({
            'Feature Name': model.feature_names,
            'Index': range(len(model.feature_names))
        })
        
        # Add feature types
        feature_types = []
        for feature in model.feature_names:
            if any(feature.startswith(cat + '_') for cat in ['PdDistrict', 'TimeOfDay', 'Season', 'DayOfWeek']):
                feature_types.append('Categorical (One-hot)')
            elif feature in ['Hour', 'DayOfWeek_num', 'Month', 'Year', 'IsWeekend', 'IsNight']:
                feature_types.append('Temporal')
            elif feature in ['X', 'Y', 'DistanceFromCenter', 'GridX', 'GridY', 'SpatialCluster']:
                feature_types.append('Spatial')
            else:
                feature_types.append('Other')
        
        feature_df['Type'] = feature_types
        st.dataframe(feature_df, use_container_width=True)
    else:
        st.warning("⚠️ No feature information available")
    
    # Class information
    st.subheader("🏷️ Class Information")
    
    if hasattr(model, 'label_encoder') and model.label_encoder:
        class_df = pd.DataFrame({
            'Crime Type': model.label_encoder.classes_,
            'Encoded Value': range(len(model.label_encoder.classes_))
        })
        
        if hasattr(model, 'train_data') and model.train_data is not None and 'Category' in model.train_data.columns:
            class_counts = model.train_data['Category'].value_counts()
            class_df['Count'] = class_df['Crime Type'].map(class_counts).fillna(0)
            class_df['Percentage'] = (class_df['Count'] / class_df['Count'].sum() * 100).round(2)
        
        st.dataframe(class_df, use_container_width=True)
    else:
        st.warning("⚠️ No label encoder information available")
    
    # Model parameters
    st.subheader("⚙️ Model Parameters")
    
    if hasattr(model, 'best_model') and model.best_model:
        try:
            if hasattr(model.best_model, 'get_params'):
                params = model.best_model.get_params()
                st.json(params)
            else:
                st.info("Model parameters not available for this model type")
        except Exception as e:
            st.error(f"Error retrieving model parameters: {e}")
    else:
        st.warning("⚠️ No trained model available")
    
    # Test prediction functionality
    st.subheader("🧪 Test Prediction")
    
    if st.button("Run Test Prediction"):
        test_input = {
            'latitude': 37.7749,
            'longitude': -122.4194,
            'district': 'CENTRAL',
            'hour': 14,
            'day_of_week': 1,
            'month': 6,
            'year': 2023
        }
        
        try:
            with st.spinner("Running test prediction..."):
                result = model.predict_crime(test_input)
                st.success("✅ Test prediction successful!")
                st.json(result)
        except Exception as e:
            st.error(f"❌ Test prediction failed: {e}")
            st.text(traceback.format_exc())
    
    # Model attributes check
    st.subheader("🔍 Model Attributes Check")
    
    required_attrs = [
        'best_model', 'best_model_name', 'label_encoder', 'feature_names',
        'scaler', 'categorical_encoders', 'feature_stats'
    ]
    
    attr_status = []
    for attr in required_attrs:
        has_attr = hasattr(model, attr)
        is_none = getattr(model, attr, None) is None
        status = "✅ Present" if has_attr and not is_none else "❌ Missing/None" if has_attr else "❌ Not Found"
        attr_status.append({
            'Attribute': attr,
            'Status': status,
            'Type': str(type(getattr(model, attr, None)).__name__) if has_attr and not is_none else "N/A"
        })
    
    attr_df = pd.DataFrame(attr_status)
    st.dataframe(attr_df, use_container_width=True)
    
    # LIME explainer check
    st.subheader("🧠 Explainable AI Status")
    
    try:
        lime_explainer = model.get_lime_explainer()
        if lime_explainer is not None:
            st.success("✅ LIME explainer can be created successfully")
        else:
            st.warning("⚠️ LIME explainer cannot be created - missing sample data")
    except Exception as e:
        st.error(f"❌ LIME explainer error: {e}")
    
    # SHAP explainer check
    if hasattr(model, 'shap_explainer'):
        if model.shap_explainer is not None:
            st.success("✅ SHAP explainer is available")
        else:
            st.info("ℹ️ SHAP explainer was removed during model saving (to avoid pickling issues)")
    else:
        st.warning("⚠️ No SHAP explainer attribute found")

def about_page():
    """About the system"""
    st.header("📋 About the Advanced Crime Prediction System")
    
    st.markdown("""
    ## 🎯 System Overview
    
    This **Advanced Crime Prediction System** uses state-of-the-art machine learning algorithms 
    combined with explainable AI techniques to predict crime types in San Francisco.
    
    ### 🚀 Key Features:
    
    - **Multiple ML Algorithms**: Random Forest, Gradient Boosting, Logistic Regression
    - **Advanced Feature Engineering**: Temporal, spatial, and contextual features
    - **Explainable AI**: SHAP and LIME for model interpretability
    - **Real-time Predictions**: Interactive web interface for instant predictions
    - **Comprehensive Analytics**: Feature importance and performance metrics
    - **Geospatial Analysis**: Crime hotspot identification and mapping
    
    ### 🛠️ Technical Implementation:
    
    #### Data Processing:
    - **Temporal Features**: Hour, day of week, season, time categories
    - **Spatial Features**: Coordinates, distance from city center, spatial clustering
    - **Contextual Features**: Police district, crime severity mapping
    
    #### Machine Learning Models:
    1. **Random Forest**: Ensemble method for robust predictions
    2. **Gradient Boosting**: Sequential learning for improved accuracy
    3. **Logistic Regression**: Linear model for interpretability
    
    #### Explainable AI:
    - **SHAP (SHapley Additive exPlanations)**: Global and local feature importance
    - **LIME (Local Interpretable Model-agnostic Explanations)**: Instance-level explanations
    
    ### 📊 Model Performance:
    - Trained on San Francisco crime dataset
    - Advanced feature engineering with 15+ engineered features
    - Cross-validation for robust performance estimation
    - Automatic best model selection
    
    ### 🚨 Use Cases:
    - **Law Enforcement**: Resource allocation and patrol planning
    - **Urban Planning**: Crime prevention infrastructure
    - **Public Safety**: Community awareness and prevention
    - **Research**: Crime pattern analysis and policy development
    
    ### 🔧 Recent Fixes:
    - **Resolved LIME Pickling Issue**: Fixed model serialization problems
    - **Improved Error Handling**: Better error messages and recovery
    - **Enhanced Model Loading**: More robust model validation
    - **Optimized Performance**: Reduced memory usage and faster predictions
    
    ### 🚀 Getting Started:
    
    1. **Train the Model**: Run `python setup_and_run.py` or `python main.py`
    2. **Launch Web Interface**: Run `streamlit run app.py`
    3. **Make Predictions**: Use the prediction interface to analyze crime patterns
    4. **Explore Analytics**: View model performance and feature importance
    5. **Analyze Hotspots**: Visualize crime patterns on interactive maps
    
    ---
    
    **Developed by**: Advanced ML Team  
    **Project**: Crime Prediction using Explainable AI  
    **Technology Stack**: Python, Scikit-learn, TensorFlow, SHAP, LIME, Streamlit  
    **Version**: 2.0 (Fixed LIME Serialization)
    """)

if __name__ == "__main__":
    main()