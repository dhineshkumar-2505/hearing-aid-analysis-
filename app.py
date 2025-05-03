import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Hearing Loss Risk Assessment",
    page_icon="🎧",
    layout="wide"
)

# App title and description
st.title("🎧 Hearing Loss Risk Assessment Tool")
st.markdown("""
This application predicts the risk of hearing loss based on audiometry data, earphone usage, 
and personal factors. It uses machine learning to analyze your hearing thresholds across 
different frequencies.
""")

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Data Generation", "Model Training", "About"])

with tab1:
    st.header("Hearing Loss Prediction")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        age = st.number_input("Age", min_value=5, max_value=100, value=30)
        gender = st.radio("Gender", ["Male", "Female"])
        
        st.subheader("Earphone Usage")
        earphone_type = st.selectbox("Earphone Type", ["In-ear", "Over-ear", "None"])
        usage_hours = st.slider("Daily Usage (hours)", 0, 12, 2)
        usage_years = st.slider("Years of Regular Use", 0, 20, 3)
        volume_level = st.slider("Typical Volume Level (1-10)", 1, 10, 6)
    
    with col2:
        st.subheader("Audiometry Results (dB)")
        st.markdown("Enter hearing thresholds in decibels (dB) for each frequency:")
        
        f250 = st.slider("250 Hz", 0, 100, 15)
        f500 = st.slider("500 Hz", 0, 100, 15)
        f1k = st.slider("1000 Hz", 0, 100, 20)
        f2k = st.slider("2000 Hz", 0, 100, 20)
        f4k = st.slider("4000 Hz", 0, 100, 25)
        f8k = st.slider("8000 Hz", 0, 100, 30)
    
    # Button to make prediction
    predict_clicked = st.button("Analyze Hearing Risk", type="primary")
    
    if predict_clicked:
        # Create input dataframe
        input_data = {
            'age': age,
            'gender': 1 if gender == "Male" else 0,
            'earphone_type': earphone_type,
            'usage_hours': usage_hours,
            'usage_years': usage_years, 
            'volume_level': volume_level,
            'f250': f250,
            'f500': f500,
            'f1k': f1k,
            'f2k': f2k,
            'f4k': f4k,
            'f8k': f8k
        }
        
        # Simple rule-based risk assessment
        high_freq_avg = (f4k + f8k) / 2
        speech_freq_avg = (f500 + f1k + f2k + f4k) / 4
        
        risk_factors = 0
        risk_factors += 1 if age > 50 else 0
        risk_factors += 1 if usage_hours > 4 and usage_years > 5 else 0
        risk_factors += 1 if volume_level > 7 else 0
        risk_factors += 1 if high_freq_avg > 25 else 0
        risk_factors += 1 if speech_freq_avg > 20 else 0
        
        # Display results
        st.subheader("Analysis Results")
        
        result_cols = st.columns(2)
        with result_cols[0]:
            if risk_factors >= 3:
                st.error("⚠️ Higher Risk of Hearing Loss Detected")
                risk_status = "High Risk"
            elif risk_factors >= 1:
                st.warning("⚠️ Moderate Risk of Hearing Loss")
                risk_status = "Moderate Risk"
            else:
                st.success("✅ Low Risk - Normal Hearing Profile")
                risk_status = "Low Risk"
                
            st.metric("Risk Level", f"{risk_factors}/5")
        
        with result_cols[1]:
            # Create audiogram visualization using seaborn
            fig = sns.Figure(figsize=(8, 4))
            
            # Create a DataFrame for plotting
            frequencies = [250, 500, 1000, 2000, 4000, 8000]
            thresholds = [f250, f500, f1k, f2k, f4k, f8k]
            audiogram_df = pd.DataFrame({
                'Frequency': frequencies,
                'Threshold': thresholds
            })
            
            # Create plot
            ax = fig.subplots()
            sns.lineplot(data=audiogram_df, x='Frequency', y='Threshold', marker='o', 
                          linewidth=2, markersize=8, ax=ax)
            
            # Set x-axis to log scale and customize
            ax.set_xscale('log')
            ax.set_xticks(frequencies)
            ax.set_xticklabels([str(f) for f in frequencies])
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Hearing Level (dB)')
            ax.set_title('Your Audiogram')
            ax.grid(True)
            ax.invert_yaxis()
            
            # Add hearing ranges
            ax.axhspan(0, 25, color='green', alpha=0.2, label='Normal')
            ax.axhspan(25, 40, color='yellow', alpha=0.2, label='Mild Loss')
            ax.axhspan(40, 70, color='orange', alpha=0.2, label='Moderate Loss')
            ax.axhspan(70, 100, color='red', alpha=0.2, label='Severe Loss')
            
            ax.legend()
            st.pyplot(fig)
        
        # Risk factors explanation
        st.subheader("Risk Factor Analysis")
        risk_table = pd.DataFrame({
            'Risk Factor': [
                'Age Factor', 
                'Earphone Usage', 
                'Volume Level', 
                'High Frequency Hearing',
                'Speech Frequency Hearing'
            ],
            'Status': [
                '⚠️ Risk' if age > 50 else '✅ Normal',
                '⚠️ Risk' if usage_hours > 4 and usage_years > 5 else '✅ Normal',
                '⚠️ Risk' if volume_level > 7 else '✅ Normal',
                '⚠️ Risk' if high_freq_avg > 25 else '✅ Normal',
                '⚠️ Risk' if speech_freq_avg > 20 else '✅ Normal'
            ],
            'Details': [
                f'Age {age} years' + (' (increased risk over 50)' if age > 50 else ''),
                f'{usage_hours}h daily for {usage_years} years' + (' (extended exposure)' if usage_hours > 4 and usage_years > 5 else ''),
                f'Volume level {volume_level}/10' + (' (high volume)' if volume_level > 7 else ''),
                f'High freq avg: {high_freq_avg:.1f} dB' + (' (elevated)' if high_freq_avg > 25 else ''),
                f'Speech freq avg: {speech_freq_avg:.1f} dB' + (' (elevated)' if speech_freq_avg > 20 else '')
            ]
        })
        
        st.table(risk_table)
        
        # Recommendations based on risk level
        st.subheader("Recommendations")
        if risk_factors >= 3:
            st.markdown("""
            1. **Consult an audiologist** for a complete hearing evaluation
            2. Reduce daily earphone usage and volume level
            3. Use noise-canceling over-ear headphones instead of in-ear models
            4. Take regular breaks when using earphones (60/60 rule - 60 minutes use, 60 minutes break)
            5. Consider using volume-limiting earphones
            """)
        elif risk_factors >= 1:
            st.markdown("""
            1. **Monitor your hearing health** with regular checkups
            2. Consider reducing volume levels when using earphones
            3. Take breaks during extended listening sessions
            4. Use the 60/60 rule (60% volume for maximum 60 minutes)
            """)
        else:
            st.markdown("""
            1. **Continue good hearing practices**
            2. Maintain moderate volume levels
            3. Consider periodic hearing assessments as a preventive measure
            """)
        
        # Disclaimer
        st.info("**Disclaimer**: This tool provides an estimate only and is not a substitute for professional medical diagnosis. If you're concerned about your hearing, please consult an audiologist or healthcare provider.")

with tab2:
    st.header("Generate Synthetic Dataset")
    st.markdown("""
    This section generates a synthetic dataset for training the hearing loss prediction model.
    The synthetic data simulates audiometry results across different age groups, earphone usage patterns,
    and hearing health conditions.
    """)
    
    # Dataset generation parameters
    st.subheader("Dataset Parameters")
    
    # Two columns for parameters
    param_col1, param_col2 = st.columns(2)
    
    with param_col1:
        n_samples = st.number_input("Number of samples", min_value=100, max_value=10000, value=1000)
        age_min = st.number_input("Minimum age", min_value=5, max_value=50, value=10)
        age_max = st.number_input("Maximum age", min_value=30, max_value=100, value=80)
        
    with param_col2:
        noise_level = st.slider("Data noise level (randomness)", 0.1, 1.0, 0.3)
        positive_ratio = st.slider("Ratio of hearing loss cases", 0.1, 0.5, 0.3)
    
    # Generate dataset button
    generate_clicked = st.button("Generate Dataset", type="primary")
    
    if generate_clicked:
        with st.spinner("Generating synthetic dataset..."):
            # Create synthetic dataset
            np.random.seed(42)  # For reproducibility
            
            # Initialize empty dataframe
            data = {
                'age': np.random.randint(age_min, age_max, n_samples),
                'gender': np.random.randint(0, 2, n_samples),
                'earphone_type': np.random.choice(['None', 'In-ear', 'Over-ear'], n_samples),
                'usage_hours': np.zeros(n_samples),
                'usage_years': np.zeros(n_samples),
                'volume_level': np.zeros(n_samples),
                'f250': np.zeros(n_samples),
                'f500': np.zeros(n_samples),
                'f1k': np.zeros(n_samples),
                'f2k': np.zeros(n_samples),
                'f4k': np.zeros(n_samples),
                'f8k': np.zeros(n_samples),
                'hearing_loss': np.zeros(n_samples, dtype=int)
            }
            
            # Set values based on relationships and add controlled noise
            for i in range(n_samples):
                # People with no earphones
                if data['earphone_type'][i] == 'None':
                    data['usage_hours'][i] = 0
                    data['usage_years'][i] = 0
                    data['volume_level'][i] = 0
                else:
                    # Earphone users
                    data['usage_hours'][i] = np.random.lognormal(1, 0.6) + noise_level * np.random.randn()
                    data['usage_hours'][i] = max(0, min(12, data['usage_hours'][i]))
                    
                    data['usage_years'][i] = np.random.lognormal(1.5, 0.8) + noise_level * np.random.randn()
                    data['usage_years'][i] = max(0, min(20, data['usage_years'][i]))
                    
                    data['volume_level'][i] = np.random.normal(6, 2) + noise_level * np.random.randn()
                    data['volume_level'][i] = max(1, min(10, data['volume_level'][i]))
                
                # Base hearing thresholds related to age
                age_factor = (data['age'][i] - age_min) / (age_max - age_min)
                base_threshold = 5 + 20 * age_factor
                
                # Add effects from earphone usage
                usage_effect = 0
                if data['earphone_type'][i] != 'None':
                    # In-ear headphones have more effect than over-ear
                    type_multiplier = 1.5 if data['earphone_type'][i] == 'In-ear' else 1.0
                    
                    # Calculate effect based on usage patterns and volume
                    usage_intensity = (data['usage_hours'][i] / 12) * (data['usage_years'][i] / 20) * (data['volume_level'][i] / 10)
                    usage_effect = 25 * usage_intensity * type_multiplier
                
                # Set frequency-specific thresholds (normal hearing is 0-25 dB)
                # Lower frequencies (250, 500, 1000 Hz) are affected less by noise exposure
                data['f250'][i] = base_threshold + 0.2 * usage_effect + noise_level * np.random.randn() * 5
                data['f500'][i] = base_threshold + 0.3 * usage_effect + noise_level * np.random.randn() * 5
                data['f1k'][i] = base_threshold + 0.5 * usage_effect + noise_level * np.random.randn() * 5
                
                # Higher frequencies (2000, 4000, 8000 Hz) are affected more by noise exposure
                data['f2k'][i] = base_threshold + 0.7 * usage_effect + noise_level * np.random.randn() * 5
                data['f4k'][i] = base_threshold + usage_effect + noise_level * np.random.randn() * 5
                data['f8k'][i] = base_threshold + 1.2 * usage_effect + noise_level * np.random.randn() * 5
                
                # Ensure all values are positive and reasonable
                for freq in ['f250', 'f500', 'f1k', 'f2k', 'f4k', 'f8k']:
                    data[freq][i] = max(0, min(100, data[freq][i]))
            
            # Define hearing loss based on WHO criteria
            # Hearing loss if average of 500, 1k, 2k, 4k Hz > 25 dB
            for i in range(n_samples):
                avg_threshold = (data['f500'][i] + data['f1k'][i] + data['f2k'][i] + data['f4k'][i]) / 4
                
                # Also consider worse high-frequency loss at 4k and 8k
                high_freq_avg = (data['f4k'][i] + data['f8k'][i]) / 2
                
                # Set hearing loss status
                if avg_threshold > 25 or high_freq_avg > 30:
                    data['hearing_loss'][i] = 1
            
            # Adjust to target positive ratio
            current_positive = data['hearing_loss'].mean()
            if current_positive != positive_ratio:
                needed_positives = int(n_samples * positive_ratio)
                current_positives = int(current_positive * n_samples)
                
                if current_positives < needed_positives:
                    # Need more positive cases
                    indices = np.where(data['hearing_loss'] == 0)[0]
                    to_convert = np.random.choice(indices, needed_positives - current_positives, replace=False)
                    
                    for idx in to_convert:
                        data['hearing_loss'][idx] = 1
                        # Make thresholds worse for these converted cases
                        for freq in ['f500', 'f1k', 'f2k', 'f4k', 'f8k']:
                            data[freq][idx] += np.random.uniform(5, 15)
                            data[freq][idx] = min(100, data[freq][idx])
                
                elif current_positives > needed_positives:
                    # Need fewer positive cases
                    indices = np.where(data['hearing_loss'] == 1)[0]
                    to_convert = np.random.choice(indices, current_positives - needed_positives, replace=False)
                    
                    for idx in to_convert:
                        data['hearing_loss'][idx] = 0
                        # Make thresholds better for these converted cases
                        for freq in ['f500', 'f1k', 'f2k', 'f4k', 'f8k']:
                            data[freq][idx] -= np.random.uniform(5, 15)
                            data[freq][idx] = max(0, data[freq][idx])
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
        
        # Display sample of the dataset
        st.subheader("Generated Dataset Preview")
        st.dataframe(df.head(10))
        
        # Display dataset statistics
        st.subheader("Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", n_samples)
        
        with col2:
            hearing_loss_count = df['hearing_loss'].sum()
            st.metric("Hearing Loss Cases", f"{hearing_loss_count} ({hearing_loss_count/n_samples:.1%})")
        
        with col3:
            st.metric("Normal Hearing Cases", f"{n_samples - hearing_loss_count} ({1 - hearing_loss_count/n_samples:.1%})")
        
        # Save dataset button
        if st.button("Save Dataset to CSV"):
            df.to_csv("hearing_dataset.csv", index=False)
            
            # Create separate files for X and y
            X = df.drop('hearing_loss', axis=1)
            y = df['hearing_loss']
            
            X.to_csv("X_synthetic.csv", index=False)
            y.to_csv("y_synthetic.csv", index=False)
            
            st.success("Dataset saved to 'hearing_dataset.csv', 'X_synthetic.csv', and 'y_synthetic.csv'!")
        
        # Visualizations
        st.subheader("Data Visualizations")
        
        viz_tabs = st.tabs(["Distribution", "Correlation", "Age Groups", "Audiograms"])
        
        with viz_tabs[0]:
            # Create distribution plots using seaborn
            fig = sns.Figure(figsize=(10, 12))
            axs = fig.subplots(3, 2)
            
            # Age distribution
            sns.histplot(data=df, x='age', hue='hearing_loss', multiple='stack', ax=axs[0, 0])
            axs[0, 0].set_title('Age Distribution by Hearing Status')
            
            # Earphone type
            sns.countplot(data=df, x='earphone_type', hue='hearing_loss', ax=axs[0, 1])
            axs[0, 1].set_title('Earphone Type by Hearing Status')
            
            # Usage hours
            sns.histplot(data=df, x='usage_hours', hue='hearing_loss', multiple='stack', ax=axs[1, 0])
            axs[1, 0].set_title('Daily Usage Hours by Hearing Status')
            
            # Volume level
            sns.histplot(data=df, x='volume_level', hue='hearing_loss', multiple='stack', ax=axs[1, 1])
            axs[1, 1].set_title('Volume Level by Hearing Status')
            
            # Frequency thresholds
            freq_df = df.melt(id_vars='hearing_loss', 
                            value_vars=['f250', 'f500', 'f1k', 'f2k', 'f4k', 'f8k'],
                            var_name='frequency', value_name='threshold')
            sns.boxplot(data=freq_df, x='frequency', y='threshold', hue='hearing_loss', ax=axs[2, 0])
            axs[2, 0].set_title('Hearing Thresholds by Frequency and Status')
            
            # 4kHz distribution
            sns.histplot(data=df, x='f4k', hue='hearing_loss', multiple='stack', ax=axs[2, 1])
            axs[2, 1].set_title('4kHz Threshold Distribution by Hearing Status')
            
            fig.tight_layout()
            st.pyplot(fig)
            
        with viz_tabs[1]:
            # Correlation matrix using seaborn
            corr_matrix = df.corr()
            
            fig = sns.Figure(figsize=(10, 8))
            ax = fig.subplots()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            ax.set_title('Correlation Matrix')
            st.pyplot(fig)
            
        with viz_tabs[2]:
            # Group by age
            age_bins = [0, 20, 35, 50, 65, 100]
            age_labels = ['<20', '20-35', '35-50', '50-65', '65+']
            df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
            
            fig = sns.Figure(figsize=(12, 5))
            axs = fig.subplots(1, 2)
            
            # Hearing loss by age group
            age_loss_data = df.groupby('age_group')['hearing_loss'].mean().reset_index()
            sns.barplot(data=age_loss_data, x='age_group', y='hearing_loss', ax=axs[0], color='skyblue')
            axs[0].set_title('Hearing Loss Probability by Age Group')
            axs[0].set_ylabel('Probability')
            
            # Average thresholds by age - prepare data for seaborn
            age_thresholds = df.groupby('age_group')[['f250', 'f500', 'f1k', 'f2k', 'f4k', 'f8k']].mean()
            age_threshold_data = age_thresholds.stack().reset_index()
            age_threshold_data.columns = ['age_group', 'frequency', 'threshold']
            
            # Map frequency codes to Hz values for better readability
            freq_map = {'f250': '250Hz', 'f500': '500Hz', 'f1k': '1kHz', 'f2k': '2kHz', 'f4k': '4kHz', 'f8k': '8kHz'}
            age_threshold_data['frequency'] = age_threshold_data['frequency'].map(freq_map)
            
            # Plot with seaborn
            sns.lineplot(data=age_threshold_data, x='frequency', y='threshold', hue='age_group', 
                         marker='o', ax=axs[1])
            axs[1].set_title('Average Hearing Thresholds by Age Group')
            axs[1].set_xlabel('Frequency')
            axs[1].set_ylabel('Threshold (dB)')
            
            fig.tight_layout()
            st.pyplot(fig)
            
        with viz_tabs[3]:
            # Sample audiograms by hearing status using seaborn
            fig = sns.Figure(figsize=(14, 6))
            axs = fig.subplots(1, 2)
            
            # Get random sample of each group
            normal_samples = df[df['hearing_loss'] == 0].sample(min(10, (df['hearing_loss'] == 0).sum()))
            loss_samples = df[df['hearing_loss'] == 1].sample(min(10, (df['hearing_loss'] == 1).sum()))
            
            frequencies = [250, 500, 1000, 2000, 4000, 8000]
            freq_cols = ['f250', 'f500', 'f1k', 'f2k', 'f4k', 'f8k']
            
            # Create long format data for seaborn
            normal_data = []
            for i, row in normal_samples.iterrows():
                for j, freq in enumerate(freq_cols):
                    normal_data.append({
                        'Frequency': frequencies[j],
                        'Threshold': row[freq],
                        'Sample': i
                    })
            normal_df = pd.DataFrame(normal_data)
            
            loss_data = []
            for i, row in loss_samples.iterrows():
                for j, freq in enumerate(freq_cols):
                    loss_data.append({
                        'Frequency': frequencies[j],
                        'Threshold': row[freq],
                        'Sample': i
                    })
            loss_df = pd.DataFrame(loss_data)
            
            # Plot normal hearing
            sns.lineplot(data=normal_df, x='Frequency', y='Threshold', hue='Sample', 
                          ax=axs[0], alpha=0.5, legend=False)
            
            axs[0].set_title('Sample Audiograms - Normal Hearing')
            axs[0].set_xscale('log')
            axs[0].set_xticks(frequencies)
            axs[0].set_xticklabels([str(f) for f in frequencies])
            axs[0].set_xlabel('Frequency (Hz)')
            axs[0].set_ylabel('Hearing Level (dB)')
            axs[0].grid(True)
            axs[0].invert_yaxis()
            
            # Add normal hearing range
            axs[0].axhspan(0, 25, color='green', alpha=0.2)
            
            # Plot hearing loss
            sns.lineplot(data=loss_df, x='Frequency', y='Threshold', hue='Sample', 
                          ax=axs[1], alpha=0.5, legend=False)
            
            axs[1].set_title('Sample Audiograms - Hearing Loss')
            axs[1].set_xscale('log')
            axs[1].set_xticks(frequencies)
            axs[1].set_xticklabels([str(f) for f in frequencies])
            axs[1].set_xlabel('Frequency (Hz)')
            axs[1].set_ylabel('Hearing Level (dB)')
            axs[1].grid(True)
            axs[1].invert_yaxis()
            
            # Add hearing range zones
            axs[1].axhspan(0, 25, color='green', alpha=0.2, label='Normal')
            axs[1].axhspan(25, 40, color='yellow', alpha=0.2, label='Mild Loss')
            axs[1].axhspan(40, 70, color='orange', alpha=0.2, label='Moderate Loss')
            axs[1].axhspan(70, 100, color='red', alpha=0.2, label='Severe Loss')
            
            # Add legend
            handles, labels = axs[1].get_legend_handles_labels()
            axs[1].legend(handles[:4], labels[:4], title='Hearing Range')
            
            fig.tight_layout()
            st.pyplot(fig)

with tab3:
    st.header("Model Training")
    st.markdown("""
    This section allows you to train a machine learning model to predict hearing loss risk.
    Upload your dataset or use the synthetic data generated in the previous section.
    """)
    
    # Check if dataset exists
    dataset_exists = os.path.exists("X_synthetic.csv") and os.path.exists("y_synthetic.csv")
    
    if not dataset_exists:
        st.warning("No dataset found. Please generate or upload a dataset first.")
        
        # File upload option
        uploaded_X = st.file_uploader("Upload features CSV (X_synthetic.csv)", type="csv")
        uploaded_y = st.file_uploader("Upload target CSV (y_synthetic.csv)", type="csv")
        
        if uploaded_X is not None and uploaded_y is not None:
            X = pd.read_csv(uploaded_X)
            y = pd.read_csv(uploaded_y).values.ravel()
            dataset_exists = True
            st.success("Dataset loaded successfully!")
    else:
        # Load existing dataset
        X = pd.read_csv("X_synthetic.csv")
        y = pd.read_csv("y_synthetic.csv").values.ravel()
        st.success(f"Found existing dataset with {len(X)} samples.")
    
    if dataset_exists:
        # Model parameters
        st.subheader("Model Parameters")
        
        model_type = st.selectbox("Model Type", ["Random Forest", "XGBoost", "Logistic Regression"])
        
        # Parameters based on model type
        if model_type == "Random Forest":
            n_estimators = st.slider("Number of trees", 10, 500, 100)
            max_depth = st.slider("Maximum tree depth", 2, 30, 10)
            min_samples_split = st.slider("Minimum samples to split", 2, 20, 5)
            class_weight = st.selectbox("Class weights", ["balanced", "None"])
            
            # Convert class_weight to None if selected
            if class_weight == "None":
                class_weight = None
        
        elif model_type == "XGBoost":
            try:
                import xgboost as xgb
                n_estimators = st.slider("Number of trees", 10, 500, 100)
                learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1, 0.01)
                max_depth = st.slider("Maximum tree depth", 2, 15, 6)
                subsample = st.slider("Subsample ratio", 0.5, 1.0, 0.8, 0.1)
            except ImportError:
                st.error("XGBoost is not installed. Please install it to use this model type.")
                st.stop()
                
        elif model_type == "Logistic Regression":
            C = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0, 0.01)
            penalty = st.selectbox("Penalty", ["l2", "l1", "elasticnet", "none"])
            solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
            
            # Check solver/penalty compatibility
            if penalty == "elasticnet" and solver != "saga":
                st.warning("Elasticnet penalty requires saga solver. Switching solver to saga.")
                solver = "saga"
            elif penalty == "l1" and solver == "lbfgs":
                st.warning("L1 penalty is not supported with lbfgs solver. Switching solver to liblinear.")
                solver = "liblinear"
            elif penalty == "none" and solver == "liblinear":
                st.warning("No penalty is not supported with liblinear solver. Switching solver to lbfgs.")
                solver = "lbfgs"
                
        # Data preprocessing options
        st.subheader("Data Preprocessing")
        use_smote = st.checkbox("Apply SMOTE for class imbalance", True)
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
        
        # Features selection
        st.subheader("Feature Selection")
        st.markdown("Select features to include in the model:")
        
        feature_cols = st.columns(3)
        with feature_cols[0]:
            use_personal = st.checkbox("Personal factors", True)
            if use_personal:
                use_age = st.checkbox("Age", True)
                use_gender = st.checkbox("Gender", True)
        
        with feature_cols[1]:
            use_earphone = st.checkbox("Earphone usage", True)
            if use_earphone:
                use_type = st.checkbox("Earphone type", True)
                use_hours = st.checkbox("Usage hours", True)
                use_years = st.checkbox("Usage years", True)
                use_volume = st.checkbox("Volume level", True)
        
        with feature_cols[2]:
            use_audio = st.checkbox("Audiometry results", True)
            if use_audio:
                use_low_freq = st.checkbox("Low frequencies (250-500Hz)", True)
                use_mid_freq = st.checkbox("Mid frequencies (1k-2kHz)", True)
                use_high_freq = st.checkbox("High frequencies (4k-8kHz)", True)
        
        # Training button
        train_button = st.button("Train Model", type="primary")
        
        if train_button:
            # Feature selection
            selected_features = []
            
            if use_personal:
                if use_age:
                    selected_features.append('age')
                if use_gender:
                    selected_features.append('gender')
            
            if use_earphone:
                if use_type:
                    selected_features.append('earphone_type')
                if use_hours:
                    selected_features.append('usage_hours')
                if use_years:
                    selected_features.append('usage_years')
                if use_volume:
                    selected_features.append('volume_level')
            
            if use_audio:
                if use_low_freq:
                    selected_features.extend(['f250', 'f500'])
                if use_mid_freq:
                    selected_features.extend(['f1k', 'f2k'])
                if use_high_freq:
                    selected_features.extend(['f4k', 'f8k'])
            
            if not selected_features:
                st.error("Please select at least one feature.")
                st.stop()
                
            with st.spinner("Training model..."):
                # Select features
                X_selected = X[selected_features].copy()
                
                # Split dataset
                X_train, X_test, y_train, y_test = train_test_split(
                    X_selected, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Create preprocessing pipeline
                categorical_features = ['earphone_type'] if 'earphone_type' in selected_features else []
                numerical_features = [f for f in selected_features if f != 'earphone_type']
                
                transformers = []
                
                if categorical_features:
                    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
                    transformers.append(('cat', categorical_transformer, categorical_features))
                
                if numerical_features:
                    numerical_transformer = StandardScaler()
                    transformers.append(('num', numerical_transformer, numerical_features))
                
                preprocessor = ColumnTransformer(transformers)
                
                # Apply SMOTE if selected
                if use_smote:
                    smote = SMOTE(random_state=42)
                    X_train_prep = preprocessor.fit_transform(X_train)
                    X_train_prep, y_train = smote.fit_resample(X_train_prep, y_train)
                    
                    # Create appropriate model based on selection
                    if model_type == "Random Forest":
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            class_weight=class_weight,
                            random_state=42
                        )
                    elif model_type == "XGBoost":
                        model = xgb.XGBClassifier(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            subsample=subsample,
                            random_state=42
                        )
                    else:  # Logistic Regression
                        model = LogisticRegression(
                            C=C,
                            penalty=penalty,
                            solver=solver,
                            random_state=42,
                            max_iter=1000
                        )
                    
                    # Train model
                    model.fit(X_train_prep, y_train)
                    
                    # Transform test data
                    X_test_prep = preprocessor.transform(X_test)
                    
                else:
                    # Create pipeline without SMOTE
                    if model_type == "Random Forest":
                        clf = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            class_weight=class_weight,
                            random_state=42
                        )
                    elif model_type == "XGBoost":
                        clf = xgb.XGBClassifier(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            subsample=subsample,
                            random_state=42
                        )
                    else:  # Logistic Regression
                        clf = LogisticRegression(
                            C=C,
                            penalty=penalty,
                            solver=solver,
                            random_state=42,
                            max_iter=1000
                        )
                    
                    # Create and train pipeline
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', clf)
                    ])
                    
                    pipeline.fit(X_train, y_train)
                    model = pipeline  # For consistency in the code
                
                # Evaluate model
                if use_smote:
                    y_pred = model.predict(X_test_prep)
                    y_prob = model.predict_proba(X_test_prep)[:, 1]
                else:
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = np.mean(y_pred == y_test)
                report = classification_report(y_test, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                # Show results
                st.subheader("Model Performance")
                st.metric("Accuracy", f"{accuracy:.2%}")
                
                # Create columns for metrics
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Precision", f"{report['1']['precision']:.2%}")
                with metric_cols[1]:
                    st.metric("Recall", f"{report['1']['recall']:.2%}")
                with metric_cols[2]:
                    st.metric("F1 Score", f"{report['1']['f1-score']:.2%}")
                with metric_cols[3]:
                    st.metric("AUC", f"{roc_auc:.2%}")
                
                # Display confusion matrix
                st.subheader("Confusion Matrix")
                fig = sns.Figure(figsize=(6, 5))
                ax = fig.subplots()
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                           xticklabels=['Normal', 'Hearing Loss'],
                           yticklabels=['Normal', 'Hearing Loss'],
                           ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
                
                # Display ROC curve
                st.subheader("ROC Curve")
                fig = sns.Figure(figsize=(6, 5))
                ax = fig.subplots()
                ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax.legend(loc='lower right')
                st.pyplot(fig)
                
                # Display feature importance if applicable
                if model_type in ["Random Forest", "XGBoost"]:
                    st.subheader("Feature Importance")
                    
                    # Extract feature importances
                    if use_smote:
                        importances = model.feature_importances_
                        feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"Feature {i}" for i in range(X_train.shape[1])]
                    else:
                        # For pipeline, get feature names from preprocessor
                        feature_names = []
                        importances = model.named_steps['classifier'].feature_importances_
                        
                        # Get feature names after preprocessing
                        if categorical_features:
                            # Get transformed categorical feature names
                            cat_encoder = model.named_steps['preprocessor'].transformers_[0][1]
                            cat_feature_names = [f"{col}_{val}" for col in categorical_features
                                              for val in cat_encoder.categories_[0]]
                            feature_names.extend(cat_feature_names)
                        
                        # Add numerical features
                        feature_names.extend(numerical_features)
                    
                    # Create importance dataframe
                    importance_df = pd.DataFrame({
                        'Feature': feature_names[:len(importances)],
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    # Plot feature importance
                    fig = sns.Figure(figsize=(10, 6))
                    ax = fig.subplots()
                    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
                    ax.set_title('Feature Importance')
                    st.pyplot(fig)
                
                # Save model button
                if st.button("Save Model"):
                    if use_smote:
                        # Save model and preprocessor separately
                        joblib.dump(model, "hearing_loss_model.pkl")
                        joblib.dump(preprocessor, "preprocessor.pkl")
                    else:
                        # Save the whole pipeline
                        joblib.dump(model, "hearing_loss_pipeline.pkl")
                    
                    st.success("Model saved successfully!")

with tab4:
    st.header("About")
    st.markdown("""
    ## Hearing Loss Risk Assessment Tool
    
    This application is designed to help users assess their risk of hearing loss based on various factors.
    It uses machine learning algorithms to analyze audiometry data along with personal factors and earphone usage patterns.
    
    ### Key Features:
    
    - **Prediction**: Assess hearing loss risk using audiometry results and personal factors
    - **Data Generation**: Create synthetic datasets for research and model training
    - **Model Training**: Train and evaluate machine learning models for hearing loss prediction
    
    ### How to Use:
    
    1. Go to the **Prediction** tab to assess your hearing risk
    2. Input your audiometry results (hearing thresholds) and personal information
    3. View your risk assessment and recommendations
    
    ### Medical Disclaimer:
    
    This tool is for educational and informational purposes only. It is not a substitute for professional medical advice,
    diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any
    questions you may have regarding a medical condition.
    
    ### Resources:
    
    - [World Health Organization - Hearing Loss](https://www.who.int/health-topics/hearing-loss)
    - [National Institute on Deafness and Other Communication Disorders](https://www.nidcd.nih.gov/health/hearing-loss)
    - [American Speech-Language-Hearing Association](https://www.asha.org/)
    """)
    
    # Credits
    st.subheader("Credits")
    st.markdown("""
    - **Created by**: [Your Name/Organization]
    - **Version**: 1.0.0
    - **Last Updated**: May 2025
    
    For questions or support, please contact: [your@email.com]
    """)
