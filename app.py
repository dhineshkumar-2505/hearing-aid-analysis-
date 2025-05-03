import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        
        # For actual prediction, would use a trained model
        # Here we'll implement a simple rule-based system for demo
        # In a real app, replace this with: prediction = model.predict(pd.DataFrame([input_data]))[0]
        
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
            # Create audiogram visualization
            fig, ax = plt.subplots(figsize=(8, 4))
            frequencies = [250, 500, 1000, 2000, 4000, 8000]
            thresholds = [f250, f500, f1k, f2k, f4k, f8k]
            
            ax.plot(frequencies, thresholds, 'b-o', linewidth=2, markersize=8)
            ax.set_xscale('log')
            ax.set_xticks(frequencies)
            ax.set_xticklabels([str(f) for f in frequencies])
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Hearing Level (dB)')
            ax.set_title('Your Audiogram')
            ax.grid(True)
            ax.invert_yaxis()
            
            # Add normal hearing range
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
            # Create distribution plots
            fig, axs = plt.subplots(3, 2, figsize=(10, 12))
            
            sns.histplot(data=df, x='age', hue='hearing_loss', multiple='stack', ax=axs[0, 0])
            axs[0, 0].set_title('Age Distribution by Hearing Status')
            
            sns.countplot(data=df, x='earphone_type', hue='hearing_loss', ax=axs[0, 1])
            axs[0, 1].set_title('Earphone Type by Hearing Status')
            
            sns.histplot(data=df, x='usage_hours', hue='hearing_loss', multiple='stack', ax=axs[1, 0])
            axs[1, 0].set_title('Daily Usage Hours by Hearing Status')
            
            sns.histplot(data=df, x='volume_level', hue='hearing_loss', multiple='stack', ax=axs[1, 1])
            axs[1, 1].set_title('Volume Level by Hearing Status')
            
            # Frequency thresholds
            sns.boxplot(data=df.melt(id_vars='hearing_loss', 
                                     value_vars=['f250', 'f500', 'f1k', 'f2k', 'f4k', 'f8k'],
                                     var_name='frequency', value_name='threshold'),
                        x='frequency', y='threshold', hue='hearing_loss', ax=axs[2, 0])
            axs[2, 0].set_title('Hearing Thresholds by Frequency and Status')
            
            sns.histplot(data=df, x='f4k', hue='hearing_loss', multiple='stack', ax=axs[2, 1])
            axs[2, 1].set_title('4kHz Threshold Distribution by Hearing Status')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        with viz_tabs[1]:
            # Correlation matrix
            corr_matrix = df.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            ax.set_title('Correlation Matrix')
            st.pyplot(fig)
            
        with viz_tabs[2]:
            # Group by age
            age_bins = [0, 20, 35, 50, 65, 100]
            age_labels = ['<20', '20-35', '35-50', '50-65', '65+']
            df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
            
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            
            # Hearing loss by age group
            hearing_by_age = df.groupby('age_group')['hearing_loss'].mean()
            hearing_by_age.plot(kind='bar', ax=axs[0], color='skyblue')
            axs[0].set_title('Hearing Loss Probability by Age Group')
            axs[0].set_ylabel('Probability')
            
            # Average thresholds by age
            age_thresholds = df.groupby('age_group')[['f250', 'f500', 'f1k', 'f2k', 'f4k', 'f8k']].mean()
            age_thresholds.T.plot(ax=axs[1], marker='o')
            axs[1].set_title('Average Hearing Thresholds by Age Group')
            axs[1].set_xlabel('Frequency')
            axs[1].set_ylabel('Threshold (dB)')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        with viz_tabs[3]:
            # Sample audiograms by hearing status
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            
            # Get random sample of each group
            normal_samples = df[df['hearing_loss'] == 0].sample(min(10, (df['hearing_loss'] == 0).sum()))
            loss_samples = df[df['hearing_loss'] == 1].sample(min(10, (df['hearing_loss'] == 1).sum()))
            
            frequencies = [250, 500, 1000, 2000, 4000, 8000]
            freq_cols = ['f250', 'f500', 'f1k', 'f2k', 'f4k', 'f8k']
            
            # Plot normal hearing
            for i, row in normal_samples.iterrows():
                thresholds = [row[col] for col in freq_cols]
                axs[0].plot(frequencies, thresholds, alpha=0.5, marker='o')
            
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
            for i, row in loss_samples.iterrows():
                thresholds = [row[col] for col in freq_cols]
                axs[1].plot(frequencies, thresholds, alpha=0.5, marker='o')
            
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
            
            axs[1].legend()
            
            plt.tight_layout()
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
            min_samples_split = st.slider("Minimum samples to split", 2, 20, 2)
        
        # SMOTE oversampling option
        use_smote = st.checkbox("Use SMOTE to handle class imbalance", value=True)
        
        # Feature selection
        st.subheader("Feature Selection")
        available_features = X.columns.tolist()
        selected_features = st.multiselect("Select features to include", available_features, default=available_features)
        
        if len(selected_features) == 0:
            st.warning("Please select at least one feature.")
        else:
            # Train model button
            train_clicked = st.button("Train Model", type="primary")
            
            if train_clicked:
                with st.spinner("Training model..."):
                    # Select features
                    X_selected = X[selected_features]
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_selected, y, stratify=y, test_size=0.2, random_state=42
                    )
                    
                    # Identify numeric and categorical features
                    numeric_features = [col for col in X_selected.columns if X_selected[col].dtype in [np.int64, np.float64] 
                                        and col != 'earphone_type']
                    categorical_features = [col for col in X_selected.columns if col == 'earphone_type']
                    
                    # Create preprocessor
                    transformers = []
                    if numeric_features:
                        transformers.append(("num", StandardScaler(), numeric_features))
                    if categorical_features:
                        transformers.append(("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features))
                    
                    preprocessor = ColumnTransformer(transformers)
                    
                    # Create classifier based on selection
                    if model_type == "Random Forest":
                        classifier = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            random_state=42
                        )
                    elif model_type == "XGBoost":
                        import xgboost as xgb
                        classifier = xgb.XGBClassifier(
                            random_state=42,
                            use_label_encoder=False,
                            eval_metric='logloss'
                        )
                    else:  # Logistic Regression
                        from sklearn.linear_model import LogisticRegression
                        classifier = LogisticRegression(random_state=42, max_iter=1000)
                    
                    # Create pipeline
                    if use_smote:
                        pipeline = Pipeline([
                            ('preprocessor', preprocessor),
                            ('smote', SMOTE(random_state=42)),
                            ('classifier', classifier)
                        ])
                    else:
                        pipeline = Pipeline([
                            ('preprocessor', preprocessor),
                            ('classifier', classifier)
                        ])
                    
                    # Train model
                    pipeline.fit(X_train, y_train)
                    
                    # Save model
                    joblib.dump(pipeline, "hearing_loss_model.pkl")
                    
                    # Evaluate model
                    y_pred = pipeline.predict(X_test)
                    y_prob = pipeline.predict_proba(X_test)[:, 1]
                    
                    # Cross-validation
                    cv_scores = cross_val_score(pipeline, X_selected, y, cv=5, scoring='roc_auc')
                    
                    # Display results
                    st.subheader("Model Evaluation")
                    
                    # Model metrics
                    metrics_col1, metrics_col2 = st.columns(2)
                    
                    with metrics_col1:
                        # Classification report
                        st.text("Classification Report:")
                        st.text(classification_report(y_test, y_pred))
                        
                    with metrics_col2:
                        # Confusion matrix
                        st.text("Confusion Matrix:")
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title('Confusion Matrix')
                        st.pyplot(fig)
                    
                    # ROC curve
                    fig, ax = plt.subplots(figsize=(8, 6))
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    
                    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('Receiver Operating Characteristic (ROC)')
                    ax.legend(loc="lower right")
                    
                    st.pyplot(fig)
                    
                    # Cross-validation results
                    st.subheader("Cross-Validation Results")
                    st.write(f"Mean ROC AUC: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
                    
                    # Feature importance
                    if model_type == "Random Forest" or model_type == "XGBoost":
                        st.subheader("Feature Importance")
                        
                        # Get feature importance from the model
                        if hasattr(pipeline['classifier'], 'feature_importances_'):
                            importances = pipeline['classifier'].feature_importances_
                            
                            # Get feature names after preprocessing
                            if categorical_features:
                                # Get the categorical feature names after one-hot encoding
                                cat_encoder = pipeline['preprocessor'].transformers_[1][1]
                                cat_features = cat_encoder.get_feature_names_out(categorical_features).tolist()
                                feature_names = numeric_features + cat_features
                            else:
                                feature_names = numeric_features
                            
                            # Create DataFrame for visualization
                            if len(importances) == len(feature_names):
                                importance_df = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Importance': importances
                                }).sort_values('Importance', ascending=False)
                                
                                # Plot feature importance
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                                ax.set_title('Feature Importance')
                                st.pyplot(fig)
                            else:
                                st.warning("Feature names length doesn't match importance values. Cannot display feature importance.")
                        else:
                            st.info("Feature importance not available for this model.")
                    
                    # Success message
                    st.success(f"Model trained successfully! Model saved as 'hearing_loss_model.pkl'")
                    
                    # Model insights
                    st.subheader("Model Insights")
                    
                    # Threshold analysis
                    threshold_range = np.arange(0.1, 0.9, 0.1)
                    thresholds_df = pd.DataFrame(columns=['Threshold', 'Precision', 'Recall', 'F1-Score'])
                    
                    for threshold in threshold_range:
                        y_pred_custom = (y_prob >= threshold).astype(int)
                        precision = np.sum((y_pred_custom == 1) & (y_test == 1)) / np.sum(y_pred_custom == 1) if np.sum(y_pred_custom == 1) > 0 else 0
                        recall = np.sum((y_pred_custom == 1) & (y_test == 1)) / np.sum(y_test == 1)
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        
                        thresholds_df.loc[len(thresholds_df)] = {
                            'Threshold': threshold,
                            'Precision': precision,
                            'Recall': recall, 
                            'F1-Score': f1
                        }
                    
                    # Plot threshold analysis
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.lineplot(data=thresholds_df.melt(id_vars='Threshold', var_name='Metric', value_name='Value'), 
                                x='Threshold', y='Value', hue='Metric', ax=ax)
                    ax.set_title('Metrics vs. Probability Threshold')
                    ax.set_xlabel('Classification Threshold')
                    ax.set_ylabel('Score')
                    st.pyplot(fig)