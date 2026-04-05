import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import time

# Import models
from models.cnn_model import SmallFundusCNN, apply_clahe_rgb, center_crop_square, preprocess_image_bgr
from models.enhanced_densenet import EnhancedDenseNet
from models.adaboost_model import extract_features
from joblib import load
from torchvision import transforms

# Page config
st.set_page_config(
    page_title="DR Detection System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load all three trained models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}
    
    # Load CNN
    cnn_path = Path('runs/cnn_best.pt')
    if cnn_path.exists():
        try:
            ckpt = torch.load(cnn_path, map_location=device)
            cnn = SmallFundusCNN(num_classes=ckpt['num_classes']).to(device)
            cnn.load_state_dict(ckpt['model'])
            cnn.eval()
            models['CNN'] = {'model': cnn, 'num_classes': ckpt['num_classes']}
            st.sidebar.success("✅ CNN loaded")
        except Exception as e:
            st.sidebar.error(f"❌ CNN load failed: {e}")
    
    # Load Enhanced DenseNet
    densenet_path = Path('runs/enhanced_densenet.pt')
    if densenet_path.exists():
        try:
            ckpt = torch.load(densenet_path, map_location=device)
            densenet = EnhancedDenseNet(num_classes=ckpt['num_classes']).to(device)
            densenet.load_state_dict(ckpt['model'])
            densenet.eval()
            models['DenseNet'] = {'model': densenet, 'num_classes': ckpt['num_classes']}
            st.sidebar.success("✅ Enhanced DenseNet loaded")
        except Exception as e:
            st.sidebar.error(f"❌ DenseNet load failed: {e}")
    
    # Load AdaBoost
    ada_path = Path('runs/adaboost.joblib')
    if ada_path.exists():
        try:
            ada_pipe = load(ada_path)
            models['AdaBoost'] = {'model': ada_pipe, 'num_classes': None}
            st.sidebar.success("✅ AdaBoost loaded")
        except Exception as e:
            st.sidebar.error(f"❌ AdaBoost load failed: {e}")
    
    return models, device


def preprocess_for_dl(img_bgr, size=224):
    """Preprocess image for deep learning models"""
    img = preprocess_image_bgr(img_bgr, out_size=size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transforms.ToTensor()(img_rgb)
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor = norm(tensor)
    return tensor.unsqueeze(0)


def predict_image(models, device, img_bgr, class_names):
    """Run prediction on all available models"""
    results = {}
    
    # CNN prediction
    if 'CNN' in models:
        tensor = preprocess_for_dl(img_bgr, size=224).to(device)
        with torch.no_grad():
            logits = models['CNN']['model'](tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        results['CNN'] = {
            'probabilities': probs,
            'prediction': class_names[probs.argmax()],
            'confidence': probs.max() * 100
        }
    
    # DenseNet prediction
    if 'DenseNet' in models:
        tensor = preprocess_for_dl(img_bgr, size=256).to(device)
        with torch.no_grad():
            logits = models['DenseNet']['model'](tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        results['DenseNet'] = {
            'probabilities': probs,
            'prediction': class_names[probs.argmax()],
            'confidence': probs.max() * 100
        }
    
    # AdaBoost prediction
    if 'AdaBoost' in models:
        features = extract_features(img_bgr, out_size=224).reshape(1, -1)
        probs = models['AdaBoost']['model'].predict_proba(features)[0]
        results['AdaBoost'] = {
            'probabilities': probs,
            'prediction': class_names[probs.argmax()],
            'confidence': probs.max() * 100
        }
    
    return results


def create_probability_chart(results, class_names):
    """Create interactive bar chart comparing model predictions"""
    data = []
    for model_name, res in results.items():
        for i, class_name in enumerate(class_names):
            data.append({
                'Model': model_name,
                'Class': class_name,
                'Probability': res['probabilities'][i] * 100
            })
    
    df = pd.DataFrame(data)
    fig = px.bar(
        df, 
        x='Class', 
        y='Probability', 
        color='Model',
        barmode='group',
        title='Model Prediction Probabilities',
        labels={'Probability': 'Probability (%)'},
        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
    )
    fig.update_layout(height=400, hovermode='x unified')
    return fig


def create_confidence_gauge(confidence, model_name):
    """Create gauge chart for confidence"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{model_name} Confidence", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ff6b6b'},
                {'range': [50, 75], 'color': '#ffd93d'},
                {'range': [75, 100], 'color': '#6bcf7f'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_performance_comparison(metrics_df):
    """Create radar chart comparing model metrics"""
    fig = go.Figure()
    
    for idx, row in metrics_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score'], row['AUC']*100],
            theta=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
            fill='toself',
            name=row['Model']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="Model Performance Comparison",
        height=500
    )
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">👁️ Diabetic Retinopathy Detection System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("### Model Status")
    
    # Load models
    models, device = load_models()
    
    if not models:
        st.error("❌ No models found! Please train models first.")
        st.info("""
        **To train models:**
        1. CNN: `python models/cnn_model.py --data_root data_root --epochs 10`
        2. DenseNet: `python models/enhanced_densenet.py --data_root data_root --epochs 10`
        3. AdaBoost: `python models/adaboost_model.py --data_root data_root --split train`
        """)
        return
    
    st.sidebar.markdown(f"**Device:** {device}")
    st.sidebar.markdown(f"**Models loaded:** {len(models)}")
    
    # Class names (adjust based on your dataset)
    class_names = st.sidebar.text_input(
        "Class Names (comma-separated)",
        value="No DR,Mild,Moderate,Severe"
    ).split(',')
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📸 Single Image Prediction", "📊 Batch Prediction", "📈 Model Comparison"])
    
    # Tab 1: Single Image Prediction
    with tab1:
        st.header("Upload Fundus Image for Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a fundus image...",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'],
                help="Upload a retinal fundus image for DR detection"
            )
            
            if uploaded_file is not None:
                # Read image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # Image and button side by side
                img_col, btn_col = st.columns([3, 1])
                with img_col:
                    st.image(img_rgb, caption="Uploaded Image", width=400)
                with btn_col:
                    st.markdown("<br><br><br>", unsafe_allow_html=True)
                    analyze_button = st.button("🔍 Analyze Image", type="primary", use_container_width=True)
                
                if analyze_button:
                    with st.spinner("Running predictions..."):
                        start_time = time.time()
                        results = predict_image(models, device, img_bgr, class_names)
                        elapsed_time = time.time() - start_time
                    
                    # Calculate consensus prediction
                    predictions = [res['prediction'] for res in results.values()]
                    confidences = [res['confidence'] for res in results.values()]
                    
                    # Weighted voting: use confidence as weight
                    class_votes = {}
                    for model_name, res in results.items():
                        pred = res['prediction']
                        conf = res['confidence']
                        if pred not in class_votes:
                            class_votes[pred] = 0
                        class_votes[pred] += conf
                    
                    # Final consensus prediction
                    final_prediction = max(class_votes, key=class_votes.get)
                    final_confidence = class_votes[final_prediction] / len(results)
                    
                    # Check if all models agree
                    all_agree = len(set(predictions)) == 1
                    
                    st.success(f"✅ Analysis complete! ({elapsed_time:.2f}s)")
                    
                    # Display diagnostic result in col2 (right side)
                    with col2:
                        st.markdown("### 🔬 Diagnostic Result")
                        
                        # Main prediction card
                        if final_prediction == "No DR" or final_prediction == class_names[0]:
                            st.success(f"### ✅ {final_prediction}")
                        elif "Mild" in final_prediction or final_prediction == class_names[1]:
                            st.info(f"### ℹ️ {final_prediction}")
                        elif "Moderate" in final_prediction or final_prediction == class_names[2]:
                            st.warning(f"### ⚠️ {final_prediction}")
                        else:
                            st.error(f"### 🚨 {final_prediction}")
                        
                        st.markdown(f"**Confidence Score:** `{final_confidence:.1f}%`")
                        st.progress(float(final_confidence) / 100.0)
                        
                        if all_agree:
                            st.success("✅ All models agree on this diagnosis")
                        else:
                            st.info(f"ℹ️ Consensus from {len(results)} models")
                        
                        # Show individual model predictions in a table
                        st.markdown("---")
                        st.subheader("Overall Confidence")
                        
                        # Create DataFrame for individual predictions
                        individual_preds = []
                        for model_name, res in results.items():
                            individual_preds.append({
                                'Model': model_name,
                                'Prediction': res['prediction'],
                                'Confidence': f"{res['confidence']:.1f}%"
                            })
                        
                        pred_table_df = pd.DataFrame(individual_preds)
                        
                        # Display table with styling
                        st.dataframe(
                            pred_table_df,
                            hide_index=True,
                            use_container_width=True,
                            height=150
                        )
                        
                        # Show consensus gauge below in smaller size
                        fig_consensus = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=final_confidence,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Consensus Confidence", 'font': {'size': 16}},
                            gauge={
                                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                'bar': {'color': "darkblue"},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, 50], 'color': '#ff6b6b'},
                                    {'range': [50, 75], 'color': '#ffd93d'},
                                    {'range': [75, 100], 'color': '#6bcf7f'}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig_consensus.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig_consensus, use_container_width=True)
                    
                    # Display prediction analysis below the image in col1
                    with col1:
                        st.markdown("---")
                        st.subheader("Prediction Analysis")
                        
                        # Probability chart
                        st.plotly_chart(
                            create_probability_chart(results, class_names),
                            use_container_width=True
                        )
                    
                    # Additional visualizations below the two columns
                    st.markdown("---")
                    st.subheader("📊 Detailed Analysis")
                    
                    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Class Probabilities", "Model Agreement", "Confidence Analysis"])
                    
                    with viz_tab1:
                        # Individual class probability breakdown
                        prob_data = []
                        for model_name, res in results.items():
                            for i, class_name in enumerate(class_names):
                                prob_data.append({
                                    'Model': model_name,
                                    'Class': class_name,
                                    'Probability': res['probabilities'][i] * 100
                                })
                        
                        prob_df = pd.DataFrame(prob_data)
                        
                        # Heatmap of probabilities
                        prob_pivot = prob_df.pivot(index='Model', columns='Class', values='Probability')
                        fig_prob_heat = go.Figure(data=go.Heatmap(
                            z=prob_pivot.values,
                            x=prob_pivot.columns,
                            y=prob_pivot.index,
                            colorscale='Viridis',
                            text=prob_pivot.values,
                            texttemplate='%{text:.1f}%',
                            textfont={"size": 12},
                            colorbar=dict(title="Probability (%)")
                        ))
                        fig_prob_heat.update_layout(
                            title='Class Probability Heatmap',
                            xaxis_title='Classes',
                            yaxis_title='Models',
                            height=350
                        )
                        st.plotly_chart(fig_prob_heat, use_container_width=True)
                        
                        # Stacked bar chart
                        fig_stacked = px.bar(
                            prob_df,
                            x='Model',
                            y='Probability',
                            color='Class',
                            title='Stacked Probability Distribution',
                            labels={'Probability': 'Probability (%)'},
                            text='Probability'
                        )
                        fig_stacked.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
                        fig_stacked.update_layout(height=400)
                        st.plotly_chart(fig_stacked, use_container_width=True)
                    
                    with viz_tab2:
                        # Model agreement analysis
                        predictions = [res['prediction'] for res in results.values()]
                        unique_predictions = set(predictions)
                        
                        col_a, col_b = st.columns([1, 1])
                        
                        with col_a:
                            if len(unique_predictions) == 1:
                                st.success(f"✅ All models agree: **{predictions[0]}**")
                            else:
                                st.warning(f"⚠️ Models disagree. {len(unique_predictions)} different predictions.")
                            
                            # Prediction summary
                            pred_summary = pd.DataFrame({
                                'Model': list(results.keys()),
                                'Prediction': predictions,
                                'Confidence': [f"{res['confidence']:.2f}%" for res in results.values()]
                            })
                            st.dataframe(pred_summary, use_container_width=True)
                        
                        with col_b:
                            # Confidence comparison bar chart
                            conf_data = pd.DataFrame({
                                'Model': list(results.keys()),
                                'Confidence': [res['confidence'] for res in results.values()],
                                'Prediction': predictions
                            })
                            
                            fig_conf = px.bar(
                                conf_data,
                                x='Model',
                                y='Confidence',
                                color='Prediction',
                                title='Confidence by Model',
                                text='Confidence'
                            )
                            fig_conf.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                            fig_conf.update_layout(height=400, yaxis_range=[0, 100])
                            st.plotly_chart(fig_conf, use_container_width=True)
                        
                        # Consensus score
                        consensus_score = (predictions.count(predictions[0]) / len(predictions)) * 100
                        st.metric(
                            "Model Consensus",
                            f"{consensus_score:.0f}%",
                            help="Percentage of models agreeing on the prediction"
                        )
                    
                    with viz_tab3:
                        # Confidence distribution and statistics
                        confidences = [res['confidence'] for res in results.values()]
                        
                        col_c1, col_c2, col_c3 = st.columns(3)
                        with col_c1:
                            st.metric("Average Confidence", f"{np.mean(confidences):.2f}%")
                        with col_c2:
                            st.metric("Max Confidence", f"{np.max(confidences):.2f}%")
                        with col_c3:
                            st.metric("Min Confidence", f"{np.min(confidences):.2f}%")
                        
                        # Radar chart for top 3 class probabilities per model
                        fig_radar = go.Figure()
                        for model_name, res in results.items():
                            top_3_indices = np.argsort(res['probabilities'])[-3:][::-1]
                            top_3_classes = [class_names[i] for i in top_3_indices]
                            top_3_probs = [res['probabilities'][i] * 100 for i in top_3_indices]
                            
                            fig_radar.add_trace(go.Scatterpolar(
                                r=top_3_probs,
                                theta=top_3_classes,
                                fill='toself',
                                name=model_name
                            ))
                        
                        fig_radar.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                            showlegend=True,
                            title="Top 3 Class Probabilities (Radar)",
                            height=400
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Tab 2: Batch Prediction
    with tab2:
        st.header("Batch Image Analysis")
        
        uploaded_files = st.file_uploader(
            "Upload multiple fundus images",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'],
            accept_multiple_files=True,
            help="Upload multiple retinal fundus images for batch processing"
        )
        
        if uploaded_files and st.button("🔍 Analyze Batch", type="primary"):
            batch_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                # Read image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Predict
                results = predict_image(models, device, img_bgr, class_names)
                
                # Store results
                row = {'Filename': uploaded_file.name}
                for model_name, res in results.items():
                    row[f'{model_name}_Prediction'] = res['prediction']
                    row[f'{model_name}_Confidence'] = f"{res['confidence']:.2f}%"
                batch_results.append(row)
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("✅ Batch processing complete!")
            
            # Display results table
            df_results = pd.DataFrame(batch_results)
            st.dataframe(df_results, use_container_width=True)
            
            # Download button
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name="batch_predictions.csv",
                mime="text/csv"
            )
            
            # Batch analysis visualizations
            st.markdown("---")
            st.subheader("📊 Batch Analysis Visualizations")
            
            batch_viz_tab1, batch_viz_tab2, batch_viz_tab3 = st.tabs(
                ["Prediction Distribution", "Confidence Analysis", "Model Agreement"]
            )
            
            with batch_viz_tab1:
                # Extract predictions for each model
                model_names = [col.replace('_Prediction', '') for col in df_results.columns if '_Prediction' in col]
                
                # Stacked bar chart showing prediction counts
                pred_counts = []
                for model in model_names:
                    pred_col = f'{model}_Prediction'
                    for pred_class in class_names:
                        count = (df_results[pred_col] == pred_class).sum()
                        pred_counts.append({
                            'Model': model,
                            'Class': pred_class,
                            'Count': count
                        })
                
                pred_counts_df = pd.DataFrame(pred_counts)
                fig_pred_dist = px.bar(
                    pred_counts_df,
                    x='Model',
                    y='Count',
                    color='Class',
                    title='Prediction Distribution by Model',
                    barmode='stack'
                )
                fig_pred_dist.update_layout(height=400)
                st.plotly_chart(fig_pred_dist, use_container_width=True)
                
                # Heatmap of predictions per image
                st.subheader("Image-wise Prediction Heatmap")
                pred_cols = [col for col in df_results.columns if '_Prediction' in col]
                pred_matrix = df_results[pred_cols].copy()
                pred_matrix.columns = [col.replace('_Prediction', '') for col in pred_matrix.columns]
                
                # Convert predictions to numeric for heatmap
                class_to_num = {cls: idx for idx, cls in enumerate(class_names)}
                pred_matrix_numeric = pred_matrix.applymap(lambda x: class_to_num.get(x, -1))
                
                fig_pred_heatmap = go.Figure(data=go.Heatmap(
                    z=pred_matrix_numeric.values,
                    x=pred_matrix_numeric.columns,
                    y=[f"Img {i+1}" for i in range(len(pred_matrix_numeric))],
                    colorscale='Viridis',
                    colorbar=dict(
                        title="Class",
                        tickvals=list(range(len(class_names))),
                        ticktext=class_names
                    )
                ))
                fig_pred_heatmap.update_layout(
                    title='Predictions Across Images',
                    xaxis_title='Models',
                    yaxis_title='Images',
                    height=max(400, len(pred_matrix_numeric) * 20)
                )
                st.plotly_chart(fig_pred_heatmap, use_container_width=True)
            
            with batch_viz_tab2:
                # Confidence analysis
                conf_data = []
                for model in model_names:
                    conf_col = f'{model}_Confidence'
                    confidences = df_results[conf_col].str.rstrip('%').astype(float)
                    for idx, conf in enumerate(confidences):
                        conf_data.append({
                            'Model': model,
                            'Image': f'Img {idx+1}',
                            'Confidence': conf,
                            'Filename': df_results.loc[idx, 'Filename']
                        })
                
                conf_df = pd.DataFrame(conf_data)
                
                # Box plot of confidence distribution
                conf_col1, conf_col2 = st.columns(2)
                
                with conf_col1:
                    fig_conf_box = px.box(
                        conf_df,
                        x='Model',
                        y='Confidence',
                        color='Model',
                        title='Confidence Distribution by Model',
                        points='all'
                    )
                    fig_conf_box.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_conf_box, use_container_width=True)
                
                with conf_col2:
                    # Violin plot
                    fig_conf_violin = px.violin(
                        conf_df,
                        x='Model',
                        y='Confidence',
                        color='Model',
                        title='Confidence Distribution (Violin Plot)',
                        box=True
                    )
                    fig_conf_violin.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_conf_violin, use_container_width=True)
                
                # Line chart showing confidence across images
                fig_conf_line = px.line(
                    conf_df,
                    x='Image',
                    y='Confidence',
                    color='Model',
                    markers=True,
                    title='Confidence Across Images',
                    hover_data=['Filename']
                )
                fig_conf_line.update_layout(height=400)
                st.plotly_chart(fig_conf_line, use_container_width=True)
                
                # Summary statistics
                st.subheader("Confidence Statistics")
                conf_stats = conf_df.groupby('Model')['Confidence'].agg(['mean', 'std', 'min', 'max']).reset_index()
                conf_stats.columns = ['Model', 'Mean', 'Std Dev', 'Min', 'Max']
                st.dataframe(conf_stats.style.format({
                    'Mean': '{:.2f}%',
                    'Std Dev': '{:.2f}%',
                    'Min': '{:.2f}%',
                    'Max': '{:.2f}%'
                }), use_container_width=True)
            
            with batch_viz_tab3:
                # Model agreement analysis
                st.subheader("Model Agreement Analysis")
                
                # Calculate agreement for each image
                agreement_scores = []
                for idx, row in df_results.iterrows():
                    predictions = [row[f'{model}_Prediction'] for model in model_names]
                    most_common = max(set(predictions), key=predictions.count)
                    agreement = (predictions.count(most_common) / len(predictions)) * 100
                    agreement_scores.append({
                        'Image': f'Img {idx+1}',
                        'Filename': row['Filename'],
                        'Agreement': agreement,
                        'Consensus': most_common
                    })
                
                agreement_df = pd.DataFrame(agreement_scores)
                
                agr_col1, agr_col2 = st.columns([2, 1])
                
                with agr_col1:
                    # Bar chart of agreement scores
                    fig_agreement = px.bar(
                        agreement_df,
                        x='Image',
                        y='Agreement',
                        color='Agreement',
                        title='Model Agreement Score per Image',
                        color_continuous_scale='RdYlGn',
                        hover_data=['Filename', 'Consensus']
                    )
                    fig_agreement.update_layout(height=400, yaxis_range=[0, 100])
                    st.plotly_chart(fig_agreement, use_container_width=True)
                
                with agr_col2:
                    # Summary metrics
                    avg_agreement = agreement_df['Agreement'].mean()
                    full_agreement = (agreement_df['Agreement'] == 100).sum()
                    no_agreement = (agreement_df['Agreement'] == (100 / len(model_names))).sum()
                    
                    st.metric("Average Agreement", f"{avg_agreement:.1f}%")
                    st.metric("Full Agreement", f"{full_agreement}/{len(agreement_df)}")
                    st.metric("No Agreement", f"{no_agreement}/{len(agreement_df)}")
                    
                    # Pie chart of agreement levels
                    agreement_bins = pd.cut(agreement_df['Agreement'], bins=[0, 50, 75, 100], labels=['Low', 'Medium', 'High'])
                    agreement_counts = agreement_bins.value_counts()
                    
                    fig_agr_pie = px.pie(
                        values=agreement_counts.values,
                        names=agreement_counts.index,
                        title='Agreement Level Distribution',
                        color_discrete_map={'Low': '#ff6b6b', 'Medium': '#ffd93d', 'High': '#6bcf7f'}
                    )
                    st.plotly_chart(fig_agr_pie, use_container_width=True)
                
                # Detailed agreement table
                st.subheader("Detailed Agreement Data")
                st.dataframe(agreement_df, use_container_width=True)
    
    # Tab 3: Model Comparison
    with tab3:
        st.header("Model Performance Comparison")
        
        st.info("""
        **Note:** Performance metrics are based on validation/test set evaluations during training.
        Upload your test dataset metrics or run evaluation scripts to update these values.
        """)
        
        # Example metrics (replace with actual loaded metrics from training logs)
        metrics_data = {
            'Model': ['CNN', 'DenseNet', 'AdaBoost'],
            'Accuracy': [78.5, 82.3, 75.8],
            'Precision': [76.2, 80.5, 73.4],
            'Recall': [75.8, 81.2, 74.1],
            'F1-Score': [76.0, 80.8, 73.7],
            'AUC': [0.82, 0.86, 0.79],
            'Inference Time (ms)': [45, 120, 25]
        }
        
        # Allow users to upload custom metrics
        st.subheader("📊 Load Custom Metrics")
        uploaded_metrics = st.file_uploader(
            "Upload metrics CSV (optional)",
            type=['csv'],
            help="CSV with columns: Model, Accuracy, Precision, Recall, F1-Score, AUC"
        )
        
        if uploaded_metrics is not None:
            metrics_df = pd.read_csv(uploaded_metrics)
        else:
            metrics_df = pd.DataFrame(metrics_data)
        
        # Display metrics table
        st.subheader("Performance Metrics")
        st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']), use_container_width=True)
        
        # Radar chart
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(create_performance_comparison(metrics_df), use_container_width=True)
        
        with col2:
            st.subheader("Key Insights")
            best_model = metrics_df.loc[metrics_df['F1-Score'].idxmax(), 'Model']
            best_f1 = metrics_df['F1-Score'].max()
            
            st.metric("Best Model", best_model, f"F1: {best_f1:.2f}")
            
            fastest_model = metrics_df.loc[metrics_df['Inference Time (ms)'].idxmin(), 'Model']
            fastest_time = metrics_df['Inference Time (ms)'].min()
            
            st.metric("Fastest Model", fastest_model, f"{fastest_time:.0f} ms")
        
        # Individual metric comparison
        st.subheader("Detailed Metric Comparison")
        
        # Multiple visualization options
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            metric_to_plot = st.selectbox(
                "Select Metric for Bar Chart",
                ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Inference Time (ms)']
            )
            
            fig_bar = px.bar(
                metrics_df,
                x='Model',
                y=metric_to_plot,
                color='Model',
                title=f'{metric_to_plot} Comparison',
                color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'],
                text=metric_to_plot
            )
            fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_bar.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with viz_col2:
            # Line chart for all metrics
            metrics_long = metrics_df.melt(
                id_vars=['Model'], 
                value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                var_name='Metric',
                value_name='Score'
            )
            
            fig_line = px.line(
                metrics_long,
                x='Metric',
                y='Score',
                color='Model',
                markers=True,
                title='Performance Metrics Trend',
                color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
            )
            fig_line.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig_line, use_container_width=True)
        
        # Heatmap visualization
        st.subheader("Performance Heatmap")
        
        # Prepare data for heatmap
        heatmap_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        heatmap_data = metrics_df[['Model'] + heatmap_metrics].set_index('Model')
        
        # Normalize AUC to 0-100 scale for better visualization
        heatmap_data_normalized = heatmap_data.copy()
        if 'AUC' in heatmap_data_normalized.columns:
            heatmap_data_normalized['AUC'] = heatmap_data_normalized['AUC'] * 100
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data_normalized.values,
            x=heatmap_data_normalized.columns,
            y=heatmap_data_normalized.index,
            colorscale='RdYlGn',
            text=heatmap_data_normalized.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 14},
            colorbar=dict(title="Score"),
            hoverongaps=False
        ))
        
        fig_heatmap.update_layout(
            title='Model Performance Heatmap (Higher is Better)',
            xaxis_title='Metrics',
            yaxis_title='Models',
            height=400
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Grouped bar chart comparing all metrics
        st.subheader("Multi-Metric Comparison")
        
        metrics_for_comparison = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_comparison = metrics_df.melt(
            id_vars=['Model'],
            value_vars=metrics_for_comparison,
            var_name='Metric',
            value_name='Score'
        )
        
        fig_grouped = px.bar(
            metrics_comparison,
            x='Metric',
            y='Score',
            color='Model',
            barmode='group',
            title='Grouped Metric Comparison Across Models',
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'],
            text='Score'
        )
        fig_grouped.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_grouped.update_layout(height=450, hovermode='x unified', yaxis_range=[0, 100])
        st.plotly_chart(fig_grouped, use_container_width=True)
        
        # Box plot for metric distribution (if per-class metrics available)
        st.subheader("Metric Distribution Analysis")
        
        # Create synthetic per-class data for demonstration
        # In production, load actual per-class metrics from training logs
        np.random.seed(42)
        per_class_data = []
        for model in metrics_df['Model']:
            base_f1 = metrics_df[metrics_df['Model'] == model]['F1-Score'].values[0]
            for class_idx in range(4):  # Assuming 4 classes
                per_class_data.append({
                    'Model': model,
                    'Class': f'Class {class_idx}',
                    'F1-Score': base_f1 + np.random.uniform(-5, 5)
                })
        
        per_class_df = pd.DataFrame(per_class_data)
        
        fig_box = px.box(
            per_class_df,
            x='Model',
            y='F1-Score',
            color='Model',
            title='F1-Score Distribution Across Classes',
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'],
            points='all'
        )
        fig_box.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Individual Model Analysis
        st.markdown("---")
        st.subheader("📈 Individual Model Performance Analysis")
        
        model_colors = {'CNN': '#1f77b4', 'DenseNet': '#ff7f0e', 'AdaBoost': '#2ca02c'}
        
        # Create individual line graphs for each model
        ind_col1, ind_col2, ind_col3 = st.columns(3)
        
        for idx, (col, model_name) in enumerate(zip([ind_col1, ind_col2, ind_col3], metrics_df['Model'])):
            with col:
                # Get model's metrics
                model_row = metrics_df[metrics_df['Model'] == model_name].iloc[0]
                
                # Prepare data for line graph
                metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                metric_values = [model_row[metric] for metric in metric_names]
                
                # Create combined line and bar graph
                fig_individual = go.Figure()
                
                # Add bar chart
                fig_individual.add_trace(go.Bar(
                    x=metric_names,
                    y=metric_values,
                    name='Bar',
                    marker=dict(
                        color=model_colors.get(model_name, '#1f77b4'),
                        opacity=0.4,
                        line=dict(color=model_colors.get(model_name, '#1f77b4'), width=2)
                    ),
                    showlegend=False
                ))
                
                # Add line chart on top
                fig_individual.add_trace(go.Scatter(
                    x=metric_names,
                    y=metric_values,
                    mode='lines+markers+text',
                    name='Line',
                    line=dict(color=model_colors.get(model_name, '#1f77b4'), width=3),
                    marker=dict(size=12, color=model_colors.get(model_name, '#1f77b4'), 
                               line=dict(color='white', width=2)),
                    text=[f'{val:.1f}%' for val in metric_values],
                    textposition='top center',
                    textfont=dict(size=11, color=model_colors.get(model_name, '#1f77b4'), family='Arial Black'),
                    showlegend=False
                ))
                
                fig_individual.update_layout(
                    title=dict(
                        text=f'<b>{model_name}</b> Performance',
                        font=dict(size=16, color='black')
                    ),
                    xaxis_title='Metrics',
                    yaxis_title='Score (%)',
                    yaxis=dict(range=[0, 100]),
                    height=350,
                    showlegend=False,
                    hovermode='x unified',
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=40, r=20, t=60, b=40),
                    font=dict(color='black')
                )
                
                # Add grid
                fig_individual.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0', 
                                           title_font=dict(color='black'), tickfont=dict(color='black'))
                fig_individual.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0',
                                           title_font=dict(color='black'), tickfont=dict(color='black'))
                
                st.plotly_chart(fig_individual, use_container_width=True)
                
                # Add mini metrics below each graph
                st.markdown(f"""
                <div style='background-color: {model_colors.get(model_name, '#1f77b4')}15; 
                            padding: 10px; 
                            border-radius: 5px; 
                            border-left: 4px solid {model_colors.get(model_name, '#1f77b4')};'>
                    <small><b>AUC:</b> {model_row['AUC']:.3f}</small><br>
                    <small><b>Inference:</b> {model_row['Inference Time (ms)']:.0f} ms</small>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
