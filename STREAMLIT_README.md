# Diabetic Retinopathy Detection - Streamlit App

## 🎯 Overview

Interactive web application for diabetic retinopathy detection using three trained models:

- **CNN** (SmallFundusCNN): Lightweight custom CNN
- **Enhanced DenseNet**: DenseNet121 with SE blocks and multi-scale fusion
- **AdaBoost**: Classical ML with LBP features

## ✨ Features

### 1. Single Image Prediction

- Upload fundus images for instant analysis
- Get predictions from all three models simultaneously
- View confidence levels with interactive gauges
- Compare probability distributions across classes

### 2. Batch Prediction

- Process multiple images at once
- Export results to CSV
- Track progress with real-time updates

### 3. Model Performance Comparison

- Interactive radar chart comparing all metrics
- Detailed performance tables
- Custom metrics upload support
- Per-metric bar chart comparisons

## 🚀 Quick Start

### Prerequisites

```powershell
# Install dependencies
pip install -r requirements.txt
```

### Generate Performance Metrics (Optional but Recommended)

```powershell
# Evaluate all models on test set and generate metrics CSV
python generate_metrics.py --data_root data_root --split test --output metrics.csv
```

### Launch the App

```powershell
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📋 Usage Instructions

### Before First Use

1. **Ensure models are trained:**

   - `runs/cnn_best.pt` (CNN model)
   - `runs/enhanced_densenet.pt` (DenseNet model)
   - `runs/adaboost.joblib` (AdaBoost model)

2. **Optional: Generate metrics file**
   ```powershell
   python generate_metrics.py --data_root data_root --split test --output metrics.csv
   ```

### Single Image Analysis

1. Navigate to "📸 Single Image Prediction" tab
2. Upload a fundus image (JPG, PNG, etc.)
3. Click "🔍 Analyze Image"
4. View predictions, confidence levels, and probability charts

### Batch Processing

1. Navigate to "📊 Batch Prediction" tab
2. Upload multiple fundus images
3. Click "🔍 Analyze Batch"
4. Download results as CSV

### Compare Models

1. Navigate to "📈 Model Comparison" tab
2. Optionally upload custom metrics CSV
3. View interactive radar chart and metric comparisons
4. Analyze individual metrics with bar charts

## 📊 Understanding the Metrics

### Performance Metrics

- **Accuracy**: Overall correctness (%)
- **Precision**: Positive predictive value (%)
- **Recall**: Sensitivity/True positive rate (%)
- **F1-Score**: Harmonic mean of precision and recall (%)
- **AUC**: Area Under ROC Curve (0-1)
- **Inference Time**: Average prediction time per image (ms)

### Confidence Interpretation

- 🔴 **0-50%**: Low confidence (red zone)
- 🟡 **50-75%**: Medium confidence (yellow zone)
- 🟢 **75-100%**: High confidence (green zone)

## 🎨 Customization

### Change Class Names

In the sidebar, update the "Class Names" field with your dataset's classes:

```
No DR,Mild,Moderate,Severe
```

### Upload Custom Metrics

1. Create a CSV with columns:
   ```
   Model,Accuracy,Precision,Recall,F1-Score,AUC,Inference Time (ms)
   ```
2. Upload in the "Model Comparison" tab
3. Example:
   ```csv
   Model,Accuracy,Precision,Recall,F1-Score,AUC,Inference Time (ms)
   CNN,78.5,76.2,75.8,76.0,0.82,45
   DenseNet,82.3,80.5,81.2,80.8,0.86,120
   AdaBoost,75.8,73.4,74.1,73.7,0.79,25
   ```

## 🔧 Advanced Configuration

### Model Paths

Edit in `app.py` if your models are in different locations:

```python
cnn_path = Path('runs/cnn_best.pt')
densenet_path = Path('runs/enhanced_densenet.pt')
ada_path = Path('runs/adaboost.joblib')
```

### Image Preprocessing Sizes

- CNN: 224x224
- DenseNet: 256x256
- AdaBoost: 224x224 (configurable)

### Batch Size for Metrics Generation

```powershell
python generate_metrics.py --data_root data_root --split test --batch 32
```

## 🐛 Troubleshooting

### Model Loading Errors

**Problem**: Models fail to load
**Solutions**:

1. Verify model files exist in `runs/` folder
2. Check PyTorch/sklearn compatibility
3. Ensure models were trained successfully

### Memory Issues with Large Batches

**Problem**: Out of memory during batch prediction
**Solutions**:

1. Reduce batch size in `generate_metrics.py`
2. Process fewer images at once
3. Use CPU instead of GPU (slower but stable)

### Image Upload Failures

**Problem**: Images fail to upload or process
**Solutions**:

1. Ensure image format is supported (JPG, PNG, BMP, TIF)
2. Check image is not corrupted
3. Verify image has 3 channels (RGB/BGR)

### Metrics Not Displaying

**Problem**: Model comparison shows placeholder data
**Solutions**:

1. Generate metrics: `python generate_metrics.py`
2. Upload custom metrics CSV in the app
3. Ensure CSV format matches expected columns

## 📝 Notes

- **GPU Support**: Automatically uses CUDA if available
- **Model Caching**: Models are cached on first load for faster subsequent runs
- **Thread Safety**: Single user at a time recommended for production
- **Data Privacy**: All processing happens locally; no data is uploaded externally

## 🎓 Citation

If using this app for research, please cite your trained models and datasets appropriately.

## 📧 Support

For issues or questions:

1. Check the troubleshooting section above
2. Verify all models are trained and saved correctly
3. Review terminal/console output for error messages

## 🔄 Updates and Maintenance

To update the app:

1. Pull latest changes
2. Update dependencies: `pip install -r requirements.txt --upgrade`
3. Regenerate metrics if model performance changes
4. Test with sample images before production use

---

**Happy Analyzing! 👁️**
