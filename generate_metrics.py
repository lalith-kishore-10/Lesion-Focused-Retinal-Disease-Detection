"""
Generate performance metrics for all trained models on test/validation data
Outputs a CSV file that can be loaded into the Streamlit app for comparison

Usage:
python generate_metrics.py --data_root data_root --split test --output metrics.csv
"""
import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from models.cnn_model import SmallFundusCNN, CLAHEDataset
from models.enhanced_densenet import EnhancedDenseNet
from models.adaboost_model import load_split
from joblib import load


def evaluate_dl_model(model, loader, device):
    """Evaluate deep learning model"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    inference_times = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            start = time.time()
            logits = model(x)
            inference_times.append((time.time() - start) * 1000 / x.size(0))  # ms per image
            
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            
            all_probs.append(probs)
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    
    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds) * 100
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    try:
        if all_probs.shape[1] == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        auc = 0.0
    
    cm = confusion_matrix(all_labels, all_preds)
    avg_inference_time = np.mean(inference_times)
    
    return {
        'accuracy': acc,
        'precision': prec * 100,
        'recall': rec * 100,
        'f1': f1 * 100,
        'auc': auc,
        'confusion_matrix': cm,
        'inference_time': avg_inference_time
    }


def evaluate_adaboost(pipe, root, split, imgsz):
    """Evaluate AdaBoost model"""
    X, y, _ = load_split(root / split, out_size=imgsz)
    
    start = time.time()
    probs = pipe.predict_proba(X)
    inference_time = (time.time() - start) * 1000 / len(X)  # ms per image
    
    preds = probs.argmax(axis=1)
    
    acc = accuracy_score(y, preds) * 100
    prec, rec, f1, _ = precision_recall_fscore_support(y, preds, average='macro', zero_division=0)
    
    try:
        if probs.shape[1] == 2:
            auc = roc_auc_score(y, probs[:, 1])
        else:
            auc = roc_auc_score(y, probs, multi_class='ovr', average='macro')
    except:
        auc = 0.0
    
    cm = confusion_matrix(y, preds)
    
    return {
        'accuracy': acc,
        'precision': prec * 100,
        'recall': rec * 100,
        'f1': f1 * 100,
        'auc': auc,
        'confusion_matrix': cm,
        'inference_time': inference_time
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = Path(args.data_root)
    
    results = []
    
    # Prepare data loaders for DL models
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    base_transform = transforms.Compose([norm])
    
    print(f"Evaluating on {args.split} split...")
    
    # Evaluate CNN
    cnn_path = Path('runs/cnn_best.pt')
    if cnn_path.exists():
        print("Evaluating CNN...")
        ckpt = torch.load(cnn_path, map_location=device)
        model = SmallFundusCNN(num_classes=ckpt['num_classes']).to(device)
        model.load_state_dict(ckpt['model'])
        
        ds = CLAHEDataset(root / args.split, transform=base_transform, size=224)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
        
        metrics = evaluate_dl_model(model, loader, device)
        results.append({
            'Model': 'CNN',
            'Accuracy': f"{metrics['accuracy']:.2f}",
            'Precision': f"{metrics['precision']:.2f}",
            'Recall': f"{metrics['recall']:.2f}",
            'F1-Score': f"{metrics['f1']:.2f}",
            'AUC': f"{metrics['auc']:.3f}",
            'Inference Time (ms)': f"{metrics['inference_time']:.1f}"
        })
        print(f"CNN - F1: {metrics['f1']:.2f}, AUC: {metrics['auc']:.3f}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    # Evaluate Enhanced DenseNet
    densenet_path = Path('runs/enhanced_densenet.pt')
    if densenet_path.exists():
        print("\nEvaluating Enhanced DenseNet...")
        ckpt = torch.load(densenet_path, map_location=device)
        model = EnhancedDenseNet(num_classes=ckpt['num_classes']).to(device)
        model.load_state_dict(ckpt['model'])
        
        ds = CLAHEDataset(root / args.split, transform=base_transform, size=256)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
        
        metrics = evaluate_dl_model(model, loader, device)
        results.append({
            'Model': 'DenseNet',
            'Accuracy': f"{metrics['accuracy']:.2f}",
            'Precision': f"{metrics['precision']:.2f}",
            'Recall': f"{metrics['recall']:.2f}",
            'F1-Score': f"{metrics['f1']:.2f}",
            'AUC': f"{metrics['auc']:.3f}",
            'Inference Time (ms)': f"{metrics['inference_time']:.1f}"
        })
        print(f"DenseNet - F1: {metrics['f1']:.2f}, AUC: {metrics['auc']:.3f}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    # Evaluate AdaBoost
    ada_path = Path('runs/adaboost.joblib')
    if ada_path.exists():
        print("\nEvaluating AdaBoost...")
        pipe = load(ada_path)
        metrics = evaluate_adaboost(pipe, root, args.split, args.imgsz)
        results.append({
            'Model': 'AdaBoost',
            'Accuracy': f"{metrics['accuracy']:.2f}",
            'Precision': f"{metrics['precision']:.2f}",
            'Recall': f"{metrics['recall']:.2f}",
            'F1-Score': f"{metrics['f1']:.2f}",
            'AUC': f"{metrics['auc']:.3f}",
            'Inference Time (ms)': f"{metrics['inference_time']:.1f}"
        })
        print(f"AdaBoost - F1: {metrics['f1']:.2f}, AUC: {metrics['auc']:.3f}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\n✅ Metrics saved to {args.output}")
        print("\nSummary:")
        print(df.to_string(index=False))
    else:
        print("❌ No models found to evaluate!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data_root', help='Path to dataset root')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'], help='Which split to evaluate')
    parser.add_argument('--batch', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--imgsz', type=int, default=224, help='Image size for AdaBoost')
    parser.add_argument('--output', type=str, default='metrics.csv', help='Output CSV file')
    args = parser.parse_args()
    main(args)
