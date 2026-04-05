"""
AdaBoost-based pipeline for DR detection using classical features
- Preprocessing: CLAHE
- Feature extraction: Local Binary Patterns (LBP) + color stats
- Classifier: AdaBoost (SAMME.R) with decision stumps

Usage:
python adaboost_model.py --data_root "path/to/dataset" --split train --model_out runs/adaboost.joblib
python adaboost_model.py --data_root "path/to/dataset" --split test  --model_in runs/adaboost.joblib

Dataset structure:
  data_root/
    train/ class0/... class1/...
    val/   class0/... class1/...
    test/  class0/... class1/...
"""
import os
import argparse
from pathlib import Path
from typing import Tuple, List
import cv2
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skimage.feature import local_binary_pattern
from joblib import dump, load


# --------- Preprocessing ---------

def apply_clahe_rgb(img_bgr: np.ndarray, clip=2.0, tile=(8,8)) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return out


def center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    m = min(h, w)
    y0 = (h - m) // 2
    x0 = (w - m) // 2
    return img[y0:y0+m, x0:x0+m]


def preprocess(img_bgr: np.ndarray, out_size: int = 256) -> np.ndarray:
    img = apply_clahe_rgb(img_bgr)
    img = center_crop_square(img)
    img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return img


# --------- Features ---------

def lbp_hist(gray: np.ndarray, P: int = 8, R: int = 1) -> np.ndarray:
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def color_stats(img_bgr: np.ndarray) -> np.ndarray:
    # Mean and std for each channel
    means = img_bgr.mean(axis=(0,1))
    stds = img_bgr.std(axis=(0,1)) + 1e-6
    return np.concatenate([means, stds]).astype(np.float32)


def vessel_enhancement(gray: np.ndarray) -> np.ndarray:
    # Simple Frangi-like via morphological operations to hint lesions
    blur = cv2.GaussianBlur(gray, (0,0), 1.0)
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, np.ones((17,17), np.uint8))
    return tophat


def extract_features(img_bgr: np.ndarray, out_size: int = 256) -> np.ndarray:
    img = preprocess(img_bgr, out_size=out_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enh = vessel_enhancement(gray)
    feats = [lbp_hist(gray), lbp_hist(enh), color_stats(img)]
    return np.concatenate(feats)


# --------- Data IO ---------

def load_split(root: Path, out_size: int = 256) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y, paths = [], [], []
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    class_to_idx = {c:i for i,c in enumerate(classes)}
    for c in classes:
        for p in (root/c).rglob('*'):
            if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                X.append(extract_features(img, out_size=out_size))
                y.append(class_to_idx[c])
                paths.append(str(p))
    return np.vstack(X), np.array(y, dtype=np.int64), paths


# --------- Main ---------

def main(args):
    split_dir = Path(args.data_root)/args.split
    if args.split == 'train':
        X, y, _ = load_split(split_dir, out_size=args.imgsz)
        
        # Compute class weights to handle imbalance
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight('balanced', y)
        
        # Use deeper tree for better performance
        base = DecisionTreeClassifier(max_depth=5, min_samples_split=5)
        # Use 'estimator' for newer scikit-learn (1.4+), fallback to 'base_estimator' for older versions
        # Note: SAMME.R was deprecated, use SAMME instead
        try:
            clf = AdaBoostClassifier(estimator=base, n_estimators=args.n_estimators, learning_rate=args.learning_rate, algorithm='SAMME', random_state=42)
        except TypeError:
            clf = AdaBoostClassifier(base_estimator=base, n_estimators=args.n_estimators, learning_rate=args.learning_rate, algorithm='SAMME', random_state=42)
        pipe = Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('clf', clf)
        ])
        # Fit with sample weights
        pipe.fit(X, y, clf__sample_weight=sample_weights)
        Path(Path(args.model_out).parent).mkdir(parents=True, exist_ok=True)
        dump(pipe, args.model_out)
        print(f"Saved model to {args.model_out}")
        
        # Print training class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"Training class distribution: {dict(zip(unique, counts))}")
    else:
        pipe = load(args.model_in)
        X, y, _ = load_split(split_dir, out_size=args.imgsz)
        probs = pipe.predict_proba(X)
        preds = probs.argmax(axis=1)
        f1 = f1_score(y, preds, average='macro')
        try:
            if probs.shape[1] == 2:
                auc = roc_auc_score(y, probs[:,1])
            else:
                auc = roc_auc_score(y, probs, multi_class='oNvr')
        except Exception:
            auc = float('nan')
        print(classification_report(y, preds))
        print("F1:", f1, "AUC:", auc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--split', type=str, choices=['train','val','test'], required=True)
    parser.add_argument('--model_out', type=str, default='runs/adaboost.joblib')
    parser.add_argument('--model_in', type=str, default='runs/adaboost.joblib')
    parser.add_argument('--imgsz', type=int, default=256, help='Resize images to this size before feature extraction')
    parser.add_argument('--n_estimators', type=int, default=200, help='Number of estimators for AdaBoost')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='Learning rate for AdaBoost')
    args = parser.parse_args()
    main(args)
