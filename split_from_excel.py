# split_from_excel.py
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import shutil
from collections import defaultdict

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_files(grouped_files, out, val_ratio=0.15, test_ratio=0.15, seed=42):
    summary = {}
    for lbl, files in grouped_files.items():
        files = list(files)
        if not files:
            continue
        train_and_val, test = train_test_split(files, test_size=test_ratio, random_state=seed)
        train, val = train_test_split(train_and_val, test_size=val_ratio/(1-test_ratio), random_state=seed)
        summary[lbl] = {'train': len(train), 'val': len(val), 'test': len(test), 'total': len(files)}
        for split_name, split_files in (('train', train), ('val', val), ('test', test)):
            out_dir = out / split_name / str(lbl)
            ensure_dir(out_dir)
            for src_path in split_files:
                shutil.copy2(src_path, out_dir / src_path.name)
    return summary

def find_file_insensitive(src_dir: Path, filename: str):
    # Quickly check common extensions and case-insensitive matches
    p = src_dir / filename
    if p.exists():
        return p
    # Try with ext variations if user supplied no ext
    basename = Path(filename).stem
    for ext in IMG_EXTS:
        cand = src_dir / (basename + ext)
        if cand.exists():
            return cand
    # Try case-insensitive search (slower)
    for f in src_dir.iterdir():
        if f.is_file() and f.name.lower() == filename.lower():
            return f
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel', required=True, help='Path to annotations Excel file (.xlsx or .xls)')
    parser.add_argument('--src', default='data_raw', help='Folder with all images (flat)')
    parser.add_argument('--out', default='data_root', help='Destination root for train/val/test')
    parser.add_argument('--img_col', default='Image name', help='Excel column name containing image filenames')
    parser.add_argument('--label_col', default='Retinopathy grade', help='Excel column name to use as label')
    parser.add_argument('--val', type=float, default=0.15)
    parser.add_argument('--test', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    excel_path = Path(args.excel)
    src_dir = Path(args.src)
    out_dir = Path(args.out)

    if not excel_path.exists():
        raise SystemExit(f"Excel file not found: {excel_path}")
    if not src_dir.exists():
        raise SystemExit(f"Source folder not found: {src_dir}")

    # read excel
    df = pd.read_excel(excel_path)
    # normalize column names (strip)
    df.columns = [c.strip() for c in df.columns]

    img_col = args.img_col.strip()
    lbl_col = args.label_col.strip()
    if img_col not in df.columns or lbl_col not in df.columns:
        raise SystemExit(f"Columns not found in Excel. Available columns: {list(df.columns)}")

    # build mapping filename -> label
    grouped = defaultdict(list)
    missing = []
    for _, row in df.iterrows():
        fname = str(row[img_col]).strip()
        if not fname or fname == 'nan':
            continue
        found = find_file_insensitive(src_dir, fname)
        if found is None:
            missing.append(fname)
            continue
        lbl = row[lbl_col]
        # convert NaN labels or None to string 'unknown'
        if pd.isna(lbl):
            lbl = 'unknown'
        # ensure label is string (so folder names are correct)
        lbl = str(lbl).strip()
        grouped[lbl].append(found)

    if missing:
        print("Warning: the following files from the Excel were NOT found in the source folder:")
        for m in missing[:20]:
            print("  ", m)
        if len(missing) > 20:
            print(f"  ...and {len(missing)-20} more")

    # create out base
    for s in ('train','val','test'):
        ensure_dir(out_dir / s)

    summary = copy_files(grouped, out_dir, val_ratio=args.val, test_ratio=args.test, seed=args.seed)

    print(f"\nCreated dataset at: {out_dir}")
    total = 0
    for lbl, info in summary.items():
        print(f"Label '{lbl}': total {info['total']} -> train {info['train']}, val {info['val']}, test {info['test']}")
        total += info['total']
    print(f"Total images processed (found & copied): {total}")
    if missing:
        print(f"Total missing images: {len(missing)}")

if __name__ == '__main__':
    main()
