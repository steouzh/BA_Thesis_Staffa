# benchmark_data.py
from typing import Optional
import os
from pathlib import Path

import pandas as pd

# ===================== DATA LOADING =====================

def inspect_columns(path: str):
    """Inspect Excel file columns."""
    try:
        df = pd.read_excel(path, nrows=5)
        print(f"📊 Columns in {path}:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        print(f"\n💡 First row sample:")
        print(df.iloc[0].to_dict() if len(df) > 0 else "No data")
    except Exception as e:
        print(f"❌ Error reading file: {e}")

def load_data(raw_path: str, ibf_path: str = None) -> pd.DataFrame:
    """Load product data and merge IBF information BEFORE normalization:
       1) match IBF AssetName to product Type (priority)
       2) fallback: match by Product brand"""

    # ------------------------------------------------------
    # 1. LOAD RAW PRODUCT DATA
    # ------------------------------------------------------
    try:
        df_raw = pd.read_excel(raw_path, sheet_name=0)
    except:
        df_raw = pd.read_excel(raw_path)

    # Keep ORIGINAL type/brand BEFORE normalization
    df_products = df_raw.copy()

    # Normalize column names to lower for matching
    df_products.columns = [c.lower().strip() for c in df_products.columns]

    # Map important columns
    rename_map = {
        "<id>": "product_id",
        "product brand": "brand",
        "type": "type"
    }
    df_products = df_products.rename(columns={k.lower(): v for k, v in rename_map.items() if k.lower() in df_products.columns})

    # Clean fields
    df_products["brand"] = df_products["brand"].astype(str).str.lower().str.strip()
    df_products["type"] = df_products["type"].astype(str).str.lower().str.strip()

    # Prepare IBF column
    df_products["ibf_data"] = ""

    # ------------------------------------------------------
    # 2. LOAD IBF DATA
    # ------------------------------------------------------
    if ibf_path and os.path.exists(ibf_path):
        df_ibf = pd.read_excel(ibf_path)

        # Normalize IBF columns
        ibf_rename = {
            "AssetName": "asset_name",
            "AssetName.1": "asset_name_alt",
            "Title": "IBF_Title",
            "Insight": "IBF_Insight",
            "Benefit": "IBF_Benefit",
            "Function": "IBF_Feature"
        }
        df_ibf = df_ibf.rename(columns=ibf_rename)

        df_ibf["asset_name"] = df_ibf["asset_name"].astype(str).str.lower().str.strip()

        # Format IBF blocks
        def format_ibfs(group):
            text = ""
            for i, row in enumerate(group.itertuples(), 1):
                text += f"\nIBF {i}:\n"
                text += f"  Title: {getattr(row, 'IBF_Title', '')}\n"
                text += f"  Insight: {getattr(row, 'IBF_Insight', '')}\n"
                text += f"  Benefit: {getattr(row, 'IBF_Benefit', '')}\n"
                text += f"  Feature: {getattr(row, 'IBF_Feature', '')}\n"
            return text

        ibf_by_asset = df_ibf.groupby("asset_name").apply(format_ibfs).to_dict()

        # ------------------------------------------------------
        # 3. MATCH IBFs BEFORE NORMALIZATION
        # ------------------------------------------------------
        def apply_ibf(row):
            t = row["type"]
            b = row["brand"]

            # match by type
            if t in ibf_by_asset:
                return ibf_by_asset[t]

            # match by brand
            if b in ibf_by_asset:
                return ibf_by_asset[b]

            return ""

        df_products["ibf_data"] = df_products.apply(apply_ibf, axis=1)

        matched = (df_products["ibf_data"] != "").sum()
        print(f"IBF match success: {matched}/{len(df_products)}")

    else:
        print("⚠️ No IBF file provided")

    # ------------------------------------------------------
    # 4. NOW NORMALIZE THE PRODUCT DATA
    # ------------------------------------------------------
    df_final = normalize_export_df(df_products)

    # The normalize function will drop type/brand, but we want IBF preserved
    df_final["ibf_data"] = df_products["ibf_data"]

    return df_final

def normalize_export_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and extract features + technical data."""
    df_copy = df.copy()

    # Product ID
    if "product_id" in df_copy.columns:
        df_copy["product_id"] = df_copy["product_id"].astype(str)
    else:
        df_copy["product_id"] = [f"prod_{i+1:04d}" for i in range(len(df_copy))]

    df_copy["name"] = df_copy["Product designation"] if "Product designation" in df_copy.columns else ""

    if "Product brand" in df_copy.columns:
        df_copy["series"] = df_copy["Product brand"]
    elif "Sortiment" in df_copy.columns:
        df_copy["series"] = df_copy["Sortiment"]
    else:
        df_copy["series"] = ""

    # Features = Characteristics + Application purposes
    if "Characteristics" in df_copy.columns or "Application purposes" in df_copy.columns:

        def merge_features(row):
            feats = []

            # Add Characteristics
            char = row.get("Characteristics")
            if pd.notna(char) and isinstance(char, str):
                for line in char.split("\n"):
                    line = line.strip()
                    if line:
                        feats.append(line)

            # Add Application purposes
            app = row.get("Application purposes")
            if pd.notna(app) and isinstance(app, str):
                for line in app.split("\n"):
                    line = line.strip()
                    if line:
                        feats.append(line)

            return feats

        df_copy["features"] = df_copy.apply(merge_features, axis=1)

    else:
        df_copy["features"] = [[] for _ in range(len(df_copy))]

    # Technical Data
    exclude_cols = {
        '<ID>', '<n>', 'Product brand', 'Type', 'Sortiment',
        'Product designation', 'Reference for product designation',
        'Image', '<0 Primary Image EPS.|Node|.URL of converted asset>',
        'Application purposes', 'Characteristics', 'Scope of delivery'
    }

    df_copy["technical_data"] = df_copy.apply(
        lambda row: collect_tech_specs(row, exclude_cols),
        axis=1
    )      
    df_copy["specs"] = df_copy["technical_data"].apply(
        lambda d: ", ".join([f"{k}: {v}" for k, v in d.items()]) if d else ""
    )

    return df_copy[["product_id", "name", "series", "features", "technical_data", "specs"]]

def collect_tech_specs(row, exclude_cols):
    specs = {}

    for col in row.index:

        if col in exclude_cols:
            continue

        val = row[col]

        if hasattr(val, "__len__") and not isinstance(val, str):
            continue

        if val is None:
            continue

        sval = str(val).strip().lower()
        if sval in {"", "nan", "none"}:
            continue

        specs[col] = str(val).strip()

    return specs

def load_gold_if_exists(gold_path: str, sbert_model) -> Optional[pd.DataFrame]:
    """Load gold standards."""
    if not gold_path or not os.path.exists(gold_path):
        return None
    
    try:
        gold = pd.read_excel(gold_path)
        if "marketing_text" not in gold.columns:
            print(f"⚠️ Gold file missing 'marketing_text' column")
            return None
        
        gold["gold_emb"] = gold["marketing_text"].apply(
            lambda x: sbert_model.encode(str(x), convert_to_tensor=True, show_progress_bar=False) if pd.notna(x) else None
        )
        print(f"✅ Loaded {len(gold)} gold standards")
        return gold
    except Exception as e:
        print(f"⚠️ Could not load gold standards: {e}")
        return None

def load_series_gold(gold_path: str, sbert_model):
    """
    Loads ONLY series-level gold baselines.
    Expected columns:
        - series
        - marketing_text
    Returns:
        dict: {series_name: {"text": ..., "emb": ...}}
    """
    import os
    import pandas as pd

    if not gold_path or not os.path.exists(gold_path):
        return {}

    try:
        gold_df = pd.read_excel(gold_path)

        if not {"series", "marketing_text"}.issubset(gold_df.columns):
            print("⚠️ Gold file must contain columns: 'series', 'marketing_text'")
            return {}

        series_gold = {}

        for _, row in gold_df.iterrows():
            series = str(row["series"]).strip()
            text = str(row["marketing_text"]).strip()

            if not text:
                continue

            emb = sbert_model.encode(text, convert_to_tensor=True, show_progress_bar=False)

            series_gold[series] = {"text": text, "emb": emb}

        print(f"✅ Loaded {len(series_gold)} series-level gold standards")
        return series_gold

    except Exception as e:
        print(f"⚠️ Failed to load series gold standards: {e}")
        return {}
