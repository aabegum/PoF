"""
STEP 4: COMPREHENSIVE ANALYTICS - BEST OF BOTH APPROACHES
===========================================================
Purpose:
- Survival Analysis (Kaplan-Meier + Weibull AFT)
- Rolling Window Backtesting
- Health Score Calculation (0-100)
- SHAP Analysis for Root Causes (FIXED)
- Interactive Risk Maps (Plotly)
- CAPEX Prioritization (5-tier)
- Model Calibration Analysis
- CatBoost Algorithm Comparison

This script combines the best features from both enhanced.py and advanced_analytics.py:
- FIXED SHAP implementation (no shape mismatch errors)
- Comprehensive model validation
- Rich interactive visualizations
- Health scoring system
- Advanced statistical analysis

Builds on: Step 3.5 optimized model (AUC 0.7148) with 16 Low VIF features
Input: outputs/step3_5_final_risk_scored.xlsx
Model: outputs/step3_5_final_model.pkl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import pickle
import json
import joblib
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Advanced Analytics Libraries
from lifelines import KaplanMeierFitter, WeibullAFTFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import shap
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_recall_curve,
    classification_report, confusion_matrix,
    precision_score, recall_score, brier_score_loss,
    roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
import xgboost as xgb

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = 'outputs/step3_5_final_risk_scored.xlsx'
MODEL_FILE = 'outputs/step3_5_final_model.pkl'
SCALER_FILE = 'outputs/step3_5_final_scaler.pkl'
FEATURE_IMPORTANCE_FILE = 'outputs/step3_5_feature_importance.csv'
OUTPUT_DIR = Path('outputs/step4_comprehensive/')
RANDOM_STATE = 42

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("STEP 4: COMPREHENSIVE ANALYTICS - BEST OF BOTH APPROACHES")
print("=" * 80)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Building on Step 3.5 optimized model (AUC 0.7148)\n")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_scale_features(scaler, X, feature_cols):
    """
    Safely scale features, handling cases where scaler needs fitting

    Args:
        scaler: StandardScaler object (may need fitting)
        X: DataFrame to scale
        feature_cols: List of columns to scale

    Returns:
        Scaled DataFrame
    """
    X_scaled = X.copy()

    # Check if scaler has feature names from training
    if hasattr(scaler, 'feature_names_in_'):
        # Scaler was fit with specific features - use only those
        scaler_features = list(scaler.feature_names_in_)

        # Find which of our features were in the scaler's training
        features_to_scale = [col for col in feature_cols if col in scaler_features]

        # Find features in data that scaler doesn't know about (won't be scaled)
        unscaled_features = [col for col in feature_cols if col not in scaler_features and col in X.columns]

        # Also check if scaler has features we don't have
        missing_in_data = [col for col in scaler_features if col not in X.columns]

        if missing_in_data:
            print(f"   ⚠️  Warning: Scaler expects features not in data: {missing_in_data}")
            # Add missing features with zeros
            for col in missing_in_data:
                X_scaled[col] = 0
            features_to_scale = scaler_features

        if features_to_scale:
            # Scale only the features the scaler knows about, in the correct order
            X_scaled[scaler_features] = scaler.transform(X_scaled[scaler_features])

        # Show info about what was scaled (only show once per run)
        if not hasattr(safe_scale_features, '_info_shown'):
            if unscaled_features:
                print(f"   ℹ️  Note: {len(scaler_features)} features scaled, {len(unscaled_features)} features kept unscaled")
            safe_scale_features._info_shown = True

        return X_scaled

    # Fallback: Check if we have features to scale
    available_cols = [col for col in feature_cols if col in X.columns]

    if not available_cols:
        return X_scaled

    # Check if scaler needs fitting
    if hasattr(scaler, '_needs_fitting') and scaler._needs_fitting:
        # Fit on this data
        scaler.fit(X[available_cols])
        scaler._needs_fitting = False

    # Transform
    X_scaled[available_cols] = scaler.transform(X[available_cols])

    return X_scaled


# ============================================================================
# PART 0: DATA LOADING & VALIDATION
# ============================================================================

def load_and_validate_data():
    """Load all required files and validate data integrity"""
    print("\n" + "="*80)
    print("PART 0: DATA LOADING & VALIDATION")
    print("="*80)

    # Expected input files
    input_files = {
        'data': INPUT_FILE,
        'model': MODEL_FILE,
        'scaler': SCALER_FILE,
        'features': FEATURE_IMPORTANCE_FILE
    }

    # Check file existence
    missing_files = []
    for name, path in input_files.items():
        if not Path(path).exists():
            missing_files.append(f"{name}: {path}")

    if missing_files:
        print("\n⚠️  MISSING FILES:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nPlease ensure all required files exist")
        return None, None, None, None

    # Load data
    print("\n✅ Loading data...")
    df = pd.read_excel(input_files['data'], engine='openpyxl')
    print(f"   Records loaded: {len(df):,}")

    # Parse dates if needed
    if 'Arıza_Tarihi' in df.columns and df['Arıza_Tarihi'].dtype == 'object':
        df['Arıza_Tarihi'] = pd.to_datetime(df['Arıza_Tarihi'])

    # Load model
    print("✅ Loading XGBoost model...")
    model = joblib.load(input_files['model'])

    # Load scaler with robust error handling
    print("✅ Loading StandardScaler...")
    try:
        scaler = joblib.load(input_files['scaler'])

        # Verify it's actually a StandardScaler
        if not isinstance(scaler, StandardScaler):
            print(f"   ⚠️  Loaded object is {type(scaler)}, not StandardScaler")
            print("   Creating new scaler (will be fitted from data)...")
            scaler = StandardScaler()
            scaler._needs_fitting = True
        else:
            print(f"   ✓ StandardScaler loaded successfully")
            scaler._needs_fitting = False

    except Exception as e:
        print(f"   ⚠️  Error loading scaler: {e}")
        print("   Creating new StandardScaler (will be fitted from data)...")
        scaler = StandardScaler()
        scaler._needs_fitting = True

    # Load feature importance
    print("✅ Loading feature importance...")
    feature_importance = pd.read_csv(input_files['features'])
    best_features = feature_importance['Feature'].tolist()[:16]  # Top 16 features

    # Validate data structure
    print("\n📊 Data Structure Validation:")
    required_cols = [
        'Ekipman Kodu', 'Ekipman Sınıfı', 'İlçe',
        'Ekipman_Yaşı_Yıl', 'PoF_12_month', 'PoF_Score_Final',
        'Risk_Category_Final', 'Arıza_Tarihi'
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"   ⚠️  Missing columns: {missing_cols}")
    else:
        print("   ✅ All required columns present")

    # Check for coordinates
    has_coords = 'KOORDINAT_X' in df.columns and 'KOORDINAT_Y' in df.columns
    if has_coords:
        print("   ✅ Coordinate columns found")
    else:
        print("   ⚠️  Coordinate columns not found")

    # Data quality checks
    print("\n📈 Data Quality Summary:")
    print(f"   - Total records: {len(df):,}")
    print(f"   - Unique equipment: {df['Ekipman Kodu'].nunique():,}")
    print(f"   - Date range: {df['Arıza_Tarihi'].min()} to {df['Arıza_Tarihi'].max()}")
    print(f"   - PoF positive rate: {df['PoF_12_month'].mean():.1%}")
    if has_coords:
        coords_missing = df[['KOORDINAT_X', 'KOORDINAT_Y']].isna().any(axis=1).sum()
        print(f"   - Records with coordinates: {len(df) - coords_missing:,}")

    return df, model, scaler, best_features


# ============================================================================
# PART 1: SURVIVAL ANALYSIS (KAPLAN-MEIER + WEIBULL AFT)
# ============================================================================

def survival_analysis(df):
    """
    Comprehensive survival analysis using Kaplan-Meier and Weibull AFT
    """
    print("\n" + "="*80)
    print("PART 1: SURVIVAL ANALYSIS")
    print("="*80)

    # Prepare survival data
    print("\n📊 Preparing survival analysis data...")

    # For each equipment, calculate time to first fault
    survival_data = df.groupby('Ekipman Kodu').agg({
        'Ekipman_Yaşı_Yıl': 'first',
        'Arıza_Tarihi': ['min', 'max', 'count'],
        'Ekipman Sınıfı': 'first',
        'İlçe': 'first',
        'PoF_12_month': 'max',
        'Toplam_Müşteri_Sayısı': 'first'
    }).reset_index()

    survival_data.columns = ['Ekipman_Kodu', 'Age', 'First_Fault', 'Last_Fault',
                             'Fault_Count', 'Equipment_Type', 'District',
                             'Had_Fault', 'Total_Customers']

    # Calculate time to failure in days
    analysis_date = df['Arıza_Tarihi'].max() + timedelta(days=30)
    survival_data['Time_To_Failure'] = (
        survival_data['Last_Fault'] - survival_data['First_Fault']
    ).dt.days

    # Event indicator - convert to boolean for proper counting
    survival_data['Failure_Event'] = (survival_data['Had_Fault'] > 0).astype(int)

    # For censored equipment
    censored_mask = survival_data['Failure_Event'] == 0
    survival_data.loc[censored_mask, 'Time_To_Failure'] = (
        analysis_date - survival_data.loc[censored_mask, 'First_Fault']
    ).dt.days

    # Filter valid data
    survival_data = survival_data[survival_data['Time_To_Failure'] > 0]

    print(f"   - Equipment tracked: {len(survival_data):,}")
    print(f"   - Failures (events): {survival_data['Failure_Event'].sum():,}")
    print(f"   - Censored (no event): {(survival_data['Failure_Event'] == 0).sum():,}")

    # ========================================
    # Kaplan-Meier Analysis
    # ========================================
    print("\n📈 Computing Kaplan-Meier survival curves...")

    kmf = KaplanMeierFitter()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Overall survival
    ax = axes[0, 0]
    kmf.fit(survival_data['Time_To_Failure'], survival_data['Failure_Event'],
            label='All Equipment')
    kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_title('Overall Survival Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Days Since First Fault')
    ax.set_ylabel('Survival Probability')
    ax.grid(True, alpha=0.3)

    # By district (top 3)
    ax = axes[0, 1]
    top_districts = survival_data['District'].value_counts().head(3).index
    for district in top_districts:
        mask = survival_data['District'] == district
        kmf.fit(survival_data.loc[mask, 'Time_To_Failure'],
               survival_data.loc[mask, 'Failure_Event'],
               label=district)
        kmf.plot_survival_function(ax=ax, ci_show=False)
    ax.set_title('Survival by Top 3 Districts', fontsize=14, fontweight='bold')
    ax.set_xlabel('Days Since First Fault')
    ax.set_ylabel('Survival Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # By customer load
    ax = axes[1, 0]
    survival_data['High_Load'] = survival_data['Total_Customers'] > survival_data['Total_Customers'].median()
    for load_level, label in [(True, 'High Load'), (False, 'Low Load')]:
        mask = survival_data['High_Load'] == load_level
        kmf.fit(survival_data.loc[mask, 'Time_To_Failure'],
               survival_data.loc[mask, 'Failure_Event'],
               label=label)
        kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_title('Survival by Customer Load', fontsize=14, fontweight='bold')
    ax.set_xlabel('Days Since First Fault')
    ax.set_ylabel('Survival Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # By equipment age groups
    ax = axes[1, 1]
    survival_data['Age_Group'] = pd.cut(survival_data['Age'],
                                        bins=[0, 10, 20, 100],
                                        labels=['0-10 years', '10-20 years', '20+ years'])
    for age_group in survival_data['Age_Group'].unique():
        if pd.notna(age_group):
            mask = survival_data['Age_Group'] == age_group
            kmf.fit(survival_data.loc[mask, 'Time_To_Failure'],
                   survival_data.loc[mask, 'Failure_Event'],
                   label=age_group)
            kmf.plot_survival_function(ax=ax, ci_show=False)
    ax.set_title('Survival by Equipment Age', fontsize=14, fontweight='bold')
    ax.set_xlabel('Days Since First Fault')
    ax.set_ylabel('Survival Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'survival_kaplan_meier.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: survival_kaplan_meier.png")
    plt.close()

    # ========================================
    # Weibull AFT Model
    # ========================================
    print("\n📈 Fitting Weibull AFT model...")

    aft_features = ['Age', 'Fault_Count']
    aft_data = survival_data[aft_features + ['Time_To_Failure', 'Failure_Event']].dropna()

    if len(aft_data) > 0:
        aft = WeibullAFTFitter()
        aft.fit(aft_data, duration_col='Time_To_Failure', event_col='Failure_Event')

        # Plot AFT results
        plt.figure(figsize=(10, 6))
        aft.plot()
        plt.title('Weibull AFT Model - Feature Effects', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'survival_weibull_aft.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: survival_weibull_aft.png")
        plt.close()

        # Save summary
        aft.summary.to_csv(OUTPUT_DIR / 'weibull_aft_summary.csv')
        print(f"   ✅ Saved: weibull_aft_summary.csv")

    # Save survival summary
    survival_summary = pd.DataFrame({
        'Category': ['All Equipment', 'High Load', 'Low Load'],
        'Count': [
            len(survival_data),
            survival_data['High_Load'].sum(),
            (~survival_data['High_Load']).sum()
        ],
        'Events': [
            survival_data['Failure_Event'].sum(),
            survival_data[survival_data['High_Load']]['Failure_Event'].sum(),
            survival_data[~survival_data['High_Load']]['Failure_Event'].sum()
        ]
    })

    survival_summary.to_excel(OUTPUT_DIR / 'survival_summary.xlsx', index=False)
    print(f"   ✅ Saved: survival_summary.xlsx")

    return survival_data


# ============================================================================
# PART 2: ROLLING WINDOW BACKTESTING
# ============================================================================

def backtesting_analysis(df, model, scaler, feature_cols):
    """
    Perform rolling window backtesting to validate model performance over time
    """
    print("\n" + "="*80)
    print("PART 2: ROLLING WINDOW BACKTESTING")
    print("="*80)

    print("\n📊 Setting up rolling window framework...")

    # Sort by date
    df_sorted = df.sort_values('Arıza_Tarihi').copy()

    # Define rolling windows
    min_date = df_sorted['Arıza_Tarihi'].min()
    max_date = df_sorted['Arıza_Tarihi'].max()

    print(f"   - Date range: {min_date.date()} to {max_date.date()}")

    # Create windows
    window_size = timedelta(days=180)  # 6 months
    test_size = timedelta(days=90)     # 3 months

    results = []
    window_id = 0

    current_date = min_date + timedelta(days=365)  # Start after 1 year

    print("\n🔄 Running rolling window validation...")

    # Get features the scaler expects (it was trained on ALL feature_cols)
    # We need to pass all features to the scaler, not just numeric ones
    scaler_features = feature_cols

    while current_date + test_size <= max_date:
        window_id += 1

        # Define periods
        train_end = current_date
        test_start = current_date
        test_end = current_date + test_size

        # Split data
        train_mask = df_sorted['Arıza_Tarihi'] < train_end
        test_mask = (df_sorted['Arıza_Tarihi'] >= test_start) & (df_sorted['Arıza_Tarihi'] < test_end)

        X_train = df_sorted.loc[train_mask, feature_cols]
        y_train = df_sorted.loc[train_mask, 'PoF_12_month']
        X_test = df_sorted.loc[test_mask, feature_cols]
        y_test = df_sorted.loc[test_mask, 'PoF_12_month']

        if len(X_train) < 100 or len(X_test) < 50:
            current_date += timedelta(days=30)
            continue

        # Scale features - use ALL features the scaler was trained on
        X_train_scaled = safe_scale_features(scaler, X_train, scaler_features)
        X_test_scaled = safe_scale_features(scaler, X_test, scaler_features)

        # Make predictions
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)

        results.append({
            'Window': window_id,
            'Train_End': train_end.date(),
            'Test_Start': test_start.date(),
            'Test_End': test_end.date(),
            'Train_Size': len(X_train),
            'Test_Size': len(X_test),
            'Test_Positives': y_test.sum(),
            'AUC': auc
        })

        print(f"   Window {window_id}: AUC = {auc:.4f} (Test: {len(X_test)}, Positives: {y_test.sum()})")

        # Move to next window
        current_date += timedelta(days=30)

    # Convert to DataFrame
    backtest_results = pd.DataFrame(results)

    if len(backtest_results) == 0:
        print("   ⚠️  No backtest windows generated")
        return None

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # AUC over time
    ax = axes[0]
    ax.plot(backtest_results['Window'], backtest_results['AUC'],
            marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax.axhline(y=backtest_results['AUC'].mean(), color='red',
               linestyle='--', label=f'Mean AUC: {backtest_results["AUC"].mean():.4f}')
    ax.fill_between(backtest_results['Window'],
                     backtest_results['AUC'].mean() - backtest_results['AUC'].std(),
                     backtest_results['AUC'].mean() + backtest_results['AUC'].std(),
                     alpha=0.2, color='red')
    ax.set_xlabel('Window Number', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('Model Performance Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Sample sizes
    ax = axes[1]
    ax.bar(backtest_results['Window'], backtest_results['Test_Size'],
           alpha=0.6, label='Test Size', color='#A23B72')
    ax.set_xlabel('Window Number', fontsize=12)
    ax.set_ylabel('Sample Size', fontsize=12)
    ax.set_title('Test Set Sizes', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'backtest_rolling_window.png', dpi=300, bbox_inches='tight')
    print(f"\n   ✅ Saved: backtest_rolling_window.png")
    plt.close()

    # Summary
    print("\n📊 Backtesting Summary:")
    print(f"   - Windows: {len(backtest_results)}")
    print(f"   - Mean AUC: {backtest_results['AUC'].mean():.4f}")
    print(f"   - Std AUC: {backtest_results['AUC'].std():.4f}")
    print(f"   - Min AUC: {backtest_results['AUC'].min():.4f}")
    print(f"   - Max AUC: {backtest_results['AUC'].max():.4f}")

    # Export
    backtest_results.to_excel(OUTPUT_DIR / 'backtest_results.xlsx', index=False)
    print(f"   ✅ Saved: backtest_results.xlsx")

    return backtest_results


# ============================================================================
# PART 3: HEALTH SCORE CALCULATION
# ============================================================================

def calculate_health_scores(df):
    """
    Calculate 0-100 health scores for all equipment
    """
    print("\n" + "="*80)
    print("PART 3: HEALTH SCORE CALCULATION")
    print("="*80)

    print("\n📊 Calculating health scores...")

    def calculate_health_score(row):
        """Convert PoF to Health Score (0-100), higher = better"""
        pof_score = row['PoF_Score_Final']

        # Base health (inverse of PoF) - 60% weight
        base_score = (1 - pof_score) * 60

        # Age penalty - 20% weight
        age_penalty = min(row['Ekipman_Yaşı_Yıl'] * 1.0, 20)

        # Fault history penalty - 20% weight
        fault_penalty = min(row.get('Arıza_Sayısı_12ay', 0) * 4, 20)

        # Operational load penalty
        load_penalty = 0
        if 'Toplam_Müşteri_Sayısı' in row:
            high_load = row['Toplam_Müşteri_Sayısı'] > df['Toplam_Müşteri_Sayısı'].median()
            if high_load:
                load_penalty = 5

        # Final score
        health_score = base_score - age_penalty - fault_penalty - load_penalty

        return max(0, min(100, health_score))

    # Apply calculation
    df['Health_Score'] = df.apply(calculate_health_score, axis=1)

    # Categorize
    def categorize_health(score):
        if score >= 80:
            return 'Excellent'
        elif score >= 60:
            return 'Good'
        elif score >= 40:
            return 'Fair'
        elif score >= 20:
            return 'Poor'
        else:
            return 'Critical'

    df['Health_Status'] = df['Health_Score'].apply(categorize_health)

    print("📊 Health Score Distribution:")
    health_stats = df['Health_Status'].value_counts()
    for status, count in health_stats.items():
        print(f"   {status:10s}: {count:6,} ({count/len(df)*100:5.1f}%)")

    print(f"\n   Average Health Score: {df['Health_Score'].mean():.1f}")
    print(f"   Median Health Score: {df['Health_Score'].median():.1f}")

    # Health by age groups
    print(f"\n📈 Health Score by Age:")
    age_groups = pd.cut(df['Ekipman_Yaşı_Yıl'], bins=[0, 5, 10, 20, 100])
    for age_group, score in df.groupby(age_groups)['Health_Score'].mean().items():
        print(f"   {age_group}: {score:.1f}")

    return df


# ============================================================================
# PART 4: SHAP ANALYSIS (FIXED VERSION)
# ============================================================================

def shap_analysis(df, model, feature_cols):
    """
    SHAP analysis with FIXED implementation (no shape mismatch)
    """
    print("\n" + "="*80)
    print("PART 4: SHAP EXPLAINABILITY ANALYSIS (FIXED)")
    print("="*80)

    print("\n🔍 Computing SHAP values...")

    # Prepare data
    df_shap = df[feature_cols].copy()
    df_shap = df_shap.fillna(df_shap.median())

    # Sample for performance
    sample_size = min(1000, len(df_shap))
    df_shap_sample = df_shap.sample(n=sample_size, random_state=RANDOM_STATE)
    print(f"   ✓ Sampled {sample_size:,} rows")

    # Get model features
    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
        print(f"   ✓ Model expects {len(model_features)} features")
    elif hasattr(model, 'get_booster'):
        booster = model.get_booster()
        model_features = booster.feature_names
        print(f"   ✓ Model expects {len(model_features)} features (from booster)")
    else:
        model_features = feature_cols
        print(f"   ⚠️  Using feature_cols as model features")

    # Ensure alignment
    missing_features = set(model_features) - set(df_shap_sample.columns)
    if missing_features:
        print(f"   ⚠️  Adding missing features: {missing_features}")
        for feat in missing_features:
            df_shap_sample[feat] = 0

    # Reorder to ensure we have all model features
    # Add any missing model features with zeros
    for feat in model_features:
        if feat not in df_shap_sample.columns:
            df_shap_sample[feat] = 0
    df_shap_sample = df_shap_sample[model_features]
    print(f"   ✓ Aligned features: {df_shap_sample.shape}")

    # Scale using safe_scale_features (load scaler)
    scaler = joblib.load(SCALER_FILE)
    X_shap = safe_scale_features(scaler, df_shap_sample, model_features)

    # Ensure columns are in the correct order for the model
    X_shap = X_shap[model_features]
    print(f"   ✓ Scaled shape: {X_shap.shape}")

    # Create explainer
    print("   ✓ Creating TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values - use numpy for stability
    print("   ✓ Computing SHAP values (this may take a moment)...")
    shap_values = explainer.shap_values(X_shap.values)
    print(f"   ✓ SHAP values shape: {np.array(shap_values).shape}")

    # Summary plot
    print("\n📊 Creating SHAP visualizations...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_shap, show=False, max_display=15)
    plt.title('SHAP Feature Importance Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: shap_summary.png")

    # Bar plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=15)
    plt.title('SHAP Mean Absolute Feature Importance', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'shap_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: shap_bar.png")

    # Waterfall plot for high-risk example
    high_risk_mask = df.loc[X_shap.index, 'Risk_Category_Final'] == 'High Risk'
    if high_risk_mask.any():
        high_risk_idx = high_risk_mask[high_risk_mask].index[0]
        high_risk_sample = X_shap.loc[high_risk_idx]

        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[X_shap.index.get_loc(high_risk_idx)],
                base_values=explainer.expected_value,
                data=high_risk_sample.values,
                feature_names=model_features
            ),
            max_display=15,
            show=False
        )
        plt.title('SHAP Waterfall: High-Risk Equipment', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'shap_waterfall.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: shap_waterfall.png")

    # Dependence plots
    top_features = ['Ekipman_Yaşı_Yıl', 'OG_Müşteri_Oranı', 'Kentsel_Müşteri_Oranı']
    top_features = [f for f in top_features if f in model_features]

    if len(top_features) >= 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, feature in enumerate(top_features[:3]):
            shap.dependence_plot(feature, shap_values, X_shap, ax=axes[i], show=False)
            axes[i].set_title(f'SHAP Dependence: {feature}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'shap_dependence.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: shap_dependence.png")

    # Calculate importance
    shap_df = pd.DataFrame(shap_values, columns=[f'SHAP_{f}' for f in X_shap.columns])
    global_importance = shap_df.abs().mean().sort_values(ascending=False)

    print(f"\n📊 Top 10 SHAP Features:")
    for i, (feature, importance) in enumerate(global_importance.head(10).items(), 1):
        print(f"   {i:2d}. {feature.replace('SHAP_', ''):30s}: {importance:.4f}")

    # Save detailed values
    shap_df['Ekipman_Kodu'] = df.loc[X_shap.index, 'Ekipman Kodu'].values
    shap_df['PoF_Score'] = df.loc[X_shap.index, 'PoF_Score_Final'].values
    shap_df['Health_Score'] = df.loc[X_shap.index, 'Health_Score'].values
    shap_df.to_csv(OUTPUT_DIR / 'shap_values_detailed.csv', index=False)
    print(f"   ✅ Saved: shap_values_detailed.csv")

    return global_importance


# ============================================================================
# PART 5: INTERACTIVE RISK MAPS (PLOTLY)
# ============================================================================

def create_risk_maps(df):
    """
    Create interactive risk maps using Plotly
    """
    print("\n" + "="*80)
    print("PART 5: INTERACTIVE RISK MAPS")
    print("="*80)

    print("\n🗺️  Creating interactive maps...")

    # Check coordinates
    if 'KOORDINAT_X' not in df.columns or 'KOORDINAT_Y' not in df.columns:
        print("   ⚠️  No coordinates available")
        return

    df_map = df[df['KOORDINAT_Y'].notna() & df['KOORDINAT_X'].notna()].copy()
    df_map['lat'] = df_map['KOORDINAT_Y']
    df_map['lon'] = df_map['KOORDINAT_X']

    print(f"   - Equipment with coordinates: {len(df_map):,}")

    risk_colors = {
        'High Risk': '#E63946',
        'Medium Risk': '#F4A261',
        'Low Risk': '#2A9D8F'
    }

    # Map 1: Overall Risk
    print("\n📍 Map 1: Overall Risk Distribution...")
    fig = px.scatter_mapbox(
        df_map,
        lat='lat',
        lon='lon',
        color='Risk_Category_Final',
        color_discrete_map=risk_colors,
        size='PoF_Score_Final',
        hover_data={
            'Ekipman Kodu': True,
            'Ekipman Sınıfı': True,
            'İlçe': True,
            'Ekipman_Yaşı_Yıl': ':.1f',
            'PoF_Score_Final': ':.3f',
            'Health_Score': ':.1f',
            'lat': False,
            'lon': False
        },
        zoom=9,
        height=700,
        title='Equipment Failure Risk Map - Overall'
    )
    fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 40, "l": 0, "b": 0})
    fig.write_html(OUTPUT_DIR / 'risk_map_overall.html')
    print(f"   ✅ Saved: risk_map_overall.html")

    # Map 2: High Risk by District
    print("\n📍 Map 2: High Risk by District...")
    df_high_risk = df_map[df_map['Risk_Category_Final'] == 'High Risk'].copy()

    if len(df_high_risk) > 0:
        fig = px.scatter_mapbox(
            df_high_risk,
            lat='lat',
            lon='lon',
            color='İlçe',
            size='PoF_Score_Final',
            hover_data={
                'Ekipman Kodu': True,
                'Ekipman Sınıfı': True,
                'Ekipman_Yaşı_Yıl': ':.1f',
                'PoF_Score_Final': ':.3f',
                'Health_Score': ':.1f',
                'lat': False,
                'lon': False
            },
            zoom=9,
            height=700,
            title='High Risk Equipment by District'
        )
        fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 40, "l": 0, "b": 0})
        fig.write_html(OUTPUT_DIR / 'risk_map_high_risk.html')
        print(f"   ✅ Saved: risk_map_high_risk.html")

    # Map 3: Age Distribution
    print("\n📍 Map 3: Equipment Age Distribution...")
    fig = px.scatter_mapbox(
        df_map,
        lat='lat',
        lon='lon',
        color='Ekipman_Yaşı_Yıl',
        color_continuous_scale='RdYlGn_r',
        size='Ekipman_Yaşı_Yıl',
        hover_data={
            'Ekipman Kodu': True,
            'Ekipman Sınıfı': True,
            'İlçe': True,
            'Risk_Category_Final': True,
            'Health_Score': ':.1f',
            'lat': False,
            'lon': False
        },
        zoom=9,
        height=700,
        title='Equipment Age Distribution'
    )
    fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 40, "l": 0, "b": 0})
    fig.write_html(OUTPUT_DIR / 'risk_map_age.html')
    print(f"   ✅ Saved: risk_map_age.html")

    # Map 4: Health Score Heatmap
    print("\n📍 Map 4: Health Score Distribution...")
    fig = px.scatter_mapbox(
        df_map,
        lat='lat',
        lon='lon',
        color='Health_Score',
        color_continuous_scale='RdYlGn',
        size='PoF_Score_Final',
        hover_data={
            'Ekipman Kodu': True,
            'Ekipman Sınıfı': True,
            'İlçe': True,
            'Health_Status': True,
            'Risk_Category_Final': True,
            'lat': False,
            'lon': False
        },
        zoom=9,
        height=700,
        title='Equipment Health Score Distribution'
    )
    fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 40, "l": 0, "b": 0})
    fig.write_html(OUTPUT_DIR / 'risk_map_health.html')
    print(f"   ✅ Saved: risk_map_health.html")

    print("\n✅ All interactive maps created!")


# ============================================================================
# PART 6: CAPEX PRIORITIZATION (5-TIER SYSTEM)
# ============================================================================

def capex_prioritization(df):
    """
    5-tier CAPEX prioritization framework
    """
    print("\n" + "="*80)
    print("PART 6: CAPEX PRIORITIZATION (5-TIER SYSTEM)")
    print("="*80)

    print("\n💰 Generating CAPEX priorities...")

    def categorize_capex(row):
        pof = row['PoF_Score_Final']
        health = row['Health_Score']

        if pof > 0.7 and health < 30:
            return 'IMMEDIATE_REPLACEMENT'
        elif pof > 0.6 and health < 40:
            return 'HIGH_PRIORITY'
        elif pof > 0.5 or health < 50:
            return 'MEDIUM_PRIORITY'
        elif pof > 0.4 or health < 60:
            return 'LOW_PRIORITY'
        else:
            return 'MONITOR_ONLY'

    df['CAPEX_Priority'] = df.apply(categorize_capex, axis=1)

    print("📊 CAPEX Priority Distribution:")
    for priority, count in df['CAPEX_Priority'].value_counts().items():
        print(f"   {priority:25s}: {count:6,} ({count/len(df)*100:5.1f}%)")

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Priority distribution
    ax = axes[0, 0]
    priority_order = ['IMMEDIATE_REPLACEMENT', 'HIGH_PRIORITY', 'MEDIUM_PRIORITY',
                     'LOW_PRIORITY', 'MONITOR_ONLY']
    priority_counts = df['CAPEX_Priority'].value_counts().reindex(priority_order, fill_value=0)
    colors = ['#E63946', '#F4A261', '#F1C453', '#A8DADC', '#2A9D8F']
    priority_counts.plot(kind='bar', ax=ax, color=colors)
    ax.set_title('CAPEX Priority Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Priority Tier')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    for i, v in enumerate(priority_counts.values):
        ax.text(i, v + 20, str(v), ha='center', fontweight='bold')

    # Priority by district
    ax = axes[0, 1]
    district_priority = df.groupby(['İlçe', 'CAPEX_Priority']).size().unstack(fill_value=0)
    top_districts = df['İlçe'].value_counts().head(5).index
    district_priority.loc[top_districts].plot(kind='bar', stacked=True, ax=ax, color=colors)
    ax.set_title('Priority by Top 5 Districts', fontsize=14, fontweight='bold')
    ax.set_xlabel('District')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Priority', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Priority vs Age
    ax = axes[1, 0]
    for priority in priority_order:
        mask = df['CAPEX_Priority'] == priority
        if mask.sum() > 0:
            ax.scatter(df.loc[mask, 'Ekipman_Yaşı_Yıl'],
                      df.loc[mask, 'PoF_Score_Final'],
                      label=priority, alpha=0.5, s=30)
    ax.set_xlabel('Equipment Age (years)')
    ax.set_ylabel('PoF Score')
    ax.set_title('Priority vs Age & PoF', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Priority vs Health
    ax = axes[1, 1]
    for priority in priority_order:
        mask = df['CAPEX_Priority'] == priority
        if mask.sum() > 0:
            ax.scatter(df.loc[mask, 'Health_Score'],
                      df.loc[mask, 'PoF_Score_Final'],
                      label=priority, alpha=0.5, s=30)
    ax.set_xlabel('Health Score')
    ax.set_ylabel('PoF Score')
    ax.set_title('Priority vs Health & PoF', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'capex_prioritization.png', dpi=300, bbox_inches='tight')
    print(f"\n   ✅ Saved: capex_prioritization.png")
    plt.close()

    # Export prioritized list
    high_priority = df[df['CAPEX_Priority'].isin(['IMMEDIATE_REPLACEMENT', 'HIGH_PRIORITY'])].copy()
    high_priority = high_priority.sort_values('PoF_Score_Final', ascending=False)

    export_cols = ['Ekipman Kodu', 'Ekipman Sınıfı', 'İlçe', 'Ekipman_Yaşı_Yıl',
                   'PoF_Score_Final', 'Health_Score', 'Health_Status', 'CAPEX_Priority']
    if 'KOORDINAT_X' in df.columns:
        export_cols.extend(['KOORDINAT_X', 'KOORDINAT_Y'])

    high_priority[export_cols].to_excel(OUTPUT_DIR / 'capex_high_priority_list.xlsx', index=False)
    print(f"   ✅ Saved: capex_high_priority_list.xlsx")
    print(f"   High-priority equipment: {len(high_priority):,}")

    return df


# ============================================================================
# PART 7: CALIBRATION ANALYSIS
# ============================================================================

def calibration_analysis(df, model, feature_cols, scaler):
    """
    Analyze probability calibration
    """
    print("\n" + "="*80)
    print("PART 7: CALIBRATION ANALYSIS")
    print("="*80)

    print("\n📊 Analyzing probability calibration...")

    # Prepare data
    X = df[feature_cols]
    y = df['PoF_12_month']

    # Scale - use ALL features the scaler was trained on
    X_scaled = safe_scale_features(scaler, X, feature_cols)

    # Predictions
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]

    # Brier score
    brier_score = brier_score_loss(y, y_pred_proba)
    print(f"   - Brier Score: {brier_score:.4f} (lower is better)")

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=10, strategy='uniform')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Calibration curve
    ax = axes[0]
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label='XGBoost', markersize=8)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('True Probability', fontsize=12)
    ax.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, f'Brier Score: {brier_score:.4f}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Prediction distribution
    ax = axes[1]
    ax.hist(y_pred_proba[y == 0], bins=30, alpha=0.5, label='No Fault', color='green')
    ax.hist(y_pred_proba[y == 1], bins=30, alpha=0.5, label='Fault', color='red')
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'calibration_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: calibration_analysis.png")

    # Export calibration data
    calibration_df = pd.DataFrame({
        'Predicted_Probability': prob_pred,
        'True_Probability': prob_true
    })
    calibration_df.to_excel(OUTPUT_DIR / 'calibration_data.xlsx', index=False)
    print(f"   ✅ Saved: calibration_data.xlsx")

    return brier_score


# ============================================================================
# PART 8: CATBOOST COMPARISON
# ============================================================================

def catboost_comparison(df, feature_cols, scaler):
    """
    Compare XGBoost with CatBoost
    """
    print("\n" + "="*80)
    print("PART 8: CATBOOST ALGORITHM COMPARISON")
    print("="*80)

    try:
        from catboost import CatBoostClassifier
        print("\n✅ CatBoost loaded")
    except ImportError:
        print("\n⚠️  Installing CatBoost...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'catboost', '--break-system-packages', '-q'])
        from catboost import CatBoostClassifier
        print("✅ CatBoost installed")

    print("\n🔄 Training CatBoost...")

    # Prepare data
    X = df[feature_cols]
    y = df['PoF_12_month']

    # Scale - use ALL features the scaler was trained on
    X_scaled = safe_scale_features(scaler, X, feature_cols)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Train CatBoost
    catboost_model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.05,
        depth=5,
        l2_leaf_reg=3,
        random_seed=RANDOM_STATE,
        verbose=False
    )

    catboost_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)

    # Predictions
    y_pred_cat = catboost_model.predict_proba(X_test)[:, 1]
    auc_catboost = roc_auc_score(y_test, y_pred_cat)

    # Load XGBoost for comparison
    model = joblib.load(MODEL_FILE)
    y_pred_xgb = model.predict_proba(X_test)[:, 1]
    auc_xgboost = roc_auc_score(y_test, y_pred_xgb)

    print(f"\n   ✅ CatBoost AUC: {auc_catboost:.4f}")
    print(f"   ✅ XGBoost AUC: {auc_xgboost:.4f}")
    print(f"   Difference: {abs(auc_catboost - auc_xgboost):.4f}")

    # ROC comparison
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_xgb)
    fpr_cat, tpr_cat, _ = roc_curve(y_test, y_pred_cat)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgboost:.4f})', linewidth=2)
    plt.plot(fpr_cat, tpr_cat, label=f'CatBoost (AUC = {auc_catboost:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve: XGBoost vs CatBoost', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'catboost_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n   ✅ Saved: catboost_comparison.png")

    # Feature importance
    cat_importance = catboost_model.get_feature_importance()
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'CatBoost_Importance': cat_importance
    }).sort_values('CatBoost_Importance', ascending=False)

    print("\n📊 CatBoost Top 10 Features:")
    print(importance_df.head(10).to_string(index=False))

    # Export comparison
    comparison_df = pd.DataFrame({
        'Model': ['XGBoost', 'CatBoost'],
        'AUC': [auc_xgboost, auc_catboost],
        'Difference': [0, auc_catboost - auc_xgboost]
    })
    comparison_df.to_excel(OUTPUT_DIR / 'model_comparison.xlsx', index=False)
    print(f"\n   ✅ Saved: model_comparison.xlsx")

    return auc_catboost


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    # Load data
    df, model, scaler, best_features = load_and_validate_data()

    if df is None:
        print("\n❌ Cannot proceed without required files")
        exit(1)

    print(f"\n✅ Using {len(best_features)} features:")
    for i, feat in enumerate(best_features, 1):
        print(f"   {i:2d}. {feat}")

    # Execute all parts
    try:
        # Part 1: Survival Analysis
        survival_data = survival_analysis(df)

        # Part 2: Backtesting
        backtest_results = backtesting_analysis(df, model, scaler, best_features)

        # Part 3: Health Scores
        df = calculate_health_scores(df)

        # Part 4: SHAP Analysis (FIXED)
        shap_importance = shap_analysis(df, model, best_features)

        # Part 5: Interactive Maps
        create_risk_maps(df)

        # Part 6: CAPEX Prioritization
        df = capex_prioritization(df)

        # Part 7: Calibration
        brier_score = calibration_analysis(df, model, best_features, scaler)

        # Part 8: CatBoost Comparison
        catboost_auc = catboost_comparison(df, best_features, scaler)

        # Save final dataset
        final_file = OUTPUT_DIR / 'comprehensive_analysis_final.xlsx'
        df.to_excel(final_file, index=False, engine='openpyxl')
        print(f"\n✅ Final dataset saved: {final_file}")

        # Save high-priority equipment
        high_priority = df[df['CAPEX_Priority'].isin(['IMMEDIATE_REPLACEMENT', 'HIGH_PRIORITY'])]
        priority_file = OUTPUT_DIR / 'high_priority_equipment.xlsx'
        high_priority.to_excel(priority_file, index=False, engine='openpyxl')
        print(f"✅ High-priority list saved: {priority_file}")

        # Generate summary report
        summary_file = OUTPUT_DIR / 'analysis_summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE ANALYTICS SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Equipment: {len(df):,}\n")
            f.write(f"Features Used: {len(best_features)}\n\n")

            f.write("HEALTH SCORES:\n")
            for status, count in df['Health_Status'].value_counts().items():
                f.write(f"  {status}: {count:,} ({count/len(df)*100:.1f}%)\n")

            f.write(f"\nCAPEX PRIORITIES:\n")
            for priority, count in df['CAPEX_Priority'].value_counts().items():
                f.write(f"  {priority}: {count:,} ({count/len(df)*100:.1f}%)\n")

            if backtest_results is not None:
                f.write(f"\nBACKTESTING:\n")
                f.write(f"  Mean AUC: {backtest_results['AUC'].mean():.4f}\n")
                f.write(f"  Std AUC: {backtest_results['AUC'].std():.4f}\n")

            f.write(f"\nMODEL PERFORMANCE:\n")
            f.write(f"  Brier Score: {brier_score:.4f}\n")
            f.write(f"  CatBoost AUC: {catboost_auc:.4f}\n")

            f.write(f"\nTOP SHAP FEATURES:\n")
            for i, (feat, imp) in enumerate(shap_importance.head(5).items(), 1):
                f.write(f"  {i}. {feat.replace('SHAP_', '')}: {imp:.4f}\n")

        print(f"✅ Summary saved: {summary_file}")

        # Final summary
        print("\n" + "="*80)
        print("🎉 COMPREHENSIVE ANALYTICS COMPLETED!")
        print("="*80)

        print(f"\n📊 KEY STATISTICS:")
        print(f"   - Total Equipment: {len(df):,}")
        print(f"   - High Priority: {len(high_priority):,}")
        print(f"   - Immediate Replacement: {(df['CAPEX_Priority'] == 'IMMEDIATE_REPLACEMENT').sum():,}")
        print(f"   - Critical Health: {(df['Health_Status'] == 'Critical').sum():,}")

        if backtest_results is not None:
            print(f"   - Backtest Mean AUC: {backtest_results['AUC'].mean():.4f}")
        print(f"   - Brier Score: {brier_score:.4f}")
        print(f"   - CatBoost AUC: {catboost_auc:.4f}")

        print(f"\n📁 OUTPUT FILES ({OUTPUT_DIR}):")
        for i, file in enumerate(sorted(OUTPUT_DIR.glob('*')), 1):
            print(f"   {i:2d}. {file.name}")

        print("\n" + "="*80)
        print("✅ All analyses completed successfully!")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
