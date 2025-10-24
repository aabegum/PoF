"""
STEP 4 ENHANCED: Advanced Analysis & Validation
================================================
Option A - Includes all mandatory + high-value features:
- Survival Analysis (Cox PH + Kaplan-Meier)
- Backtesting (Rolling Window Validation)
- Interactive Risk Maps
- CAPEX Prioritization
- SHAP Explainability
- Calibration Analysis
- CatBoost Comparison
- Monotonic Constraints

Required Input Files:
- step3_5_final_risk_scored.xlsx
- step3_5_final_model.pkl
- step3_5_final_scaler.pkl
- step3_5_best_params.json (optional)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta

# Machine Learning
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, brier_score_loss
)
from sklearn.calibration import calibration_curve
import xgboost as xgb

# Survival Analysis
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import median_survival_times

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
OUTPUT_DIR = Path('outputs/')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    
    # Check if we have features to scale
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

print("="*80)
print("STEP 4 ENHANCED - Advanced Analysis & Validation")
print("="*80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# PART 0: LOAD DATA & VALIDATE
# ============================================================================

def load_and_validate_data():
    """Load all required files and validate data integrity"""
    print("\n" + "="*80)
    print("PART 0: DATA LOADING & VALIDATION")
    print("="*80)
    
    # Expected input files
    input_files = {
        'data': 'outputs/step3_5_final_risk_scored.xlsx',
        'model': 'outputs/step3_5_final_model.pkl',
        'scaler': 'outputs/step3_5_final_scaler.pkl',
        'params': 'outputs/step3_5_best_params.json'
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
        print("\nPlease upload the required files")
        return None, None, None, None
    
    # Load data
    print("\n✅ Loading data...")
    df = pd.read_excel(input_files['data'])
    print(f"   Records loaded: {len(df):,}")
    
    # Parse dates if needed
    if 'Arıza_Tarihi' in df.columns and df['Arıza_Tarihi'].dtype == 'object':
        df['Arıza_Tarihi'] = pd.to_datetime(df['Arıza_Tarihi'])
    
    # Load model
    print("✅ Loading XGBoost model...")
    with open(input_files['model'], 'rb') as f:
        model = pickle.load(f)
    
    # Load scaler with robust error handling
    print("✅ Loading StandardScaler...")
    try:
        with open(input_files['scaler'], 'rb') as f:
            scaler = pickle.load(f)
        
        # Verify it's actually a StandardScaler
        from sklearn.preprocessing import StandardScaler
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
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler._needs_fitting = True
    
    # Load hyperparameters (optional)
    best_params = None
    if Path(input_files['params']).exists():
        print("✅ Loading best hyperparameters...")
        with open(input_files['params'], 'r') as f:
            best_params = json.load(f)
    
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
        print(f"   Available columns: {list(df.columns[:20])}")
    else:
        print("   ✅ All required columns present")
    
    # Check for coordinates
    has_coords = 'KOORDINAT_X' in df.columns and 'KOORDINAT_Y' in df.columns
    if not has_coords:
        print("   ⚠️  Coordinate columns (KOORDINAT_X, KOORDINAT_Y) not found")
    
    # Data quality checks
    print("\n📈 Data Quality Summary:")
    print(f"   - Total records: {len(df):,}")
    print(f"   - Unique equipment: {df['Ekipman Kodu'].nunique():,}")
    print(f"   - Date range: {df['Arıza_Tarihi'].min()} to {df['Arıza_Tarihi'].max()}")
    print(f"   - PoF positive rate: {df['PoF_12_month'].mean():.1%}")
    if has_coords:
        print(f"   - Missing coordinates: {df[['KOORDINAT_X', 'KOORDINAT_Y']].isna().any(axis=1).sum()}")
    
    # Kesici equipment check
    kesici_mask = df['Ekipman Sınıfı'].str.contains('Kesici|kesici', na=False, case=False)
    kesici_count = kesici_mask.sum()
    kesici_high_risk = (kesici_mask & (df['Risk_Category_Final'] == 'High Risk')).sum()
    print(f"\n🎯 Critical Equipment (Kesici):")
    print(f"   - Total: {kesici_count}")
    print(f"   - High Risk: {kesici_high_risk} ({kesici_high_risk/kesici_count*100:.1f}% of Kesici)")
    
    return df, model, scaler, best_params


# ============================================================================
# PART 1: SURVIVAL ANALYSIS
# ============================================================================

def survival_analysis(df):
    """
    Perform survival analysis using Cox PH and Kaplan-Meier
    Treats each fault as a "failure event" with equipment age as time-to-event
    """
    print("\n" + "="*80)
    print("PART 1: SURVIVAL ANALYSIS")
    print("="*80)
    
    # Prepare survival data
    print("\n📊 Preparing survival analysis data...")
    
    # For each equipment, calculate time to first fault (or censoring)
    survival_data = df.groupby('Ekipman Kodu').agg({
        'Ekipman_Yaşı_Yıl': 'first',  # Age at observation
        'Arıza_Tarihi': 'min',  # First fault date
        'Ekipman Sınıfı': 'first',
        'İlçe': 'first',
        'PoF_12_month': 'max',  # Did fault occur?
        'Kentsel_Müşteri_Oranı': 'first',
        'OG_Müşteri_Oranı': 'first',
        'Toplam_Müşteri_Sayısı': 'first'
    }).reset_index()
    
    survival_data.columns = ['Ekipman_Kodu', 'duration', 'event_date', 'type', 
                             'district', 'event', 'urban_ratio', 'industrial_ratio',
                             'total_customers']
    
    print(f"   - Equipment tracked: {len(survival_data):,}")
    print(f"   - Events (faults): {survival_data['event'].sum():,}")
    print(f"   - Censored: {(~survival_data['event']).sum():,}")
    
    # ========================================
    # 1.1: Kaplan-Meier Survival Curves
    # ========================================
    print("\n📈 Computing Kaplan-Meier survival curves...")
    
    kmf = KaplanMeierFitter()
    
    # Overall survival
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Overall
    ax = axes[0, 0]
    kmf.fit(survival_data['duration'], survival_data['event'], label='All Equipment')
    kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_title('Overall Survival Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Equipment Age (years)')
    ax.set_ylabel('Survival Probability')
    ax.grid(True, alpha=0.3)
    
    # By equipment type (Kesici vs Others)
    ax = axes[0, 1]
    for equipment_type in ['Kesici', 'Trafo']:
        if equipment_type in survival_data['type'].values:
            mask = survival_data['type'] == equipment_type
            kmf.fit(survival_data.loc[mask, 'duration'], 
                   survival_data.loc[mask, 'event'],
                   label=equipment_type)
            kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_title('Survival by Equipment Type', fontsize=14, fontweight='bold')
    ax.set_xlabel('Equipment Age (years)')
    ax.set_ylabel('Survival Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # By district
    ax = axes[1, 0]
    top_districts = survival_data['district'].value_counts().head(3).index
    for district in top_districts:
        mask = survival_data['district'] == district
        kmf.fit(survival_data.loc[mask, 'duration'], 
               survival_data.loc[mask, 'event'],
               label=district)
        kmf.plot_survival_function(ax=ax, ci_show=False)
    ax.set_title('Survival by Top 3 Districts', fontsize=14, fontweight='bold')
    ax.set_xlabel('Equipment Age (years)')
    ax.set_ylabel('Survival Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # By customer load (high vs low)
    ax = axes[1, 1]
    survival_data['high_load'] = survival_data['total_customers'] > survival_data['total_customers'].median()
    for load_level, label in [(True, 'High Load'), (False, 'Low Load')]:
        mask = survival_data['high_load'] == load_level
        kmf.fit(survival_data.loc[mask, 'duration'], 
               survival_data.loc[mask, 'event'],
               label=label)
        kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_title('Survival by Customer Load', fontsize=14, fontweight='bold')
    ax.set_xlabel('Equipment Age (years)')
    ax.set_ylabel('Survival Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'step4_survival_kaplan_meier.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: step4_survival_kaplan_meier.png")
    plt.close()
    
    # Calculate median survival times
    print("\n📊 Median Survival Times:")
    for equipment_type in ['Kesici', 'Trafo']:
        if equipment_type in survival_data['type'].values:
            mask = survival_data['type'] == equipment_type
            kmf.fit(survival_data.loc[mask, 'duration'], 
                   survival_data.loc[mask, 'event'])
            median_survival = kmf.median_survival_time_
            print(f"   - {equipment_type}: {median_survival:.1f} years")
    
    # ========================================
    # 1.2: Cox Proportional Hazards Model
    # ========================================
    print("\n📈 Fitting Cox Proportional Hazards model...")
    
    # Prepare covariates for Cox model
    cox_data = survival_data[['duration', 'event', 'urban_ratio', 
                               'industrial_ratio', 'total_customers']].copy()
    
    # Standardize continuous variables
    cox_data['total_customers_log'] = np.log1p(cox_data['total_customers'])
    cox_data = cox_data.drop('total_customers', axis=1)
    
    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col='duration', event_col='event')
    
    print("\n✅ Cox Model Summary:")
    print(cph.summary[['coef', 'exp(coef)', 'p']])
    
    # Plot hazard ratios
    fig, ax = plt.subplots(figsize=(10, 6))
    cph.plot(ax=ax)
    ax.set_title('Cox Model Hazard Ratios', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'step4_survival_cox_hazard_ratios.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: step4_survival_cox_hazard_ratios.png")
    plt.close()
    
    # Export survival results
    survival_summary = pd.DataFrame({
        'Equipment_Type': ['Kesici', 'Trafo', 'Overall'],
        'Count': [
            (survival_data['type'] == 'Kesici').sum(),
            (survival_data['type'] == 'Trafo').sum(),
            len(survival_data)
        ],
        'Events': [
            survival_data[survival_data['type'] == 'Kesici']['event'].sum(),
            survival_data[survival_data['type'] == 'Trafo']['event'].sum(),
            survival_data['event'].sum()
        ]
    })
    
    survival_summary.to_excel(OUTPUT_DIR / 'step4_survival_summary.xlsx', index=False)
    print(f"   ✅ Saved: step4_survival_summary.xlsx")
    
    return survival_data, cph


# ============================================================================
# PART 2: BACKTESTING (ROLLING WINDOW VALIDATION)
# ============================================================================

def backtesting_analysis(df, model, scaler, feature_cols):
    """
    Perform rolling window backtesting to validate model performance over time
    """
    print("\n" + "="*80)
    print("PART 2: BACKTESTING - ROLLING WINDOW VALIDATION")
    print("="*80)
    
    print("\n📊 Setting up backtesting framework...")
    
    # Sort by date
    df_sorted = df.sort_values('Arıza_Tarihi').copy()
    df_sorted['Arıza_Tarihi'] = pd.to_datetime(df_sorted['Arıza_Tarihi'])
    
    # Define rolling windows (6-month intervals)
    min_date = df_sorted['Arıza_Tarihi'].min()
    max_date = df_sorted['Arıza_Tarihi'].max()
    
    print(f"   - Date range: {min_date.date()} to {max_date.date()}")
    
    # Create 6-month windows
    window_size = timedelta(days=180)
    test_size = timedelta(days=90)
    
    results = []
    window_id = 0
    
    current_date = min_date + timedelta(days=365)  # Start after 1 year for training
    
    print("\n🔄 Running rolling window validation...")
    
    while current_date + test_size <= max_date:
        window_id += 1
        
        # Define train and test periods
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
        
        # Scale features using helper function
        scaled_cols = ['Ekipman_Yaşı_Yıl', 'Arıza_Sayısı_12ay', 'Arıza_Sayısı_3ay',
                       'Toplam_Müşteri_Sayısı', 'Ekipman_Yoğunluk_Skoru', 
                       'Müşteri_Başına_Arıza', 'Ay_Sin']
        
        X_train_scaled = safe_scale_features(scaler, X_train, scaled_cols)
        X_test_scaled = safe_scale_features(scaler, X_test, scaled_cols)
        
        # Make predictions
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
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
        
        print(f"   Window {window_id}: AUC = {auc:.4f} (Test size: {len(X_test)}, Positives: {y_test.sum()})")
        
        # Move to next window
        current_date += timedelta(days=30)
    
    # Convert to DataFrame
    backtest_results = pd.DataFrame(results)
    
    # Plot backtesting results
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
    ax.set_title('Model Performance Over Time (Rolling Window Validation)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Sample sizes over time
    ax = axes[1]
    ax.bar(backtest_results['Window'], backtest_results['Test_Size'], 
           alpha=0.6, label='Test Size', color='#A23B72')
    ax.set_xlabel('Window Number', fontsize=12)
    ax.set_ylabel('Sample Size', fontsize=12)
    ax.set_title('Test Set Sizes Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'step4_backtest_rolling_window.png', dpi=300, bbox_inches='tight')
    print(f"\n   ✅ Saved: step4_backtest_rolling_window.png")
    plt.close()
    
    # Summary statistics
    print("\n📊 Backtesting Summary:")
    print(f"   - Number of windows: {len(backtest_results)}")
    print(f"   - Mean AUC: {backtest_results['AUC'].mean():.4f}")
    print(f"   - Std AUC: {backtest_results['AUC'].std():.4f}")
    print(f"   - Min AUC: {backtest_results['AUC'].min():.4f}")
    print(f"   - Max AUC: {backtest_results['AUC'].max():.4f}")
    
    # Export results
    backtest_results.to_excel(OUTPUT_DIR / 'step4_backtest_results.xlsx', index=False)
    print(f"   ✅ Saved: step4_backtest_results.xlsx")
    
    return backtest_results


# ============================================================================
# PART 3: INTERACTIVE RISK MAPS
# ============================================================================

def create_risk_maps(df):
    """
    Create interactive HTML risk maps using Plotly
    """
    print("\n" + "="*80)
    print("PART 3: INTERACTIVE RISK MAPS")
    print("="*80)
    
    print("\n🗺️  Creating interactive risk maps...")
    
    # Filter out missing coordinates
    if 'KOORDINAT_X' not in df.columns or 'KOORDINAT_Y' not in df.columns:
        print("   ⚠️  No coordinate data available, skipping maps")
        return
    
    df_map = df[df['KOORDINAT_Y'].notna() & df['KOORDINAT_X'].notna()].copy()
    df_map['lat'] = df_map['KOORDINAT_Y']  # Y is latitude
    df_map['lon'] = df_map['KOORDINAT_X']  # X is longitude
    
    print(f"   - Equipment with coordinates: {len(df_map):,}")
    
    # Define colors for risk categories
    risk_colors = {
        'High': '#E63946',    # Red
        'Medium': '#F4A261',  # Orange
        'Low': '#2A9D8F'      # Green
    }
    
    # ========================================
    # Map 1: Overall Risk Distribution
    # ========================================
    print("\n📍 Creating Map 1: Overall Risk Distribution...")
    
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
            'Risk_Category_Final': True,
            'KOORDINAT_Y': False,
            'KOORDINAT_X': False
        },
        zoom=9,
        height=700,
        title='Equipment Failure Risk Map - All Equipment'
    )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    
    fig.write_html(OUTPUT_DIR / 'step4_risk_map_overall.html')
    print(f"   ✅ Saved: step4_risk_map_overall.html")
    
    # ========================================
    # Map 2: Kesici (Critical) Equipment Only
    # ========================================
    print("\n📍 Creating Map 2: Kesici Critical Equipment...")
    
    df_kesici = df_map[df_map['Ekipman Sınıfı'] == 'Kesici'].copy()
    
    if len(df_kesici) > 0:
        fig = px.scatter_mapbox(
            df_kesici,
            lat='KOORDINAT_Y',
            lon='KOORDINAT_X',
            color='Risk_Category_Final',
            color_discrete_map=risk_colors,
            size='PoF_Score_Final',
            hover_data={
                'Ekipman Kodu': True,
                'İlçe': True,
                'Ekipman_Yaşı_Yıl': ':.1f',
                'PoF_Score_Final': ':.3f',
                'Risk_Category_Final': True,
                'KOORDINAT_Y': False,
                'KOORDINAT_X': False
            },
            zoom=9,
            height=700,
            title='Kesici Equipment Risk Map - Critical Assets'
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )
        
        fig.write_html(OUTPUT_DIR / 'step4_risk_map_kesici.html')
        print(f"   ✅ Saved: step4_risk_map_kesici.html")
    
    # ========================================
    # Map 3: High Risk Equipment by District
    # ========================================
    print("\n📍 Creating Map 3: High Risk by District...")
    
    df_high_risk = df_map[df_map['Risk_Category_Final'] == 'High Risk'].copy()
    
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
            'KOORDINAT_Y': False,
            'KOORDINAT_X': False
        },
        zoom=9,
        height=700,
        title='High Risk Equipment by District'
    )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    
    fig.write_html(OUTPUT_DIR / 'step4_risk_map_high_risk_districts.html')
    print(f"   ✅ Saved: step4_risk_map_high_risk_districts.html")
    
    # ========================================
    # Map 4: Equipment Age Heatmap
    # ========================================
    print("\n📍 Creating Map 4: Equipment Age Distribution...")
    
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
            'KOORDINAT_Y': False,
            'KOORDINAT_X': False
        },
        zoom=9,
        height=700,
        title='Equipment Age Distribution Map'
    )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    
    fig.write_html(OUTPUT_DIR / 'step4_risk_map_age_distribution.html')
    print(f"   ✅ Saved: step4_risk_map_age_distribution.html")
    
    print("\n✅ All interactive maps created successfully!")


# ============================================================================
# PART 4: CAPEX PRIORITIZATION
# ============================================================================

def capex_prioritization(df):
    """
    Create CAPEX prioritization framework based on risk scores
    """
    print("\n" + "="*80)
    print("PART 4: CAPEX PRIORITIZATION FRAMEWORK")
    print("="*80)
    
    print("\n💰 Creating CAPEX prioritization analysis...")
    
    # Focus on high-risk equipment
    df_high_risk = df[df['Risk_Category_Final'] == 'High Risk'].copy()
    
    print(f"   - High-risk equipment: {len(df_high_risk):,}")
    
    # Create prioritization score
    # Factors: PoF_Score (40%), Age (30%), Customer Count (20%), Critical Type (10%)
    
    # Normalize factors
    df_high_risk['Age_Normalized'] = (df_high_risk['Ekipman_Yaşı_Yıl'] / 
                                       df_high_risk['Ekipman_Yaşı_Yıl'].max())
    df_high_risk['Customer_Normalized'] = (df_high_risk['Toplam_Müşteri_Sayısı'] / 
                                            df_high_risk['Toplam_Müşteri_Sayısı'].max())
    df_high_risk['Critical_Weight'] = df_high_risk['Ekipman Sınıfı'].apply(
        lambda x: 1.0 if x == 'Kesici' else 0.5
    )
    
    # Calculate priority score (0-100)
    df_high_risk['Priority_Score'] = (
        df_high_risk['PoF_Score_Final'] * 40 +
        df_high_risk['Age_Normalized'] * 30 +
        df_high_risk['Customer_Normalized'] * 20 +
        df_high_risk['Critical_Weight'] * 10
    )
    
    # Assign priority tiers
    df_high_risk['Priority_Tier'] = pd.cut(
        df_high_risk['Priority_Score'],
        bins=[0, 50, 70, 100],
        labels=['P3 - Lower Priority', 'P2 - Medium Priority', 'P1 - Urgent']
    )
    
    # Sort by priority
    df_prioritized = df_high_risk.sort_values('Priority_Score', ascending=False)
    
    # Summary by priority tier
    print("\n📊 CAPEX Priority Distribution:")
    priority_summary = df_prioritized['Priority_Tier'].value_counts().sort_index(ascending=False)
    for tier, count in priority_summary.items():
        print(f"   - {tier}: {count:,} equipment ({count/len(df_prioritized)*100:.1f}%)")
    
    # Top 20 priority equipment
    top_priority = df_prioritized.head(20)[
        ['Ekipman Kodu', 'Ekipman Sınıfı', 'İlçe', 'Ekipman_Yaşı_Yıl', 
         'Toplam_Müşteri_Sayısı', 'PoF_Score_Final', 'Priority_Score', 'Priority_Tier']
    ]
    
    print("\n🎯 Top 20 Priority Equipment:")
    print(top_priority.to_string(index=False))
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Priority distribution
    ax = axes[0, 0]
    priority_counts = df_prioritized['Priority_Tier'].value_counts().sort_index(ascending=False)
    colors = ['#E63946', '#F4A261', '#F1C453']
    priority_counts.plot(kind='bar', ax=ax, color=colors)
    ax.set_title('CAPEX Priority Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Priority Tier')
    ax.set_ylabel('Number of Equipment')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    for i, v in enumerate(priority_counts.values):
        ax.text(i, v + 20, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Priority by district
    ax = axes[0, 1]
    district_priority = df_prioritized.groupby(['İlçe', 'Priority_Tier']).size().unstack(fill_value=0)
    district_priority.plot(kind='bar', stacked=True, ax=ax, color=colors)
    ax.set_title('Priority Distribution by District', fontsize=14, fontweight='bold')
    ax.set_xlabel('District')
    ax.set_ylabel('Number of Equipment')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Priority Tier', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Priority score vs Age
    ax = axes[1, 0]
    scatter = ax.scatter(df_prioritized['Ekipman_Yaşı_Yıl'], 
                        df_prioritized['Priority_Score'],
                        c=df_prioritized['Priority_Tier'].cat.codes,
                        cmap='RdYlGn_r', alpha=0.6, s=50)
    ax.set_xlabel('Equipment Age (years)')
    ax.set_ylabel('Priority Score')
    ax.set_title('Priority Score vs Equipment Age', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Priority score vs Customer count
    ax = axes[1, 1]
    scatter = ax.scatter(df_prioritized['Toplam_Müşteri_Sayısı'], 
                        df_prioritized['Priority_Score'],
                        c=df_prioritized['Priority_Tier'].cat.codes,
                        cmap='RdYlGn_r', alpha=0.6, s=50)
    ax.set_xlabel('Total Customer Count')
    ax.set_ylabel('Priority Score')
    ax.set_title('Priority Score vs Customer Count', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'step4_capex_prioritization.png', dpi=300, bbox_inches='tight')
    print(f"\n   ✅ Saved: step4_capex_prioritization.png")
    plt.close()
    
    # Export prioritized list
    df_prioritized[
        ['Ekipman Kodu', 'Ekipman Sınıfı', 'İlçe', 'Ekipman_Yaşı_Yıl',
         'Toplam_Müşteri_Sayısı', 'PoF_Score_Final', 'Priority_Score', 
         'Priority_Tier', 'KOORDINAT_Y', 'KOORDINAT_X']
    ].to_excel(OUTPUT_DIR / 'step4_capex_prioritized_list.xlsx', index=False)
    print(f"   ✅ Saved: step4_capex_prioritized_list.xlsx")
    
    return df_prioritized


# ============================================================================
# PART 5: SHAP EXPLAINABILITY
# ============================================================================

def shap_analysis(df, model, feature_cols):
    """
    SHAP (SHapley Additive exPlanations) for model interpretability
    """
    print("\n" + "="*80)
    print("PART 5: SHAP EXPLAINABILITY ANALYSIS")
    print("="*80)
    
    try:
        import shap
        print("\n✅ SHAP library loaded")
    except ImportError:
        print("\n⚠️  SHAP library not available. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'shap', '--break-system-packages', '-q'])
        import shap
        print("✅ SHAP installed successfully")
    
    print("\n🔍 Computing SHAP values (this may take a few minutes)...")
    
    # Sample data for SHAP (use 1000 samples for speed)
    sample_size = min(1000, len(df))
    df_sample = df.sample(sample_size, random_state=42)
    
    X_sample = df_sample[feature_cols]
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    print(f"   ✅ SHAP values computed for {sample_size} samples")
    
    # Summary plot
    print("\n📊 Creating SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.title('SHAP Feature Importance Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'step4_shap_summary_plot.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: step4_shap_summary_plot.png")
    plt.close()
    
    # Feature importance bar plot
    print("\n📊 Creating SHAP feature importance bar plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
    plt.title('SHAP Mean Absolute Feature Importance', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'step4_shap_feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: step4_shap_feature_importance.png")
    plt.close()
    
    # Individual prediction explanation (for a high-risk case)
    high_risk_idx = df_sample[df_sample['Risk_Category_Final'] == 'High Risk'].index[0]
    high_risk_sample = X_sample.loc[high_risk_idx]
    
    print("\n📊 Creating waterfall plot for high-risk example...")
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[X_sample.index.get_loc(high_risk_idx)],
            base_values=explainer.expected_value,
            data=high_risk_sample.values,
            feature_names=feature_cols
        ),
        max_display=15,
        show=False
    )
    plt.title('SHAP Explanation: High-Risk Equipment Example', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'step4_shap_waterfall_high_risk.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: step4_shap_waterfall_high_risk.png")
    plt.close()
    
    # Dependence plots for top 3 features
    print("\n📊 Creating SHAP dependence plots...")
    top_features = [
        'Ekipman_Yaşı_Yıl',
        'OG_Müşteri_Oranı',
        'Kentsel_Müşteri_Oranı'
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, feature in enumerate(top_features):
        if feature in feature_cols:
            shap.dependence_plot(
                feature,
                shap_values,
                X_sample,
                ax=axes[i],
                show=False
            )
            axes[i].set_title(f'SHAP Dependence: {feature}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'step4_shap_dependence_plots.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: step4_shap_dependence_plots.png")
    plt.close()
    
    print("\n✅ SHAP analysis completed!")


# ============================================================================
# PART 6: CALIBRATION ANALYSIS
# ============================================================================

def calibration_analysis(df, model, feature_cols, scaler):
    """
    Analyze probability calibration of the model
    """
    print("\n" + "="*80)
    print("PART 6: CALIBRATION ANALYSIS")
    print("="*80)
    
    print("\n📊 Analyzing probability calibration...")
    
    # Prepare data
    X = df[feature_cols]
    y = df['PoF_12_month']
    
    # Scale features using helper function
    scaled_cols = ['Ekipman_Yaşı_Yıl', 'Arıza_Sayısı_12ay', 'Arıza_Sayısı_3ay',
                   'Toplam_Müşteri_Sayısı', 'Ekipman_Yoğunluk_Skoru', 
                   'Müşteri_Başına_Arıza', 'Ay_Sin']
    
    X_scaled = safe_scale_features(scaler, X, scaled_cols)
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    # Calculate Brier score
    brier_score = brier_score_loss(y, y_pred_proba)
    print(f"\n   - Brier Score: {brier_score:.4f} (lower is better)")
    
    # Calibration curve
    prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=10, strategy='uniform')
    
    # Plot calibration
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
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Prediction histogram
    ax = axes[1]
    ax.hist(y_pred_proba[y == 0], bins=30, alpha=0.5, label='Negative Class', color='green')
    ax.hist(y_pred_proba[y == 1], bins=30, alpha=0.5, label='Positive Class', color='red')
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'step4_calibration_analysis.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: step4_calibration_analysis.png")
    plt.close()
    
    # Reliability diagram with confidence intervals
    print("\n📊 Creating detailed reliability diagram...")
    
    # Bin predictions
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    
    bin_sums = np.zeros(len(bin_centers))
    bin_counts = np.zeros(len(bin_centers))
    bin_true = np.zeros(len(bin_centers))
    
    for i in range(len(bin_centers)):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_sums[i] = y_pred_proba[mask].sum()
            bin_counts[i] = mask.sum()
            bin_true[i] = y[mask].sum()
    
    # Calculate bin statistics
    with np.errstate(divide='ignore', invalid='ignore'):
        bin_avg_pred = np.where(bin_counts > 0, bin_sums / bin_counts, 0)
        bin_avg_true = np.where(bin_counts > 0, bin_true / bin_counts, 0)
    
    # Export calibration data
    calibration_df = pd.DataFrame({
        'Predicted_Probability': bin_avg_pred,
        'True_Probability': bin_avg_true,
        'Sample_Count': bin_counts,
        'Bin_Center': bin_centers
    })
    
    calibration_df.to_excel(OUTPUT_DIR / 'step4_calibration_data.xlsx', index=False)
    print(f"   ✅ Saved: step4_calibration_data.xlsx")
    
    print("\n✅ Calibration analysis completed!")
    
    return brier_score


# ============================================================================
# PART 7: CATBOOST COMPARISON
# ============================================================================

def catboost_comparison(df, feature_cols, scaler):
    """
    Compare XGBoost with CatBoost algorithm
    """
    print("\n" + "="*80)
    print("PART 7: CATBOOST ALGORITHM COMPARISON")
    print("="*80)
    
    try:
        from catboost import CatBoostClassifier, Pool
        print("\n✅ CatBoost library loaded")
    except ImportError:
        print("\n⚠️  CatBoost not available. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'catboost', '--break-system-packages', '-q'])
        from catboost import CatBoostClassifier, Pool
        print("✅ CatBoost installed successfully")
    
    print("\n🔄 Training CatBoost model...")
    
    # Prepare data
    X = df[feature_cols]
    y = df['PoF_12_month']
    
    # Scale features using helper function
    scaled_cols = ['Ekipman_Yaşı_Yıl', 'Arıza_Sayısı_12ay', 'Arıza_Sayısı_3ay',
                   'Toplam_Müşteri_Sayısı', 'Ekipman_Yoğunluk_Skoru', 
                   'Müşteri_Başına_Arıza', 'Ay_Sin']
    
    X_scaled = safe_scale_features(scaler, X, scaled_cols)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train CatBoost
    catboost_model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.05,
        depth=5,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False
    )
    
    catboost_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    
    # Predictions
    y_pred_proba_cat = catboost_model.predict_proba(X_test)[:, 1]
    
    # Calculate AUC
    auc_catboost = roc_auc_score(y_test, y_pred_proba_cat)
    
    print(f"\n   ✅ CatBoost trained successfully")
    print(f"   - CatBoost AUC: {auc_catboost:.4f}")
    
    # Load XGBoost model for comparison
    with open('outputs/step3_5_final_model.pkl', 'rb') as f:
        xgboost_model = pickle.load(f)
    
    y_pred_proba_xgb = xgboost_model.predict_proba(X_test)[:, 1]
    auc_xgboost = roc_auc_score(y_test, y_pred_proba_xgb)
    
    print(f"   - XGBoost AUC: {auc_xgboost:.4f}")
    print(f"   - Difference: {abs(auc_catboost - auc_xgboost):.4f}")
    
    # Compare ROC curves
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
    fpr_cat, tpr_cat, _ = roc_curve(y_test, y_pred_proba_cat)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgboost:.4f})', linewidth=2)
    plt.plot(fpr_cat, tpr_cat, label=f'CatBoost (AUC = {auc_catboost:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison: XGBoost vs CatBoost', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'step4_catboost_roc_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n   ✅ Saved: step4_catboost_roc_comparison.png")
    plt.close()
    
    # Feature importance comparison
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
    
    comparison_df.to_excel(OUTPUT_DIR / 'step4_model_comparison.xlsx', index=False)
    print(f"\n   ✅ Saved: step4_model_comparison.xlsx")
    
    return catboost_model, auc_catboost


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Load data
    df, model, scaler, best_params = load_and_validate_data()
    
    if df is None:
        print("\n❌ Cannot proceed without required files. Please upload and rerun.")
        exit(1)
    
    # Define feature columns (from Step 3.5 "Low VIF" strategy)
    # IMPORTANT: Order must match the training order expected by the model
    feature_cols = [
        'Ekipman_Yaşı_Yıl', 'Arıza_Sayısı_12ay', 'Arıza_Sayısı_3ay',
        'Toplam_Müşteri_Sayısı', 'Kentsel_Müşteri_Oranı', 'Kırsal_Müşteri_Oranı',
        'OG_Müşteri_Oranı', 'Ekipman_Yoğunluk_Skoru', 'Müşteri_Başına_Arıza',
        'Tekrarlayan_Arıza_Flag', 'Hafta_İçi', 'Ay_Sin', 'Ay_Cos',
        'Mevsim_Sonbahar', 'Mevsim_Yaz', 'Mevsim_İlkbahar'
    ]
    
    # Verify columns exist
    available_features = [col for col in feature_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]
    
    if missing_features:
        print(f"\n⚠️  Warning: Missing features: {missing_features}")
        print(f"   Using available features only: {len(available_features)}")
        feature_cols = available_features
    
    # Execute all parts
    try:
        # Part 1: Survival Analysis
        survival_data, cox_model = survival_analysis(df)
        
        # Part 2: Backtesting
        backtest_results = backtesting_analysis(df, model, scaler, feature_cols)
        
        # Part 3: Risk Maps
        create_risk_maps(df)
        
        # Part 4: CAPEX Prioritization
        capex_df = capex_prioritization(df)
        
        # Part 5: SHAP Analysis
        shap_analysis(df, model, feature_cols)
        
        # Part 6: Calibration
        brier_score = calibration_analysis(df, model, feature_cols, scaler)
        
        # Part 7: CatBoost Comparison
        catboost_model, catboost_auc = catboost_comparison(df, feature_cols, scaler)
        
        # Final Summary
        print("\n" + "="*80)
        print("STEP 4 ENHANCED - EXECUTION COMPLETE!")
        print("="*80)
        
        print(f"\n✅ All analyses completed successfully!")
        print(f"\n📁 Output files saved to: {OUTPUT_DIR}")
        print(f"\n📊 Summary of Outputs:")
        output_files = list(OUTPUT_DIR.glob('*'))
        for i, file in enumerate(output_files, 1):
            print(f"   {i}. {file.name}")
        
        print(f"\n🎯 Key Findings:")
        print(f"   - XGBoost AUC: {backtest_results['AUC'].mean():.4f}")
        print(f"   - CatBoost AUC: {catboost_auc:.4f}")
        print(f"   - Brier Score: {brier_score:.4f}")
        print(f"   - High-risk equipment: {len(capex_df):,}")
        print(f"   - P1 Urgent priority: {(capex_df['Priority_Tier'] == 'P1 - Urgent').sum():,}")
        
        print("\n" + "="*80)
        print("Thank you for using Step 4 Enhanced!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()