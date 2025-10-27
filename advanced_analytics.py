"""
STEP 4: ADVANCED ANALYTICS & INTEGRATION (ADJUSTED)
====================================================
Purpose: 
- Survival Analysis (Kaplan-Meier + Weibull AFT)
- Backtesting Framework
- Health Score Calculation (0-100) - Adjusted for no maintenance data
- SHAP Analysis for Root Causes
- Risk Categorization & CAPEX Prioritization

Builds on: Step 3.5 optimized model (AUC 0.7148) with 16 Low VIF features
Input: outputs/step3_5_final_risk_scored.xlsx
Model: outputs/step3_5_final_model.pkl
Features: 16 optimized Low VIF features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Advanced Analytics Libraries
from lifelines import KaplanMeierFitter, WeibullAFTFitter
from lifelines.statistics import logrank_test
import shap
import joblib
from sklearn.metrics import (roc_auc_score, f1_score, precision_recall_curve, 
                            classification_report, confusion_matrix, 
                            precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Geographical plotting
import folium
from folium.plugins import HeatMap

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = 'outputs/step3_5_final_risk_scored.xlsx'
MODEL_FILE = 'outputs/step3_5_final_model.pkl'
SCALER_FILE = 'outputs/step3_5_final_scaler.pkl'
OUTPUT_DIR = 'outputs/step4_advanced/'
RANDOM_STATE = 42

# Create output directory
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("STEP 4: ADVANCED ANALYTICS & INTEGRATION (ADJUSTED)")
print("=" * 80)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
print(f"Building on Step 3.5 optimized model (AUC 0.7148)")
print(f"Using 16 optimized Low VIF features\n")

# ============================================================================
# 1. LOAD DATA & MODEL
# ============================================================================
print("[1/7] Loading optimized model and data...")

# Load data
df = pd.read_excel(INPUT_FILE, engine='openpyxl')
print(f"✓ Data loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Load model and scaler
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
print(f"✓ Model loaded: {MODEL_FILE}")
print(f"✓ Scaler loaded: {SCALER_FILE}")

# Parse dates if needed
if df['Arıza_Tarihi'].dtype == 'object':
    df['Arıza_Tarihi'] = pd.to_datetime(df['Arıza_Tarihi'])

# Get the 16 optimized Low VIF features from Step 3.5
feature_importance = pd.read_csv('outputs/step3_5_feature_importance.csv')
best_features = feature_importance['Feature'].tolist()[:16]  # Top 16 Low VIF features
print(f"✓ Using {len(best_features)} optimized Low VIF features:")
for i, feature in enumerate(best_features, 1):
    print(f"  {i:2d}. {feature}")

print()

# ============================================================================
# 2. SURVIVAL ANALYSIS
# ============================================================================
print("[2/7] Performing Survival Analysis...")

def prepare_survival_data(df):
    """Prepare data for survival analysis - adjusted for available data"""
    df_survival = df.copy()

    # Use Equipment_Type if available, otherwise fall back to Ekipman Sınıfı
    equipment_type_col = 'Equipment_Type' if 'Equipment_Type' in df_survival.columns else 'Ekipman Sınıfı'

    # Calculate time-to-failure for each equipment
    equipment_failures = df_survival.groupby('Ekipman Kodu').agg({
        'Arıza_Tarihi': ['min', 'max', 'count'],
        'Ekipman_Yaşı_Yıl': 'first',
        equipment_type_col: 'first',
        'PoF_12_month': 'max'  # Use the PoF target
    }).reset_index()

    equipment_failures.columns = ['Ekipman_Kodu', 'First_Fault', 'Last_Fault',
                                 'Fault_Count', 'Age', 'Equipment_Class', 'Had_Future_Fault']
    
    # Reference date for right-censoring (use max fault date + safety margin)
    analysis_date = df_survival['Arıza_Tarihi'].max() + timedelta(days=30)
    
    # Calculate survival time (days since first fault to last observation or failure)
    equipment_failures['Time_To_Failure'] = (
        equipment_failures['Last_Fault'] - equipment_failures['First_Fault']
    ).dt.days
    
    # Event indicator (1 if equipment had future fault within 12 months, 0 otherwise)
    equipment_failures['Failure_Event'] = equipment_failures['Had_Future_Fault']
    
    # For equipment without future faults, they are censored at analysis date
    censored_mask = equipment_failures['Failure_Event'] == 0
    equipment_failures.loc[censored_mask, 'Time_To_Failure'] = (
        analysis_date - equipment_failures.loc[censored_mask, 'First_Fault']
    ).dt.days
    
    # Filter out equipment with Time_To_Failure <= 0 (data issues)
    equipment_failures = equipment_failures[equipment_failures['Time_To_Failure'] > 0]
    
    return equipment_failures

# Prepare survival data
df_survival = prepare_survival_data(df)
print(f"✓ Survival data prepared: {len(df_survival):,} equipment")
print(f"  - Failures: {df_survival['Failure_Event'].sum():,}")
print(f"  - Censored: {(df_survival['Failure_Event'] == 0).sum():,}")

# Kaplan-Meier Analysis
print("\n📊 Kaplan-Meier Survival Analysis...")
kmf = KaplanMeierFitter()

plt.figure(figsize=(12, 8))

# Plot overall survival curve
kmf.fit(
    df_survival['Time_To_Failure'],
    df_survival['Failure_Event'],
    label='All Equipment'
)
kmf.plot_survival_function(ci_show=True)

plt.title('Survival Function - All Equipment', fontsize=14, fontweight='bold')
plt.xlabel('Days Since First Fault', fontsize=12)
plt.ylabel('Survival Probability', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'survival_analysis_overall.png', dpi=300, bbox_inches='tight')
print(f"✓ Kaplan-Meier plot saved: {OUTPUT_DIR}survival_analysis_overall.png")

# Weibull AFT Model with available features
print("\n📊 Weibull AFT Model Fitting...")
aft_features = ['Age', 'Fault_Count']
aft_data = df_survival[aft_features + ['Time_To_Failure', 'Failure_Event']].dropna()

if len(aft_data) > 0:
    aft = WeibullAFTFitter()
    aft.fit(aft_data, duration_col='Time_To_Failure', event_col='Failure_Event')
    
    # Plot AFT results
    plt.figure(figsize=(10, 6))
    aft.plot()
    plt.title('Weibull AFT Model - Feature Effects on Survival', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'survival_analysis_aft.png', dpi=300, bbox_inches='tight')
    print(f"✓ Weibull AFT plot saved: {OUTPUT_DIR}survival_analysis_aft.png")
    
    # Save AFT summary
    aft_summary = aft.summary
    aft_summary.to_csv(OUTPUT_DIR + 'weibull_aft_summary.csv')
    print(f"✓ Weibull AFT summary saved: {OUTPUT_DIR}weibull_aft_summary.csv")
    
    # FIXED: Handle Weibull AFT summary display
    print(f"\n📈 Weibull AFT Key Insights:")
    try:
        # Try different ways to access the parameters
        if hasattr(aft, 'lambda_'):
            print(f"  - Scale (λ): {aft.lambda_:.3f}")
        elif 'lambda_' in aft.summary.index:
            print(f"  - Scale (λ): {aft.summary.loc['lambda_', 'coef']:.3f}")
        else:
            print(f"  - Scale (λ): {aft.summary.iloc[0, 0]:.3f}")
            
        if hasattr(aft, 'rho_'):
            print(f"  - Shape (ρ): {aft.rho_:.3f}")
        elif 'rho_' in aft.summary.index:
            print(f"  - Shape (ρ): {aft.summary.loc['rho_', 'coef']:.3f}")
        else:
            print(f"  - Shape (ρ): {aft.summary.iloc[1, 0]:.3f}")
            
    except Exception as e:
        print(f"  - Could not extract AFT parameters: {e}")
        print(f"  - AFT model fitted successfully with {len(aft_data)} samples")
    
    # Display feature effects
    if 'Age' in aft.summary.index:
        try:
            age_effect = aft.summary.loc['Age', 'coef']
            age_direction = 'shorter' if age_effect < 0 else 'longer'
            print(f"  - Age effect: {age_effect:.3f} (older → {age_direction} survival)")
        except:
            print(f"  - Age effect: Available in summary")
    
    if 'Fault_Count' in aft.summary.index:
        try:
            fault_effect = aft.summary.loc['Fault_Count', 'coef']
            fault_direction = 'shorter' if fault_effect < 0 else 'longer'
            print(f"  - Fault count effect: {fault_effect:.3f} (more faults → {fault_direction} survival)")
        except:
            print(f"  - Fault count effect: Available in summary")

# ============================================================================
# 3. BACKTESTING FRAMEWORK
# ============================================================================
print("\n[3/7] Running Backtesting Framework...")

def temporal_backtest(df, features, horizons=[90, 180, 365]):
    """Temporal backtesting with time-based splits"""
    
    backtest_results = {}
    df_sorted = df.sort_values('Arıza_Tarihi').reset_index(drop=True)
    
    # Use time-based splits (60%, 70%, 80% cutoffs)
    cutoff_dates = [
        df_sorted['Arıza_Tarihi'].quantile(0.6),  # 60% cutoff
        df_sorted['Arıza_Tarihi'].quantile(0.7),  # 70% cutoff  
        df_sorted['Arıza_Tarihi'].quantile(0.8)   # 80% cutoff
    ]
    
    for i, cutoff_date in enumerate(cutoff_dates):
        print(f"\n  🔄 Backtest Split {i+1}: Cutoff = {cutoff_date.strftime('%Y-%m-%d')}")
        
        # Split data temporally
        train_mask = df_sorted['Arıza_Tarihi'] <= cutoff_date
        test_mask = (df_sorted['Arıza_Tarihi'] > cutoff_date) & \
                   (df_sorted['Arıza_Tarihi'] <= cutoff_date + timedelta(days=365))
        
        X_train = df_sorted.loc[train_mask, features]
        y_train = df_sorted.loc[train_mask, 'PoF_12_month']
        X_test = df_sorted.loc[test_mask, features]
        y_test = df_sorted.loc[test_mask, 'PoF_12_month']
        
        print(f"    Train: {len(X_train):,} (until {cutoff_date.strftime('%Y-%m-%d')})")
        print(f"    Test:  {len(X_test):,} (next 12 months)")
        
        if len(X_test) == 0 or y_test.sum() < 5:
            print("    ⚠️ Skipping - insufficient test data")
            continue
        
        # Scale features using the same scaler pattern
        scaler_split = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train_scaled[numeric_cols] = scaler_split.fit_transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = scaler_split.transform(X_test[numeric_cols])
        
        # Retrain model with same architecture
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model_retrained = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=5,
            learning_rate=0.05,  # From tuned parameters
            n_estimators=100,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            tree_method='hist'
        )
        
        model_retrained.fit(X_train_scaled, y_train, verbose=False)
        
        # Predictions
        y_pred_proba = model_retrained.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        backtest_results[f'split_{i+1}'] = {
            'cutoff_date': cutoff_date,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'test_positives': y_test.sum(),
            'auc': auc,
            'f1_score': f1,
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0)
        }
        
        print(f"    📊 AUC: {auc:.4f}, F1: {f1:.4f}")
    
    return backtest_results

# Run backtesting
backtest_results = temporal_backtest(df, best_features)

# Save backtest results
if backtest_results:
    backtest_df = pd.DataFrame.from_dict(backtest_results, orient='index')
    backtest_df.to_excel(OUTPUT_DIR + 'backtesting_results.xlsx')
    print(f"\n✓ Backtesting results saved: {OUTPUT_DIR}backtesting_results.xlsx")
    
    # Calculate average performance
    avg_auc = backtest_df['auc'].mean()
    print(f"📊 Average Backtest AUC: {avg_auc:.4f}")
else:
    print("⚠️ No backtesting results generated - insufficient temporal splits")

# ============================================================================
# 4. HEALTH SCORE CALCULATION (ADJUSTED - NO MAINTENANCE DATA)
# ============================================================================
print("\n[4/7] Calculating Health Scores (Adjusted for No Maintenance Data)...")

def calculate_health_score_adjusted(row):
    """
    Convert PoF to Health Score (0-100) - ADJUSTED VERSION
    Higher score = better health
    Adjusted for no maintenance data availability
    """
    pof_score = row['PoF_Score_Final']
    
    # Base health score (inverse of PoF) - 60% weight
    base_score = (1 - pof_score) * 60
    
    # Age penalty (older equipment gets penalty) - 20% weight
    age_penalty = min(row['Ekipman_Yaşı_Yıl'] * 1.0, 20)  # Max 20 point penalty
    
    # Fault history penalty - 20% weight
    fault_penalty = min(row.get('Arıza_Sayısı_12ay', 0) * 4, 20)
    
    # Operational load penalty (based on customer count)
    load_penalty = 0
    if 'Toplam_Müşteri_Sayısı' in row:
        high_load = row['Toplam_Müşteri_Sayısı'] > df['Toplam_Müşteri_Sayısı'].median()
        if high_load:
            load_penalty = 5  # Small penalty for high operational load
    
    # Calculate final health score
    health_score = base_score - age_penalty - fault_penalty - load_penalty
    
    # Ensure score is between 0-100
    return max(0, min(100, health_score))

# Apply adjusted health score calculation
df['Health_Score'] = df.apply(calculate_health_score_adjusted, axis=1)

# Categorize health status
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

print("📊 Health Score Distribution (Adjusted):")
health_stats = df['Health_Status'].value_counts()
for status, count in health_stats.items():
    print(f"  {status:10s}: {count:6,} ({count/len(df)*100:5.1f}%)")

print(f"\n  Average Health Score: {df['Health_Score'].mean():.1f}")
print(f"  Median Health Score: {df['Health_Score'].median():.1f}")

# Health score by key factors
print(f"\n📈 Health Score by Age Groups:")
age_groups = pd.cut(df['Ekipman_Yaşı_Yıl'], bins=[0, 5, 10, 20, 100])
health_by_age = df.groupby(age_groups)['Health_Score'].mean()
for age_group, score in health_by_age.items():
    print(f"  {age_group}: {score:.1f}")

# ============================================================================
# 5. SHAP ANALYSIS FOR ROOT CAUSES (FIXED)
# ============================================================================
print("\n[5/7] Performing SHAP Analysis...")

# Prepare data for SHAP - ensure same features as during training
df_shap = df[best_features].copy()
df_shap = df_shap.fillna(df_shap.median())

# Sample data for SHAP to avoid memory issues (SHAP is computationally expensive)
sample_size = min(1000, len(df_shap))
df_shap_sample = df_shap.sample(n=sample_size, random_state=RANDOM_STATE)
print(f"✓ Sampled {sample_size:,} rows for SHAP analysis")

# Get the feature names the model expects
if hasattr(model, 'feature_names_in_'):
    model_features = list(model.feature_names_in_)
    print(f"✓ Model expects {len(model_features)} features")
elif hasattr(model, 'get_booster'):
    # XGBoost specific
    booster = model.get_booster()
    model_features = booster.feature_names
    print(f"✓ Model expects {len(model_features)} features (from booster)")
else:
    # Fallback to best_features
    model_features = best_features
    print(f"⚠️ Using best_features as model features")

# Ensure we have all required features in the correct order
missing_features = set(model_features) - set(df_shap_sample.columns)
if missing_features:
    print(f"⚠️ Missing features: {missing_features}")
    # Add missing features with zeros
    for feat in missing_features:
        df_shap_sample[feat] = 0

# Reorder columns to match model's expected order
df_shap_sample = df_shap_sample[model_features]
print(f"✓ Aligned features: {df_shap_sample.shape}")

# Apply the same scaling transformation as during training
# The scaler was fit on the training data, so we just transform
X_shap_scaled = scaler.transform(df_shap_sample)

# Convert back to DataFrame to preserve feature names
X_shap = pd.DataFrame(
    X_shap_scaled,
    columns=model_features,
    index=df_shap_sample.index
)
print(f"✓ Scaled data shape: {X_shap.shape}")

# Verify shape matches model expectations
expected_features = len(model_features)
actual_features = X_shap.shape[1]
if expected_features != actual_features:
    print(f"⚠️ Warning: Expected {expected_features} features, got {actual_features}")

# Create SHAP explainer with the model
print("✓ Creating SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)

# Calculate SHAP values - use numpy array for stability
print("✓ Computing SHAP values (this may take a moment)...")
shap_values = explainer.shap_values(X_shap.values)

print(f"✓ SHAP values calculated: shape {np.array(shap_values).shape}")

# SHAP Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap, show=False, max_display=15)
plt.title('SHAP Feature Importance - Root Cause Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ SHAP summary plot saved: {OUTPUT_DIR}shap_summary.png")

# SHAP Bar Plot (mean absolute SHAP values)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=15)
plt.title('SHAP Feature Importance (Mean Absolute Impact)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'shap_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ SHAP bar plot saved: {OUTPUT_DIR}shap_bar.png")

# Calculate global feature importance from SHAP
shap_df = pd.DataFrame(shap_values, columns=[f'SHAP_{f}' for f in X_shap.columns])
global_shap_importance = shap_df.abs().mean().sort_values(ascending=False)

print(f"\n📊 Global SHAP Feature Importance (Top 10):")
for i, (feature, importance) in enumerate(global_shap_importance.head(10).items(), 1):
    feature_name = feature.replace('SHAP_', '')
    print(f"  {i:2d}. {feature_name:30s}: {importance:.4f}")

# Save detailed SHAP values for the sampled data
shap_df['Ekipman_Kodu'] = df.loc[X_shap.index, 'Ekipman Kodu'].values
shap_df['PoF_Score'] = df.loc[X_shap.index, 'PoF_Score_Final'].values
shap_df['Health_Score'] = df.loc[X_shap.index, 'Health_Score'].values
shap_df.to_csv(OUTPUT_DIR + 'shap_values_detailed.csv', index=False)
print(f"✓ Detailed SHAP values saved: {OUTPUT_DIR}shap_values_detailed.csv")
print(f"  (Computed for {sample_size:,} sampled equipment)")
# ============================================================================
# 6. CAPEX PRIORITIZATION & RISK CATEGORIZATION
# ============================================================================
print("\n[6/7] Generating CAPEX Prioritization...")

def categorize_capex_priority(row):
    """Categorize equipment for CAPEX investment prioritization"""
    pof_score = row['PoF_Score_Final']
    health_score = row['Health_Score']
    age = row['Ekipman_Yaşı_Yıl']
    
    # High risk + poor health + old age = HIGHEST priority
    if pof_score > 0.7 and health_score < 30:
        return 'IMMEDIATE_REPLACEMENT'
    elif pof_score > 0.6 and health_score < 40:
        return 'HIGH_PRIORITY'
    elif pof_score > 0.5 or health_score < 50:
        return 'MEDIUM_PRIORITY'
    elif pof_score > 0.4 or health_score < 60:
        return 'LOW_PRIORITY'
    else:
        return 'MONITOR_ONLY'

df['CAPEX_Priority'] = df.apply(categorize_capex_priority, axis=1)

print("📊 CAPEX Prioritization Distribution:")
capex_stats = df['CAPEX_Priority'].value_counts()
for priority, count in capex_stats.items():
    print(f"  {priority:20s}: {count:6,} ({count/len(df)*100:5.1f}%)")

# High-risk equipment requiring immediate attention
high_risk_equipment = df[df['CAPEX_Priority'].isin(['IMMEDIATE_REPLACEMENT', 'HIGH_PRIORITY'])]
print(f"\n⚠️  High-risk equipment requiring action: {len(high_risk_equipment):,}")

# Equipment class analysis for CAPEX planning
print(f"\n🔧 EQUIPMENT TYPE - CAPEX ANALYSIS:")
equipment_type_col = 'Equipment_Type' if 'Equipment_Type' in df.columns else 'Ekipman Sınıfı'
equipment_capex = df.groupby(equipment_type_col)['CAPEX_Priority'].value_counts().unstack().fillna(0)
top_classes = equipment_capex.sum(axis=1).sort_values(ascending=False).head(5)

for eq_class in top_classes.index:
    class_data = equipment_capex.loc[eq_class]
    high_risk_pct = (class_data.get('IMMEDIATE_REPLACEMENT', 0) + class_data.get('HIGH_PRIORITY', 0)) / class_data.sum() * 100
    print(f"  {eq_class:25s}: {class_data.sum():4,.0f} total, {high_risk_pct:4.1f}% high risk")

# ============================================================================
# 7. GEOGRAPHICAL RISK MAPPING (using available coordinates)
# ============================================================================
print("\n[7/7] Creating Geographical Risk Visualization...")

def create_risk_map(df, output_file):
    """Create interactive risk map using Folium - adjusted for available coordinates"""
    
    # Check if coordinates are available
    if 'KOORDINAT_X' not in df.columns or 'KOORDINAT_Y' not in df.columns:
        print("⚠️  Coordinate columns not found, skipping map creation")
        return None
    
    # Filter valid coordinates
    valid_coords = df.dropna(subset=['KOORDINAT_X', 'KOORDINAT_Y']).copy()
    valid_coords = valid_coords[(valid_coords['KOORDINAT_X'] != 0) & (valid_coords['KOORDINAT_Y'] != 0)]
    
    if len(valid_coords) == 0:
        print("⚠️  No valid coordinates found, skipping map creation")
        return None
    
    print(f"  Plotting {len(valid_coords):,} equipment with valid coordinates")
    
    # Create base map centered on average coordinates
    avg_lat = valid_coords['KOORDINAT_Y'].mean()
    avg_lon = valid_coords['KOORDINAT_X'].mean()
    
    risk_map = folium.Map(location=[avg_lat, avg_lon], zoom_start=10)
    
    # Add risk heatmap
    heat_data = []
    for _, row in valid_coords.iterrows():
        heat_data.append([row['KOORDINAT_Y'], row['KOORDINAT_X'], row['PoF_Score_Final']])
    
    HeatMap(heat_data, radius=15, blur=10, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(risk_map)
    
    # Save map
    risk_map.save(output_file)
    return risk_map

# Create risk map
risk_map = create_risk_map(df, OUTPUT_DIR + 'risk_heatmap.html')
if risk_map:
    print(f"✓ Risk heatmap saved: {OUTPUT_DIR}risk_heatmap.html")

# Create static risk visualization as fallback
plt.figure(figsize=(12, 8))
if 'KOORDINAT_X' in df.columns and 'KOORDINAT_Y' in df.columns:
    valid_coords = df.dropna(subset=['KOORDINAT_X', 'KOORDINAT_Y'])
    valid_coords = valid_coords[(valid_coords['KOORDINAT_X'] != 0) & (valid_coords['KOORDINAT_Y'] != 0)]
    
    if len(valid_coords) > 0:
        scatter = plt.scatter(
            valid_coords['KOORDINAT_X'], 
            valid_coords['KOORDINAT_Y'],
            c=valid_coords['PoF_Score_Final'],
            cmap='RdYlGn_r',
            alpha=0.6,
            s=30
        )
        plt.colorbar(scatter, label='PoF Score')
        plt.title('Geographical Risk Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Longitude (KOORDINAT_X)')
        plt.ylabel('Latitude (KOORDINAT_Y)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR + 'risk_distribution_static.png', dpi=300, bbox_inches='tight')
        print(f"✓ Static risk distribution saved: {OUTPUT_DIR}risk_distribution_static.png")

# ============================================================================
# FINAL OUTPUTS & SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: COMPREHENSIVE OUTPUTS")
print("=" * 80)

# Save final analyzed dataset
final_output_file = OUTPUT_DIR + 'step4_comprehensive_analysis.xlsx'
df.to_excel(final_output_file, index=False, engine='openpyxl')
print(f"✓ Comprehensive analysis saved: {final_output_file}")

# Save high-risk equipment for action
high_risk_equipment = df[df['CAPEX_Priority'].isin(['IMMEDIATE_REPLACEMENT', 'HIGH_PRIORITY'])]
high_risk_file = OUTPUT_DIR + 'step4_high_risk_equipment.xlsx'
high_risk_equipment.to_excel(high_risk_file, index=False, engine='openpyxl')
print(f"✓ High-risk equipment list: {high_risk_file}")

# Generate summary report
summary_file = OUTPUT_DIR + 'step4_analysis_summary.txt'
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("STEP 4: ADVANCED ANALYTICS SUMMARY REPORT (ADJUSTED)\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Equipment Analyzed: {len(df):,}\n")
    f.write(f"Model AUC: 0.7148 (from Step 3.5)\n")
    f.write(f"Features Used: {len(best_features)} optimized Low VIF features\n\n")
    
    f.write("HEALTH SCORE DISTRIBUTION:\n")
    for status, count in df['Health_Status'].value_counts().items():
        f.write(f"  {status}: {count:,} ({count/len(df)*100:.1f}%)\n")
    
    f.write(f"\nCAPEX PRIORITIZATION:\n")
    for priority, count in df['CAPEX_Priority'].value_counts().items():
        f.write(f"  {priority}: {count:,} ({count/len(df)*100:.1f}%)\n")
    
    f.write(f"\nHIGH-RISK EQUIPMENT:\n")
    f.write(f"  Immediate Replacement: {(df['CAPEX_Priority'] == 'IMMEDIATE_REPLACEMENT').sum():,}\n")
    f.write(f"  High Priority: {(df['CAPEX_Priority'] == 'HIGH_PRIORITY').sum():,}\n")
    
    if backtest_results:
        f.write(f"\nBACKTESTING PERFORMANCE:\n")
        for split, results in backtest_results.items():
            f.write(f"  {split}: AUC = {results['auc']:.4f}, F1 = {results['f1_score']:.4f}\n")
    
    f.write(f"\nTOP 5 SHAP FEATURES:\n")
    for i, (feature, importance) in enumerate(global_shap_importance.head(5).items(), 1):
        feature_name = feature.replace('SHAP_', '')
        f.write(f"  {i}. {feature_name}: {importance:.4f}\n")

print(f"✓ Analysis summary: {summary_file}")

print("\n" + "=" * 80)
print("📊 STEP 4 COMPLETED SUCCESSFULLY!")
print("=" * 80)

print(f"""
🎯 KEY DELIVERABLES (ADJUSTED):

1. SURVIVAL ANALYSIS:
   • Kaplan-Meier survival curves for all equipment
   • Weibull AFT model with age and fault count effects
   • Equipment lifetime expectations based on available data

2. BACKTESTING RESULTS:
   • Temporal validation across multiple time splits
   • Model performance consistency check
   • Real-world predictive power assessment

3. HEALTH SCORING (ADJUSTED):
   • 0-100 health scores for all equipment
   • Adjusted algorithm for no maintenance data
   • Health status categorization (Excellent → Critical)

4. ROOT CAUSE ANALYSIS:
   • SHAP feature importance plots
   • Detailed SHAP values for each equipment
   • Transparent model explanations

5. CAPEX PRIORITIZATION:
   • Immediate action recommendations
   • Equipment replacement prioritization
   • Data-driven investment planning

6. GEOGRAPHICAL INSIGHTS:
   • Risk heatmaps (using available coordinates)
   • Regional risk concentration analysis

📊 KEY STATISTICS:
   • High-risk equipment: {len(high_risk_equipment):,}
   • Critical health status: {(df['Health_Status'] == 'Critical').sum():,}
   • Immediate replacement needed: {(df['CAPEX_Priority'] == 'IMMEDIATE_REPLACEMENT').sum():,}

📁 OUTPUT FILES:
   ✓ {OUTPUT_DIR}step4_comprehensive_analysis.xlsx
   ✓ {OUTPUT_DIR}step4_high_risk_equipment.xlsx
   ✓ {OUTPUT_DIR}survival_analysis_overall.png
   ✓ {OUTPUT_DIR}survival_analysis_aft.png
   ✓ {OUTPUT_DIR}backtesting_results.xlsx
   ✓ {OUTPUT_DIR}shap_summary.png
   ✓ {OUTPUT_DIR}shap_bar.png
   ✓ {OUTPUT_DIR}shap_values_detailed.csv
   ✓ {OUTPUT_DIR}risk_heatmap.html (if coordinates available)
   ✓ {OUTPUT_DIR}step4_analysis_summary.txt

🔧 ACTIONABLE INSIGHTS:
   • {(df['CAPEX_Priority'] == 'IMMEDIATE_REPLACEMENT').sum():,} equipment require immediate replacement
   • {(df['CAPEX_Priority'] == 'HIGH_PRIORITY').sum():,} equipment need high-priority attention
   • Top risk factors identified via SHAP analysis
   • Health scores enable preventive maintenance planning

💡 ADJUSTMENTS MADE:
   • Health scoring adapted for no maintenance data
   • Used 16 optimized Low VIF features from Step 3.5
   • Built on AUC 0.7148 optimized model
   • Focused on general equipment analysis (Kesici analysis omitted)

🔜 POTENTIAL NEXT STEPS:
   • Integrate SCADA operational data when available
   • Add maintenance history records if obtained
   • Implement real-time monitoring dashboard
   • Develop equipment-specific deep dives
""")

print("=" * 80)
print("✅ R&D PROJECT ADVANCED ANALYTICS PHASE COMPLETED")
print("=" * 80)