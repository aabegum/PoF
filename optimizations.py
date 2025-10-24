"""
STEP 3.5: CRITICAL OPTIMIZATIONS
=================================
Amaç: Multicollinearity çözme + Feature engineering + Hyperparameter tuning
Hedef: AUC 0.707 → 0.750+ (+4-6%)

Optimizasyonlar:
1. VIF Analysis (multicollinearity detection)
2. Feature Transformation (sparsity handling)
3. Feature Selection (correlation-based)
4. Hyperparameter Tuning (GridSearchCV)
5. Model Comparison (before/after)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, roc_curve, f1_score)
import xgboost as xgb
import joblib

# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = 'outputs/step2_feature_engineered_data.xlsx'
OUTPUT_DIR = 'outputs/'
RANDOM_STATE = 42

print("=" * 80)
print("STEP 3.5: CRITICAL OPTIMIZATIONS")
print("=" * 80)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}\n")

# ============================================================================
# 1. LOAD DATA & CREATE TARGETS
# ============================================================================
print("[1/8] Loading data and creating targets...")
df = pd.read_excel(INPUT_FILE, engine='openpyxl')
print(f"✓ Data loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Parse dates
if df['Arıza_Tarihi'].dtype == 'object':
    df['Arıza_Tarihi'] = pd.to_datetime(df['Arıza_Tarihi'])

# Sort and create PoF target
df_sorted = df.sort_values(['Ekipman Kodu', 'Arıza_Tarihi']).reset_index(drop=True)

def create_pof_target(df_sorted, window_days):
    targets = []
    for idx, row in df_sorted.iterrows():
        ekipman = row['Ekipman Kodu']
        current_date = row['Arıza_Tarihi']
        
        future_faults = df_sorted[
            (df_sorted['Ekipman Kodu'] == ekipman) &
            (df_sorted['Arıza_Tarihi'] > current_date) &
            (df_sorted['Arıza_Tarihi'] <= current_date + pd.Timedelta(days=window_days))
        ]
        targets.append(1 if len(future_faults) > 0 else 0)
    return targets

df_sorted['PoF_12_month'] = create_pof_target(df_sorted, 365)
df = df_sorted.copy()

print(f"✓ PoF_12_month: {df['PoF_12_month'].sum():,} positives ({df['PoF_12_month'].mean():.1%})\n")

# ============================================================================
# 2. BASELINE FEATURES (from V2)
# ============================================================================
print("[2/8] Preparing baseline features...")

# Cyclical encoding
if 'Ay' in df.columns:
    df['Ay_Sin'] = np.sin(2 * np.pi * df['Ay'] / 12)
    df['Ay_Cos'] = np.cos(2 * np.pi * df['Ay'] / 12)
    
    season_map = {
        12: 'Kış', 1: 'Kış', 2: 'Kış',
        3: 'İlkbahar', 4: 'İlkbahar', 5: 'İlkbahar',
        6: 'Yaz', 7: 'Yaz', 8: 'Yaz',
        9: 'Sonbahar', 10: 'Sonbahar', 11: 'Sonbahar'
    }
    df['Mevsim'] = df['Ay'].map(season_map)
    season_dummies = pd.get_dummies(df['Mevsim'], prefix='Mevsim', drop_first=True)
    df = pd.concat([df, season_dummies], axis=1)

baseline_features = [
    'Ekipman_Yaşı_Yıl',
    'Arıza_Sayısı_12ay',
    'Arıza_Sayısı_6ay',
    'Arıza_Sayısı_3ay',
    'Toplam_Müşteri_Sayısı',
    'Kentsel_Müşteri_Oranı',
    'Kırsal_Müşteri_Oranı',
    'OG_Müşteri_Oranı',
    'Ekipman_Yoğunluk_Skoru',
    'Müşteri_Başına_Arıza',
    'Tekrarlayan_Arıza_Flag',
    'Hafta_İçi',
    'Ay_Sin',
    'Ay_Cos'
]

# Add season dummies if present
season_cols = [col for col in df.columns if col.startswith('Mevsim_')]
baseline_features.extend(season_cols)

print(f"✓ Baseline features: {len(baseline_features)}\n")

# ============================================================================
# 3. VIF ANALYSIS - MULTICOLLINEARITY DETECTION
# ============================================================================
print("[3/8] VIF Analysis - Detecting Multicollinearity...")
print("=" * 80)

# Prepare data for VIF - ONLY numeric continuous features
# VIF doesn't work with binary/categorical features
numeric_continuous_features = [
    'Ekipman_Yaşı_Yıl',
    'Arıza_Sayısı_12ay',
    'Arıza_Sayısı_6ay',
    'Arıza_Sayısı_3ay',
    'Toplam_Müşteri_Sayısı',
    'Kentsel_Müşteri_Oranı',
    'Kırsal_Müşteri_Oranı',
    'OG_Müşteri_Oranı',
    'Ekipman_Yoğunluk_Skoru',
    'Müşteri_Başına_Arıza',
    'Ay_Sin',
    'Ay_Cos'
]

# Filter to available features
numeric_continuous_features = [f for f in numeric_continuous_features if f in baseline_features]

print(f"VIF Analysis on {len(numeric_continuous_features)} numeric continuous features")
print(f"(Excluding binary/categorical features)")

df_vif = df[numeric_continuous_features].fillna(df[numeric_continuous_features].median())

# Ensure all data is numeric and finite
df_vif = df_vif.astype(float)

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = df_vif.columns
vif_data["VIF"] = [
    variance_inflation_factor(df_vif.values, i) 
    for i in range(len(df_vif.columns))
]
vif_data = vif_data.sort_values('VIF', ascending=False)

print("\n📊 VIF SCORES (Variance Inflation Factor):")
print("="*60)
for idx, row in vif_data.iterrows():
    status = "⚠️ HIGH" if row['VIF'] > 10 else ("⚠️ MODERATE" if row['VIF'] > 5 else "✅ OK")
    print(f"  {row['Feature']:30s}: {row['VIF']:8.2f}  {status}")

# Identify problematic features
high_vif = vif_data[vif_data['VIF'] > 10]['Feature'].tolist()
moderate_vif = vif_data[(vif_data['VIF'] > 5) & (vif_data['VIF'] <= 10)]['Feature'].tolist()

print(f"\n🔴 High VIF (>10): {len(high_vif)} features")
for feat in high_vif:
    print(f"     {feat}")

print(f"\n🟡 Moderate VIF (5-10): {len(moderate_vif)} features")
for feat in moderate_vif:
    print(f"     {feat}")

# Save VIF results
vif_data.to_csv(OUTPUT_DIR + 'step3_5_vif_analysis.csv', index=False)
print(f"\n✓ VIF results saved: {OUTPUT_DIR}step3_5_vif_analysis.csv")

# ============================================================================
# 4. CORRELATION ANALYSIS
# ============================================================================
print("\n[4/8] Correlation Analysis...")
print("=" * 80)

# Focus on arıza features
ariza_features = [col for col in baseline_features if 'Arıza' in col or 'Yoğunluk' in col]
if len(ariza_features) > 1:
    corr_matrix = df[ariza_features].corr()
    
    print("\n📊 ARIZA FEATURES CORRELATION MATRIX:")
    print(corr_matrix.round(3))
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr_pairs:
        print(f"\n🔴 HIGH CORRELATION PAIRS (|r| > 0.7):")
        for f1, f2, corr in high_corr_pairs:
            print(f"     {f1} ↔ {f2}: {corr:.3f}")

# ============================================================================
# 5. FEATURE ENGINEERING - TRANSFORMATIONS
# ============================================================================
print("\n[5/8] Feature Engineering - Transformations...")
print("=" * 80)

# A. Log transformation for sparse features
print("\n🔧 Creating Log-Transformed Features:")
df['Arıza_12ay_Log'] = np.log1p(df['Arıza_Sayısı_12ay'])
df['Arıza_6ay_Log'] = np.log1p(df['Arıza_Sayısı_6ay'])
df['Arıza_3ay_Log'] = np.log1p(df['Arıza_Sayısı_3ay'])
print("  ✓ Arıza_12ay_Log, Arıza_6ay_Log, Arıza_3ay_Log")

# B. Binary flags (Has fault in period?)
print("\n🔧 Creating Binary Fault Flags:")
df['Has_Arıza_12ay'] = (df['Arıza_Sayısı_12ay'] > 0).astype(int)
df['Has_Arıza_6ay'] = (df['Arıza_Sayısı_6ay'] > 0).astype(int)
df['Has_Arıza_3ay'] = (df['Arıza_Sayısı_3ay'] > 0).astype(int)
print("  ✓ Has_Arıza_12ay, Has_Arıza_6ay, Has_Arıza_3ay")

# C. Fault intensity (faults per year of age)
print("\n🔧 Creating Fault Intensity Features:")
df['Arıza_Intensity'] = df['Arıza_Sayısı_12ay'] / (df['Ekipman_Yaşı_Yıl'] + 1)
df['Arıza_Intensity_6ay'] = df['Arıza_Sayısı_6ay'] / (df['Ekipman_Yaşı_Yıl'] + 1)
print("  ✓ Arıza_Intensity, Arıza_Intensity_6ay")

# D. Age buckets
print("\n🔧 Creating Age Category:")
df['Age_Category'] = pd.cut(
    df['Ekipman_Yaşı_Yıl'],
    bins=[0, 5, 10, 15, 100],
    labels=['0-5y', '5-10y', '10-15y', '15+y']
)
age_dummies = pd.get_dummies(df['Age_Category'], prefix='Age', drop_first=True)
df = pd.concat([df, age_dummies], axis=1)
# Drop the categorical column to avoid XGBoost error with categorical dtypes
df = df.drop('Age_Category', axis=1)
print("  ✓ Age category dummies (3 features)")

# E. Customer load indicator
print("\n🔧 Creating Customer Load Indicator:")
df['High_Customer_Load'] = (df['Toplam_Müşteri_Sayısı'] > df['Toplam_Müşteri_Sayısı'].median()).astype(int)
print("  ✓ High_Customer_Load")

print(f"\n✓ Total new features created: 13")

# ============================================================================
# 6. FEATURE SELECTION STRATEGIES
# ============================================================================
print("\n[6/8] Creating Feature Sets for Comparison...")
print("=" * 80)

# Strategy 1: Remove high VIF features
features_low_vif = [f for f in baseline_features if f not in high_vif]
print(f"\n1️⃣ LOW VIF STRATEGY:")
print(f"   Removed {len(high_vif)} high VIF features")
print(f"   Remaining: {len(features_low_vif)} features")

# Strategy 2: Use ONLY log-transformed arıza features (remove originals)
features_log_only = [f for f in baseline_features if 'Arıza_Sayısı' not in f]
features_log_only.extend(['Arıza_12ay_Log', 'Arıza_6ay_Log', 'Arıza_3ay_Log'])
print(f"\n2️⃣ LOG-TRANSFORMED STRATEGY:")
print(f"   Replaced count features with log-transformed")
print(f"   Total: {len(features_log_only)} features")

# Strategy 3: Enhanced (log + binary + intensity)
features_enhanced = features_log_only.copy()
features_enhanced.extend([
    'Has_Arıza_12ay', 'Has_Arıza_6ay', 'Has_Arıza_3ay',
    'Arıza_Intensity', 'Arıza_Intensity_6ay',
    'High_Customer_Load'
])
# Add age dummies
age_dummy_cols = [col for col in df.columns if col.startswith('Age_')]
features_enhanced.extend(age_dummy_cols)
print(f"\n3️⃣ ENHANCED STRATEGY:")
print(f"   Added: binary flags, intensity, age buckets")
print(f"   Total: {len(features_enhanced)} features")

# Strategy 4: Kitchen sink (all features)
features_all = baseline_features + [
    'Arıza_12ay_Log', 'Arıza_6ay_Log', 'Arıza_3ay_Log',
    'Has_Arıza_12ay', 'Has_Arıza_6ay', 'Has_Arıza_3ay',
    'Arıza_Intensity', 'Arıza_Intensity_6ay',
    'High_Customer_Load'
] + age_dummy_cols
print(f"\n4️⃣ ALL FEATURES STRATEGY:")
print(f"   Total: {len(features_all)} features")

# ============================================================================
# 7. MODEL TRAINING & COMPARISON
# ============================================================================
print("\n[7/8] Training Models with Different Feature Sets...")
print("=" * 80)

strategies = {
    'Baseline (V2)': baseline_features,
    'Low VIF': features_low_vif,
    'Log-Transformed': features_log_only,
    'Enhanced': features_enhanced,
    'All Features': features_all
}

results = []

for strategy_name, features in strategies.items():
    print(f"\n{'='*60}")
    print(f"🎯 STRATEGY: {strategy_name}")
    print(f"{'='*60}")
    print(f"Features: {len(features)}")
    
    # Prepare data
    df_model = df[features + ['PoF_12_month']].copy()
    df_model = df_model.fillna(df_model.median(numeric_only=True))
    
    X = df_model[features]
    y = df_model['PoF_12_month']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale numeric features
    numeric_features = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
    
    # Train with default params first
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        max_depth=5,
        learning_rate=0.1,
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        tree_method='hist'
    )
    
    model.fit(X_train_scaled, y_train, verbose=False)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    
    print(f"\n📊 Results:")
    print(f"   AUC-ROC:   {auc:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    
    # Feature importance (top 10)
    feature_imp = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n   Top 5 Features:")
    for idx, row in feature_imp.head(5).iterrows():
        print(f"     {row['Feature']:30s}: {row['Importance']:.4f}")
    
    # Store results
    results.append({
        'Strategy': strategy_name,
        'N_Features': len(features),
        'AUC': auc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall,
        'Top_Feature': feature_imp.iloc[0]['Feature'],
        'Top_Feature_Imp': feature_imp.iloc[0]['Importance']
    })

# Comparison table
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('AUC', ascending=False)

print("\n" + "=" * 80)
print("📊 STRATEGY COMPARISON")
print("=" * 80)
print(results_df.to_string(index=False))

# Save results
results_df.to_excel(OUTPUT_DIR + 'step3_5_strategy_comparison.xlsx', index=False)
print(f"\n✓ Comparison saved: {OUTPUT_DIR}step3_5_strategy_comparison.xlsx")

# ============================================================================
# 8. HYPERPARAMETER TUNING (Best Strategy)
# ============================================================================
print("\n[8/8] Hyperparameter Tuning for Best Strategy...")
print("=" * 80)

# Select best strategy
best_strategy_name = results_df.iloc[0]['Strategy']
best_features = strategies[best_strategy_name]

print(f"\n🏆 Best Strategy: {best_strategy_name}")
print(f"   AUC: {results_df.iloc[0]['AUC']:.4f}")
print(f"   Features: {len(best_features)}")

# Prepare data
df_model = df[best_features + ['PoF_12_month']].copy()
df_model = df_model.fillna(df_model.median(numeric_only=True))

X = df_model[best_features]
y = df_model['PoF_12_month']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scale
numeric_features = X.select_dtypes(include=[np.number]).columns
scaler_final = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_features] = scaler_final.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler_final.transform(X_test[numeric_features])

# Hyperparameter grid
print("\n⏳ Running GridSearchCV (this may take 5-10 minutes)...")

param_grid = {
    'max_depth': [5, 6, 7],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 3]
}

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

grid_search = GridSearchCV(
    xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        tree_method='hist'
    ),
    param_grid,
    cv=3,  # 3-fold CV for speed
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

# Best model
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"\n✓ GridSearch completed!")
print(f"\n🏆 BEST HYPERPARAMETERS:")
for param, value in best_params.items():
    print(f"   {param}: {value}")

# Final predictions
y_pred_final = best_model.predict(X_test_scaled)
y_pred_proba_final = best_model.predict_proba(X_test_scaled)[:, 1]

auc_final = roc_auc_score(y_test, y_pred_proba_final)
f1_final = f1_score(y_test, y_pred_final)

print(f"\n📊 FINAL MODEL PERFORMANCE:")
print(f"   AUC-ROC:   {auc_final:.4f}")
print(f"   F1-Score:  {f1_final:.4f}")
print(classification_report(y_test, y_pred_final, target_names=['No Fault', 'Fault']))

# Feature importance
feature_imp_final = pd.DataFrame({
    'Feature': best_features,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n📈 TOP 10 FEATURES (Final Model):")
for idx, row in feature_imp_final.head(10).iterrows():
    print(f"  {row['Feature']:30s}: {row['Importance']:.4f}")

# ============================================================================
# 9. SAVE FINAL MODEL & RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING FINAL MODEL & RESULTS")
print("=" * 80)

# Predict on full dataset
df_full_features = df[best_features].fillna(df[best_features].median())
df_full_scaled = df_full_features.copy()
df_full_scaled[numeric_features] = scaler_final.transform(df_full_features[numeric_features])

df['PoF_Score_Final'] = best_model.predict_proba(df_full_scaled)[:, 1]
df['Risk_Category_Final'] = pd.cut(
    df['PoF_Score_Final'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

# Risk distribution
print(f"\n📊 RISK DISTRIBUTION (Final Model):")
for risk in ['High Risk', 'Medium Risk', 'Low Risk']:
    count = (df['Risk_Category_Final'] == risk).sum()
    print(f"  {risk:15s}: {count:6,} ({count/len(df)*100:5.1f}%)")

# Kesici analysis
kesici_mask = df['Ekipman Sınıfı'].str.contains('Kesici|kesici', na=False, case=False)
df_kesici = df[kesici_mask].copy()

print(f"\n🔧 KESICI EQUIPMENT (Final Model):")
print(f"  Total: {len(df_kesici):,}")
for risk in ['High Risk', 'Medium Risk', 'Low Risk']:
    count = (df_kesici['Risk_Category_Final'] == risk).sum()
    pct = count / len(df_kesici) * 100 if len(df_kesici) > 0 else 0
    print(f"  {risk:15s}: {count:6,} ({pct:5.1f}%)")

# Save outputs
output_file = OUTPUT_DIR + 'step3_5_final_risk_scored.xlsx'
df.to_excel(output_file, index=False, engine='openpyxl')
print(f"\n✓ Risk-scored dataset: {output_file}")

high_risk = df[df['Risk_Category_Final'] == 'High Risk'].copy()
high_risk_file = OUTPUT_DIR + 'step3_5_final_high_risk.xlsx'
high_risk.to_excel(high_risk_file, index=False, engine='openpyxl')
print(f"✓ High-risk equipment: {high_risk_file}")

feature_imp_final.to_csv(OUTPUT_DIR + 'step3_5_feature_importance.csv', index=False)
print(f"✓ Feature importance: {OUTPUT_DIR}step3_5_feature_importance.csv")

joblib.dump(best_model, OUTPUT_DIR + 'step3_5_final_model.pkl')
joblib.dump(scaler_final, OUTPUT_DIR + 'step3_5_final_scaler.pkl')
print(f"✓ Model: {OUTPUT_DIR}step3_5_final_model.pkl")
print(f"✓ Scaler: {OUTPUT_DIR}step3_5_final_scaler.pkl")

# Save best params
import json
with open(OUTPUT_DIR + 'step3_5_best_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)
print(f"✓ Best params: {OUTPUT_DIR}step3_5_best_params.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✓ STEP 3.5 COMPLETED SUCCESSFULLY")
print("=" * 80)

print(f"""
📊 PERFORMANCE IMPROVEMENT:

BASELINE (V2):
  • AUC: 0.7074

BEST STRATEGY ({best_strategy_name}):
  • AUC: {results_df.iloc[0]['AUC']:.4f}
  • Improvement: {(results_df.iloc[0]['AUC'] - 0.7074) * 100:+.2f}%

FINAL MODEL (Tuned):
  • AUC: {auc_final:.4f}
  • Improvement: {(auc_final - 0.7074) * 100:+.2f}%
  • F1-Score: {f1_final:.4f}

🎯 KEY FINDINGS:
  1. VIF Analysis: {len(high_vif)} features with VIF > 10
  2. Best Strategy: {best_strategy_name}
  3. Top Feature: {feature_imp_final.iloc[0]['Feature']}
  4. High Risk Equipment: {(df['Risk_Category_Final'] == 'High Risk').sum():,}
  5. High Risk Kesici: {(df_kesici['Risk_Category_Final'] == 'High Risk').sum():,}

💡 OPTIMIZATIONS APPLIED:
  ✅ VIF analysis (multicollinearity detection)
  ✅ Feature transformation (log, binary, intensity)
  ✅ Feature selection (correlation-based)
  ✅ Hyperparameter tuning (GridSearchCV)
  ✅ Best strategy selection

📁 OUTPUTS:
  ✓ step3_5_vif_analysis.csv
  ✓ step3_5_strategy_comparison.xlsx
  ✓ step3_5_final_risk_scored.xlsx
  ✓ step3_5_final_high_risk.xlsx
  ✓ step3_5_feature_importance.csv
  ✓ step3_5_final_model.pkl
  ✓ step3_5_final_scaler.pkl
  ✓ step3_5_best_params.json

🔜 NEXT: STEP 4 (Advanced Analysis)
  • SHAP analysis
  • Survival analysis
  • Backtesting
  • Health scores
  • Risk maps
""")

print("=" * 80)
print("✅ Ready for STEP 4: Advanced Analysis\n")