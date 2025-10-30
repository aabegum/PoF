"""
STEP 3.5: Survival Modeling for Multi-Horizon Predictions
==========================================================
Purpose:
- Train survival analysis model instead of binary classification
- Generate predictions for multiple time horizons (3m, 6m, 12m, 24m)
- Single unified model for all horizons
- Output survival probabilities and risk scores

Method: Random Survival Forest or Gradient Boosting Survival Analysis
"""

import pandas as pd
import numpy as np
import warnings
import os
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Survival analysis imports
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, integrated_brier_score

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = 'outputs/step2_feature_engineered_data.xlsx'  # Using existing feature engineering output
OUTPUT_DIR = 'outputs/'
RANDOM_STATE = 42
MAX_OBSERVATION_DAYS = 730  # 2 years maximum observation window

print("=" * 80)
print("STEP 3.5: SURVIVAL MODELING FOR MULTI-HORIZON PREDICTIONS")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("[1/9] Loading Feature-Engineered Data...")

try:
    df = pd.read_excel(INPUT_FILE, engine='openpyxl')
    print(f"✓ Data loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
except Exception as e:
    print(f"✗ Excel error: {e}")
    print("Trying CSV format...")
    df = pd.read_csv(INPUT_FILE.replace('.xlsx', '.csv'))
    print(f"✓ Data loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Parse dates
if df['Arıza_Tarihi'].dtype == 'object':
    df['Arıza_Tarihi'] = pd.to_datetime(df['Arıza_Tarihi'])

# ============================================================================
# 2. CREATE SURVIVAL TARGETS
# ============================================================================
print("\n[2/9] Creating Survival Analysis Targets...")
print("   (time_to_event, event_occurred)")

df_sorted = df.sort_values(['Ekipman Kodu', 'Arıza_Tarihi']).reset_index(drop=True)

def create_survival_target(df_sorted, max_days=730):
    """
    Create survival analysis targets:
    - time_to_event: days until next failure (or censoring)
    - event_occurred: True if failure observed, False if censored
    """
    times = []
    events = []

    for idx, row in df_sorted.iterrows():
        ekipman = row['Ekipman Kodu']
        current_date = row['Arıza_Tarihi']

        # Find next failure for this equipment
        future_faults = df_sorted[
            (df_sorted['Ekipman Kodu'] == ekipman) &
            (df_sorted['Arıza_Tarihi'] > current_date)
        ]

        if len(future_faults) > 0:
            # Event occurred - calculate time to next failure
            next_fault_date = future_faults.iloc[0]['Arıza_Tarihi']
            days_to_failure = (next_fault_date - current_date).days

            # Cap at max observation window
            if days_to_failure <= max_days:
                times.append(days_to_failure)
                events.append(True)  # Event occurred
            else:
                times.append(max_days)
                events.append(False)  # Censored at max_days
        else:
            # No future fault observed - censored
            times.append(max_days)
            events.append(False)

    return times, events

# Create survival targets
print(f"   Max observation window: {MAX_OBSERVATION_DAYS} days ({MAX_OBSERVATION_DAYS/365:.1f} years)")
time_to_event, event_occurred = create_survival_target(df_sorted, MAX_OBSERVATION_DAYS)

df_sorted['time_to_event'] = time_to_event
df_sorted['event_occurred'] = event_occurred

# Statistics
n_events = sum(event_occurred)
n_censored = len(event_occurred) - n_events
median_time_events = np.median([t for t, e in zip(time_to_event, event_occurred) if e])

print(f"\n   ✓ Total samples: {len(df_sorted):,}")
print(f"   ✓ Events (failures): {n_events:,} ({n_events/len(df_sorted)*100:.1f}%)")
print(f"   ✓ Censored: {n_censored:,} ({n_censored/len(df_sorted)*100:.1f}%)")
print(f"   ✓ Median time to failure (events): {median_time_events:.0f} days ({median_time_events/30:.1f} months)")

df = df_sorted.copy()

# ============================================================================
# 3. FEATURE SELECTION (VIF Analysis)
# ============================================================================
print("\n[3/9] Feature Selection with VIF Analysis...")

# Define features (same as original pipeline)
all_features = [
    'Ekipman_Yaşı_Yıl',
    'Arıza_Sayısı_12ay', 'Arıza_Sayısı_6ay', 'Arıza_Sayısı_3ay',
    'MTBF_days',
    'Toplam_Müşteri_Sayısı', 'Kentsel_Müşteri_Sayısı', 'Kırsal_Müşteri_Sayısı',
    'Kentsel_Müşteri_Oranı', 'Kırsal_Müşteri_Oranı',
    'Age_0-5y', 'Age_5-10y', 'Age_10-15y', 'Age_15+y',
    'Season_Spring', 'Season_Summer', 'Season_Fall', 'Season_Winter'
]

# Filter features that exist in data
available_features = [f for f in all_features if f in df.columns]
print(f"   Available features: {len(available_features)}/{len(all_features)}")

# Remove features with high VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df, features):
    """Calculate VIF for each feature"""
    df_vif = df[features].fillna(df[features].median())

    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(df_vif.values, i) for i in range(len(features))]

    return vif_data.sort_values('VIF', ascending=False)

# Initial VIF
vif_df = calculate_vif(df, available_features)
print(f"\n   Initial VIF (top 5):")
for idx, row in vif_df.head(5).iterrows():
    print(f"     {row['Feature']:30s}: {row['VIF']:8.2f}")

# Remove high VIF features (VIF > 10)
selected_features = vif_df[vif_df['VIF'] <= 10]['Feature'].tolist()

# If too restrictive, use VIF > 20
if len(selected_features) < 8:
    selected_features = vif_df[vif_df['VIF'] <= 20]['Feature'].tolist()
    print(f"\n   ⚠️ Using VIF threshold of 20 (had too few features with VIF < 10)")

print(f"\n   ✓ Selected features after VIF: {len(selected_features)}")
print(f"     {', '.join(selected_features[:10])}")
if len(selected_features) > 10:
    print(f"     ... and {len(selected_features) - 10} more")

# ============================================================================
# 3B. ADD EQUIPMENT TYPE FEATURES (ONE-HOT ENCODING)
# ============================================================================
print("\n[3B/9] Adding Equipment Type Features...")

# Check for Equipment_Type column (consolidated type)
equipment_type_col = None
if 'Equipment_Type' in df.columns:
    equipment_type_col = 'Equipment_Type'
    print(f"   ✓ Using Equipment_Type column")
elif 'Ekipman Sınıfı' in df.columns:
    equipment_type_col = 'Ekipman Sınıfı'
    print(f"   ✓ Using Ekipman Sınıfı column (fallback)")
else:
    print(f"   ⚠️  No equipment type column found, skipping equipment type features")

equipment_type_features = []
equipment_type_mapping = {}

if equipment_type_col:
    # Get equipment type distribution
    eq_type_counts = df[equipment_type_col].value_counts()
    print(f"   ✓ Found {len(eq_type_counts)} unique equipment types")

    # Keep only top N types (to avoid too many features)
    TOP_N_TYPES = 15  # Keep top 15 most common types
    MIN_COUNT = 50    # Minimum 50 occurrences to include

    top_types = eq_type_counts.head(TOP_N_TYPES)
    top_types = top_types[top_types >= MIN_COUNT]

    print(f"   ✓ Keeping top {len(top_types)} equipment types (min count: {MIN_COUNT})")
    print(f"     Top 5 types:")
    for eq_type, count in top_types.head(5).items():
        print(f"       - {eq_type}: {count:,} ({count/len(df)*100:.1f}%)")

    # Create one-hot encoded features
    for eq_type in top_types.index:
        # Create clean feature name
        feature_name = f"EqType_{eq_type}".replace(' ', '_').replace('/', '_').replace('-', '_')
        feature_name = feature_name[:50]  # Truncate if too long

        # Create binary feature
        df[feature_name] = (df[equipment_type_col] == eq_type).astype(int)
        equipment_type_features.append(feature_name)
        equipment_type_mapping[feature_name] = eq_type

    print(f"   ✓ Created {len(equipment_type_features)} equipment type features")

    # Add equipment type features to selected features
    selected_features = selected_features + equipment_type_features

    print(f"   ✓ Total features (including equipment type): {len(selected_features)}")
else:
    print(f"   ⚠️  Proceeding without equipment type features")

# ============================================================================
# 4. PREPARE DATA FOR SURVIVAL MODEL
# ============================================================================
print("\n[4/9] Preparing Data for Survival Model...")

# Fill missing values
df_model = df[selected_features + ['time_to_event', 'event_occurred']].copy()
df_model[selected_features] = df_model[selected_features].fillna(df_model[selected_features].median())

X = df_model[selected_features]
# Create structured array for survival target (required by scikit-survival)
y = Surv.from_arrays(
    event=df_model['event_occurred'].values,
    time=df_model['time_to_event'].values
)

print(f"   ✓ X shape: {X.shape}")
print(f"   ✓ y shape: {y.shape}")

# Train-test split (stratified by event)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE,
    stratify=df_model['event_occurred']
)

print(f"\n   Train: {len(X_train):,} samples ({y_train['event'].sum():,} events, {y_train['event'].sum()/len(y_train)*100:.1f}%)")
print(f"   Test:  {len(X_test):,} samples ({y_test['event'].sum():,} events, {y_test['event'].sum()/len(y_test)*100:.1f}%)")

# ============================================================================
# 5. STANDARDIZE FEATURES
# ============================================================================
print("\n[5/9] Standardizing Features...")

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

print(f"   ✓ Scaler fitted on training data")

# ============================================================================
# 6. TRAIN SURVIVAL MODELS (Compare RSF vs GBS)
# ============================================================================
print("\n[6/9] Training Survival Models...")
print("   Comparing: Random Survival Forest vs Gradient Boosting")

models = {}
results = []

# Model 1: Random Survival Forest
print("\n   🌲 Training Random Survival Forest...")
rsf = RandomSurvivalForest(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=0
)

rsf.fit(X_train_scaled, y_train)
models['RSF'] = rsf

# Evaluate
rsf_cindex = rsf.score(X_test_scaled, y_test)
print(f"      ✓ C-index (test): {rsf_cindex:.4f}")

results.append({
    'Model': 'Random Survival Forest',
    'C-index': rsf_cindex,
    'N_estimators': 100,
    'Max_depth': 10
})

# Model 2: Gradient Boosting Survival Analysis
print("\n   ⚡ Training Gradient Boosting Survival...")
gbs = GradientBoostingSurvivalAnalysis(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    subsample=0.8,
    random_state=RANDOM_STATE,
    verbose=0
)

gbs.fit(X_train_scaled, y_train)
models['GBS'] = gbs

# Evaluate
gbs_cindex = gbs.score(X_test_scaled, y_test)
print(f"      ✓ C-index (test): {gbs_cindex:.4f}")

results.append({
    'Model': 'Gradient Boosting Survival',
    'C-index': gbs_cindex,
    'N_estimators': 100,
    'Learning_rate': 0.1
})

# Select best model
results_df = pd.DataFrame(results)
best_model_name = results_df.loc[results_df['C-index'].idxmax(), 'Model']
best_cindex = results_df['C-index'].max()
best_model = models['GBS'] if best_model_name == 'Gradient Boosting Survival' else models['RSF']

print("\n" + "=" * 80)
print("📊 MODEL COMPARISON")
print("=" * 80)
print(results_df.to_string(index=False))
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   C-index: {best_cindex:.4f}")

# ============================================================================
# 7. GENERATE MULTI-HORIZON PREDICTIONS
# ============================================================================
print("\n[7/9] Generating Multi-Horizon Predictions...")

# Define prediction horizons
horizons = {
    '3_month': 91,
    '6_month': 182,
    '12_month': 365,
    '24_month': 730
}

# Get survival functions
surv_funcs_test = best_model.predict_survival_function(X_test_scaled)

# Extract probabilities at each horizon
horizon_predictions = {}

for horizon_name, horizon_days in horizons.items():
    # Survival probability = P(survive beyond horizon)
    # Risk = 1 - Survival probability
    survival_probs = [fn(horizon_days) for fn in surv_funcs_test]
    risk_probs = [1 - s for s in survival_probs]

    horizon_predictions[f'Risk_{horizon_name}'] = risk_probs

    print(f"   ✓ {horizon_name:10s}: Mean risk = {np.mean(risk_probs):.1%}, "
          f"High risk (>0.6) = {sum(r > 0.6 for r in risk_probs):,} ({sum(r > 0.6 for r in risk_probs)/len(risk_probs)*100:.1f}%)")

# Add to test dataframe
df_test_predictions = X_test.copy()
for col, values in horizon_predictions.items():
    df_test_predictions[col] = values

# ============================================================================
# 8. EVALUATE CALIBRATION
# ============================================================================
print("\n[8/9] Evaluating Prediction Calibration...")

# For each horizon, check if predictions correlate with actual events
for horizon_name, horizon_days in [('3_month', 91), ('6_month', 182), ('12_month', 365)]:
    # Create binary target: did failure occur within horizon?
    actual_events = []
    for i, idx in enumerate(X_test.index):
        time = y_test[i]['time']
        event = y_test[i]['event']

        # Failure within horizon?
        if event and time <= horizon_days:
            actual_events.append(1)
        else:
            actual_events.append(0)

    # Calculate AUC
    predictions = df_test_predictions[f'Risk_{horizon_name}'].values
    if sum(actual_events) > 0 and sum(actual_events) < len(actual_events):
        auc = roc_auc_score(actual_events, predictions)
        print(f"   {horizon_name:10s} AUC: {auc:.4f} ({sum(actual_events):,} actual failures)")
    else:
        print(f"   {horizon_name:10s} - Cannot calculate AUC (all same class)")

# ============================================================================
# 9. SAVE MODEL AND PREDICTIONS
# ============================================================================
print("\n[9/9] Saving Model and Results...")

# Save model
model_file = OUTPUT_DIR + 'step3_5_survival_model.pkl'
joblib.dump(best_model, model_file)
print(f"   ✓ Model: {model_file}")

# Save scaler
scaler_file = OUTPUT_DIR + 'step3_5_survival_scaler.pkl'
joblib.dump(scaler, scaler_file)
print(f"   ✓ Scaler: {scaler_file}")

# Save features
features_file = OUTPUT_DIR + 'step3_5_survival_features.json'
with open(features_file, 'w') as f:
    json.dump(selected_features, f, indent=2)
print(f"   ✓ Features: {features_file}")

# Save equipment type mapping
eq_type_file = OUTPUT_DIR + 'step3_5_equipment_type_mapping.json'
with open(eq_type_file, 'w') as f:
    json.dump({
        'equipment_type_col': equipment_type_col,
        'equipment_type_features': equipment_type_features,
        'equipment_type_mapping': equipment_type_mapping,
        'top_n_types': len(equipment_type_features)
    }, f, indent=2)
print(f"   ✓ Equipment type mapping: {eq_type_file}")

# Save metadata
metadata = {
    'model_type': best_model_name,
    'c_index': float(best_cindex),
    'n_features': len(selected_features),
    'n_features_base': len(selected_features) - len(equipment_type_features),
    'n_features_equipment_type': len(equipment_type_features),
    'equipment_type_col': equipment_type_col,
    'n_train': len(X_train),
    'n_test': len(X_test),
    'n_events_train': int(y_train['event'].sum()),
    'n_events_test': int(y_test['event'].sum()),
    'max_observation_days': MAX_OBSERVATION_DAYS,
    'horizons': horizons,
    'trained_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

metadata_file = OUTPUT_DIR + 'step3_5_survival_metadata.json'
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   ✓ Metadata: {metadata_file}")

# Generate predictions for FULL dataset
print("\n   Generating predictions for full dataset...")
df_full_features = df[selected_features].fillna(df[selected_features].median())
df_full_scaled = pd.DataFrame(
    scaler.transform(df_full_features),
    columns=selected_features,
    index=df_full_features.index
)

surv_funcs_full = best_model.predict_survival_function(df_full_scaled)

for horizon_name, horizon_days in horizons.items():
    survival_probs = [fn(horizon_days) for fn in surv_funcs_full]
    risk_probs = [1 - s for s in survival_probs]
    df[f'Risk_{horizon_name}'] = risk_probs

# Add risk categories (based on 12-month risk)
df['Risk_Category'] = pd.cut(
    df['Risk_12_month'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

# Risk distribution
print(f"\n📊 RISK DISTRIBUTION (12-month horizon):")
for risk in ['High Risk', 'Medium Risk', 'Low Risk']:
    count = (df['Risk_Category'] == risk).sum()
    print(f"  {risk:15s}: {count:6,} ({count/len(df)*100:5.1f}%)")

# Save full dataset with predictions
output_file = OUTPUT_DIR + 'step3_5_survival_risk_scored.xlsx'
df.to_excel(output_file, index=False, engine='openpyxl')
print(f"\n   ✓ Risk-scored dataset: {output_file}")

# Save high-risk equipment
high_risk = df[df['Risk_Category'] == 'High Risk'].sort_values('Risk_12_month', ascending=False)
high_risk_file = OUTPUT_DIR + 'step3_5_survival_high_risk.xlsx'
high_risk.to_excel(high_risk_file, index=False, engine='openpyxl')
print(f"   ✓ High-risk equipment: {high_risk_file} ({len(high_risk):,} records)")

# Save model comparison
results_df.to_excel(OUTPUT_DIR + 'step3_5_survival_model_comparison.xlsx', index=False)
print(f"   ✓ Model comparison: {OUTPUT_DIR}step3_5_survival_model_comparison.xlsx")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ STEP 3.5: SURVIVAL MODELING COMPLETED SUCCESSFULLY")
print("=" * 80)

eq_type_summary = f"""
  • Equipment type features: {len(equipment_type_features)} types
  • Equipment type column: {equipment_type_col}
""" if equipment_type_features else "  • Equipment type features: Not used"

print(f"""
📊 MODEL PERFORMANCE:
  • Model: {best_model_name}
  • C-index: {best_cindex:.4f} (higher is better, 0.5 = random, 1.0 = perfect)
  • Total features: {len(selected_features)}
  • Base features: {len(selected_features) - len(equipment_type_features)}
{eq_type_summary}
  • Events: {n_events:,} ({n_events/len(df)*100:.1f}%)
  • Censored: {n_censored:,} ({n_censored/len(df)*100:.1f}%)

📈 MULTI-HORIZON PREDICTIONS:
  • 3-month risk scores generated
  • 6-month risk scores generated
  • 12-month risk scores generated
  • 24-month risk scores generated

🎯 DELIVERABLES:
  • Single unified survival model
  • Equipment-specific failure patterns learned
  • Risk scores for all equipment at all horizons
  • High-risk equipment identified ({len(high_risk):,} records)

✅ Ready for STEP 5: Scoring Pipeline & Analytics
""")

print("=" * 80)
