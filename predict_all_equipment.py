"""
STEP 5: Predict Failure Risk for All Equipment (Multi-Horizon)
===============================================================
Purpose:
- Load trained survival model
- Score ANY equipment (including those without fault history)
- Generate multi-horizon risk predictions (3m, 6m, 12m, 24m)
- Output actionable equipment prioritization lists

Input:
- Processed equipment data (with features)
- Trained survival model, scaler, features

Output:
- Excel file with risk scores for all equipment at all horizons
- Prioritized action lists by horizon
- Summary statistics and visualizations
"""

import pandas as pd
import numpy as np
import warnings
import os
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = 'outputs/step2_feature_engineered_data.xlsx'  # Using existing feature engineering output
MODEL_FILE = 'outputs/step3_5_survival_model.pkl'
SCALER_FILE = 'outputs/step3_5_survival_scaler.pkl'
FEATURES_FILE = 'outputs/step3_5_survival_features.json'
METADATA_FILE = 'outputs/step3_5_survival_metadata.json'
EQ_TYPE_FILE = 'outputs/step3_5_equipment_type_mapping.json'
OUTPUT_DIR = 'outputs/'

print("=" * 80)
print("STEP 5: MULTI-HORIZON RISK PREDICTIONS FOR ALL EQUIPMENT")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# 1. LOAD MODEL AND METADATA
# ============================================================================
print("[1/6] Loading Trained Survival Model...")

# Load model
model = joblib.load(MODEL_FILE)
print(f"   ✓ Model loaded: {MODEL_FILE}")

# Load scaler
scaler = joblib.load(SCALER_FILE)
print(f"   ✓ Scaler loaded: {SCALER_FILE}")

# Load features
with open(FEATURES_FILE, 'r') as f:
    feature_names = json.load(f)
print(f"   ✓ Features loaded: {len(feature_names)} features")

# Load metadata
with open(METADATA_FILE, 'r') as f:
    metadata = json.load(f)
print(f"   ✓ Metadata loaded")
print(f"     Model type: {metadata['model_type']}")
print(f"     C-index: {metadata['c_index']:.4f}")
print(f"     Trained on: {metadata['trained_on']}")

horizons = metadata['horizons']

# Load equipment type mapping
equipment_type_info = None
if os.path.exists(EQ_TYPE_FILE):
    with open(EQ_TYPE_FILE, 'r') as f:
        equipment_type_info = json.load(f)
    print(f"   ✓ Equipment type mapping loaded")
    print(f"     Equipment types: {equipment_type_info['top_n_types']}")
    print(f"     Column used: {equipment_type_info['equipment_type_col']}")
else:
    print(f"   ℹ️  No equipment type mapping found (model trained without equipment type features)")

# ============================================================================
# 2. LOAD EQUIPMENT DATA
# ============================================================================
print("\n[2/6] Loading Equipment Data...")

try:
    df = pd.read_excel(INPUT_FILE, engine='openpyxl')
    print(f"   ✓ Data loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
except:
    df = pd.read_csv(INPUT_FILE.replace('.xlsx', '.csv'))
    print(f"   ✓ Data loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Get unique equipment (aggregate by equipment code)
print("\n   Aggregating to equipment level...")
if 'Ekipman Kodu' in df.columns:
    # Take most recent record per equipment
    df_sorted = df.sort_values(['Ekipman Kodu', 'Arıza_Tarihi'], ascending=[True, False])
    df_equipment = df_sorted.groupby('Ekipman Kodu').first().reset_index()

    print(f"   ✓ Total equipment: {len(df_equipment):,}")
    print(f"   ✓ Total work orders: {len(df):,}")
else:
    df_equipment = df.copy()
    print(f"   ✓ Total records: {len(df_equipment):,}")

# ============================================================================
# 3. PREPARE FEATURES (Including Equipment Type)
# ============================================================================
print("\n[3/6] Preparing Features...")

# Create equipment type features if model was trained with them
if equipment_type_info and equipment_type_info['equipment_type_col']:
    print("\n   Creating equipment type features...")
    equipment_type_col = equipment_type_info['equipment_type_col']
    equipment_type_mapping = equipment_type_info['equipment_type_mapping']

    if equipment_type_col in df_equipment.columns:
        # Create one-hot encoded features for equipment types
        for feature_name, eq_type in equipment_type_mapping.items():
            df_equipment[feature_name] = (df_equipment[equipment_type_col] == eq_type).astype(int)

        print(f"   ✓ Created {len(equipment_type_mapping)} equipment type features")

        # Show distribution
        eq_type_dist = df_equipment[equipment_type_col].value_counts().head(5)
        print(f"   ✓ Top 5 equipment types in prediction data:")
        for eq_type, count in eq_type_dist.items():
            print(f"      - {eq_type}: {count:,} ({count/len(df_equipment)*100:.1f}%)")
    else:
        print(f"   ⚠️  Equipment type column '{equipment_type_col}' not found in data")
        print(f"   ℹ️  Will add equipment type features with zeros")

        # Add equipment type features with zeros
        for feature_name in equipment_type_info['equipment_type_features']:
            df_equipment[feature_name] = 0

# Check which features are available
missing_features = [f for f in feature_names if f not in df_equipment.columns]
available_features = [f for f in feature_names if f in df_equipment.columns]

if missing_features:
    print(f"   ⚠️  Missing {len(missing_features)} features:")
    for f in missing_features[:5]:
        print(f"       - {f}")
    if len(missing_features) > 5:
        print(f"       ... and {len(missing_features) - 5} more")
    print(f"   ℹ️  Will add missing features with median values")

# Extract features
X = df_equipment[available_features].copy()

# Handle missing features
for feat in missing_features:
    X[feat] = 0  # Add missing features with zeros

# Reorder to match training
X = X[feature_names]

# Fill missing values
X_filled = X.fillna(X.median())
print(f"   ✓ Features prepared: {X_filled.shape}")

# ============================================================================
# 4. GENERATE PREDICTIONS
# ============================================================================
print("\n[4/6] Generating Multi-Horizon Predictions...")

# Scale features
X_scaled = pd.DataFrame(
    scaler.transform(X_filled),
    columns=feature_names,
    index=X_filled.index
)

# Get survival functions
print("   Computing survival probabilities...")
surv_funcs = model.predict_survival_function(X_scaled)

# Extract risk at each horizon
for horizon_name, horizon_days in horizons.items():
    survival_probs = [fn(horizon_days) for fn in surv_funcs]
    risk_probs = [1 - s for s in survival_probs]

    df_equipment[f'Risk_{horizon_name}'] = risk_probs

    high_risk_count = sum(r > 0.6 for r in risk_probs)
    print(f"   ✓ {horizon_name:10s}: Mean risk = {np.mean(risk_probs):.1%}, "
          f"High risk (>0.6) = {high_risk_count:,} ({high_risk_count/len(risk_probs)*100:.1f}%)")

# Add risk categories (based on 12-month)
df_equipment['Risk_Category_12m'] = pd.cut(
    df_equipment['Risk_12_month'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

# ============================================================================
# 5. PRIORITIZATION & ACTIONABLE INSIGHTS
# ============================================================================
print("\n[5/6] Creating Prioritization Lists...")

# Define priority tiers
priority_tiers = {
    'IMMEDIATE': df_equipment['Risk_3_month'] > 0.7,
    'HIGH': (df_equipment['Risk_6_month'] > 0.6) & (df_equipment['Risk_3_month'] <= 0.7),
    'MEDIUM': (df_equipment['Risk_12_month'] > 0.5) & (df_equipment['Risk_6_month'] <= 0.6),
    'LOW': (df_equipment['Risk_12_month'] > 0.3) & (df_equipment['Risk_12_month'] <= 0.5),
    'MONITOR': df_equipment['Risk_12_month'] <= 0.3
}

df_equipment['Priority_Tier'] = 'MONITOR'
for tier, condition in priority_tiers.items():
    df_equipment.loc[condition, 'Priority_Tier'] = tier

# Print distribution
print("\n   📊 Priority Distribution:")
for tier in ['IMMEDIATE', 'HIGH', 'MEDIUM', 'LOW', 'MONITOR']:
    count = (df_equipment['Priority_Tier'] == tier).sum()
    print(f"      {tier:10s}: {count:6,} equipment ({count/len(df_equipment)*100:5.1f}%)")

# Create actionable lists
priority_order = ['IMMEDIATE', 'HIGH', 'MEDIUM', 'LOW', 'MONITOR']
df_equipment['Priority_Rank'] = df_equipment['Priority_Tier'].apply(
    lambda x: priority_order.index(x) if x in priority_order else 999
)

# Sort by priority
df_prioritized = df_equipment.sort_values(
    ['Priority_Rank', 'Risk_3_month', 'Risk_6_month', 'Risk_12_month'],
    ascending=[True, False, False, False]
)

# ============================================================================
# 6. SAVE OUTPUTS
# ============================================================================
print("\n[6/6] Saving Outputs...")

# Save full predictions
output_file = OUTPUT_DIR + 'step5_all_equipment_predictions.xlsx'

# Create Excel writer with multiple sheets
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:

    # Sheet 1: All equipment with predictions
    cols_to_save = ['Ekipman Kodu', 'Ekipman_Yaşı_Yıl',
                    'Risk_3_month', 'Risk_6_month', 'Risk_12_month', 'Risk_24_month',
                    'Priority_Tier', 'Risk_Category_12m']

    # Add optional columns if they exist
    optional_cols = ['Equipment_Type', 'Ekipman Sınıfı', 'İlçe', 'Fider ID',
                     'Toplam_Müşteri_Sayısı', 'Arıza_Sayısı_12ay']
    for col in optional_cols:
        if col in df_prioritized.columns:
            cols_to_save.append(col)

    df_prioritized[cols_to_save].to_excel(writer, sheet_name='All_Equipment', index=False)

    # Sheet 2: IMMEDIATE action (3-month high risk)
    immediate = df_prioritized[df_prioritized['Priority_Tier'] == 'IMMEDIATE']
    immediate[cols_to_save].to_excel(writer, sheet_name='IMMEDIATE_Action', index=False)

    # Sheet 3: HIGH priority (6-month high risk)
    high = df_prioritized[df_prioritized['Priority_Tier'] == 'HIGH']
    high[cols_to_save].to_excel(writer, sheet_name='HIGH_Priority', index=False)

    # Sheet 4: MEDIUM priority (12-month moderate risk)
    medium = df_prioritized[df_prioritized['Priority_Tier'] == 'MEDIUM']
    medium[cols_to_save].to_excel(writer, sheet_name='MEDIUM_Priority', index=False)

    # Sheet 5: Summary statistics
    summary_data = {
        'Metric': [],
        'Value': []
    }

    summary_data['Metric'].extend([
        'Total Equipment',
        'IMMEDIATE Action',
        'HIGH Priority',
        'MEDIUM Priority',
        'LOW Priority',
        'MONITOR Only',
        '',
        'Mean Risk (3-month)',
        'Mean Risk (6-month)',
        'Mean Risk (12-month)',
        'Mean Risk (24-month)',
        '',
        'Model Type',
        'C-index',
        'Features Used',
        'Predicted On'
    ])

    summary_data['Value'].extend([
        len(df_equipment),
        (df_equipment['Priority_Tier'] == 'IMMEDIATE').sum(),
        (df_equipment['Priority_Tier'] == 'HIGH').sum(),
        (df_equipment['Priority_Tier'] == 'MEDIUM').sum(),
        (df_equipment['Priority_Tier'] == 'LOW').sum(),
        (df_equipment['Priority_Tier'] == 'MONITOR').sum(),
        '',
        f"{df_equipment['Risk_3_month'].mean():.1%}",
        f"{df_equipment['Risk_6_month'].mean():.1%}",
        f"{df_equipment['Risk_12_month'].mean():.1%}",
        f"{df_equipment['Risk_24_month'].mean():.1%}",
        '',
        metadata['model_type'],
        metadata['c_index'],
        metadata['n_features'],
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ])

    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

print(f"   ✓ Saved: {output_file}")
print(f"     - All_Equipment sheet: {len(df_equipment):,} records")
print(f"     - IMMEDIATE_Action sheet: {len(immediate):,} records")
print(f"     - HIGH_Priority sheet: {len(high):,} records")
print(f"     - MEDIUM_Priority sheet: {len(medium):,} records")

# Save simple CSV for easy import
csv_file = OUTPUT_DIR + 'step5_equipment_risks_simple.csv'
df_prioritized[cols_to_save].to_csv(csv_file, index=False)
print(f"   ✓ Saved CSV: {csv_file}")

# Create visualization
print("\n   Creating risk distribution visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Multi-Horizon Risk Distribution', fontsize=16, fontweight='bold')

for idx, (horizon_name, horizon_days) in enumerate(horizons.items()):
    ax = axes[idx // 2, idx % 2]

    col = f'Risk_{horizon_name}'
    data = df_equipment[col]

    # Histogram
    ax.hist(data, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.2f}')
    ax.axvline(0.6, color='orange', linestyle='--', linewidth=2, label='High Risk Threshold')

    ax.set_xlabel('Risk Score', fontsize=11)
    ax.set_ylabel('Equipment Count', fontsize=11)
    ax.set_title(f'{horizon_name.replace("_", " ").title()} ({horizon_days} days)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_file = OUTPUT_DIR + 'step5_risk_distribution.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved plot: {plot_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ STEP 5: MULTI-HORIZON PREDICTIONS COMPLETED")
print("=" * 80)

print(f"""
📊 PREDICTION SUMMARY:
  • Total Equipment Scored: {len(df_equipment):,}

  • Priority Breakdown:
    - IMMEDIATE Action (3m risk >0.7):  {(df_equipment['Priority_Tier'] == 'IMMEDIATE').sum():,}
    - HIGH Priority (6m risk >0.6):     {(df_equipment['Priority_Tier'] == 'HIGH').sum():,}
    - MEDIUM Priority (12m risk >0.5):  {(df_equipment['Priority_Tier'] == 'MEDIUM').sum():,}
    - LOW Priority:                     {(df_equipment['Priority_Tier'] == 'LOW').sum():,}
    - MONITOR Only:                     {(df_equipment['Priority_Tier'] == 'MONITOR').sum():,}

  • Average Risks:
    - 3-month:  {df_equipment['Risk_3_month'].mean():.1%}
    - 6-month:  {df_equipment['Risk_6_month'].mean():.1%}
    - 12-month: {df_equipment['Risk_12_month'].mean():.1%}
    - 24-month: {df_equipment['Risk_24_month'].mean():.1%}

📁 OUTPUT FILES:
  • {output_file}
    → Multi-sheet Excel with prioritized equipment lists
  • {csv_file}
    → Simple CSV for easy import
  • {plot_file}
    → Risk distribution visualization

🎯 DELIVERABLES READY:
  ✓ Multi-horizon predictions (3m, 6m, 12m, 24m)
  ✓ Survival analysis-based risk scores
  ✓ Prioritized action lists
  ✓ Equipment-level scoring for entire population
""")

print("=" * 80)
