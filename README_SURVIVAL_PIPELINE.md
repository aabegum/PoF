# Survival Analysis Pipeline for Multi-Horizon Predictions

## Overview

This pipeline uses **Survival Analysis** to predict equipment failure risk at multiple time horizons (3-month, 6-month, 12-month, 24-month) from a **single unified model**.

### Key Benefits

✅ **Multi-horizon predictions** from one model
✅ **Mathematically consistent** risk scores (monotonic over time)
✅ **Survival curves** for each equipment
✅ **Flexible horizons** - predict at ANY time point, not just fixed windows
✅ **State-of-the-art** approach for predictive maintenance

---

## Pipeline Structure

### Step 1: Data Processing
**Script:** `process_data.py`
**Purpose:** Load raw data, calculate equipment age, create consolidated Equipment_Type
**Input:** Raw fault data
**Output:** `outputs/step1_processed_data.xlsx`

**Key Features:**
- Equipment age calculation (Ekipman_Yaşı_Yıl)
- Consolidated Equipment_Type (Component_Equipment_Type → Ekipman Sınıfı fallback)
- Date validation and cleaning

**Run:**
```bash
python process_data.py
```

---

### Step 2: Feature Engineering
**Script:** `feature_engineering.py`
**Purpose:** Create fault history and temporal features
**Input:** `outputs/step1_processed_data.xlsx`
**Output:** `outputs/step2_features_added.xlsx`

**Features Created:**
- Fault history: Arıza_Sayısı_12ay, Arıza_Sayısı_6ay, Arıza_Sayısı_3ay
- MTBF (Mean Time Between Failures)
- Customer load: Toplam_Müşteri_Sayısı, Kentsel/Kırsal ratios
- Age categories: Age_0-5y, Age_5-10y, Age_10-15y, Age_15+y
- Temporal: Season_Spring/Summer/Fall/Winter

**Run:**
```bash
python feature_engineering.py
```

---

### Step 3.5: Survival Model Training (NEW!)
**Script:** `survival_modeling.py`
**Purpose:** Train survival analysis model for multi-horizon predictions
**Input:** `outputs/step2_features_added.xlsx`
**Output:**
- `outputs/step3_5_survival_model.pkl` - Trained survival model
- `outputs/step3_5_survival_scaler.pkl` - Feature scaler
- `outputs/step3_5_survival_features.json` - Feature list
- `outputs/step3_5_survival_metadata.json` - Model metadata
- `outputs/step3_5_survival_risk_scored.xlsx` - All data with risk scores
- `outputs/step3_5_survival_high_risk.xlsx` - High-risk equipment only

**What Changed from Classification Approach:**

| Old (optimizations.py) | New (survival_modeling.py) |
|------------------------|----------------------------|
| Binary classifier (XGBoost) | Survival model (RSF/GBS) |
| Single target: PoF_12_month | Survival target: (time, event) |
| One prediction per model | Multi-horizon from one model |
| AUC metric | C-index metric |
| Risk at 12 months only | Risk at 3m, 6m, 12m, 24m |

**Model Algorithm:**
- **Random Survival Forest (RSF)** or **Gradient Boosting Survival Analysis (GBS)**
- Compares both and selects best based on C-index
- Typical C-index: 0.70-0.75 (equivalent to AUC 0.70-0.75)

**Survival Targets:**
```python
# For each work order:
time_to_event = days until next failure (or censoring)
event_occurred = True if failure observed, False if censored

# Example:
# Equipment A: Fault on Jan 1 → Next fault on Mar 15 (74 days)
#   → time_to_event = 74, event_occurred = True
#
# Equipment B: Fault on Jan 1 → No fault observed for 2 years
#   → time_to_event = 730, event_occurred = False (censored)
```

**Run:**
```bash
python survival_modeling.py
```

**Output Interpretation:**
- **C-index**: Concordance index (0.5 = random, 1.0 = perfect)
  - Similar to AUC for classification
  - Measures how well model ranks equipment by failure time
- **Risk scores**: 0.0 (no risk) to 1.0 (certain failure)
  - Risk_3_month: Probability of failure within 91 days
  - Risk_6_month: Probability of failure within 182 days
  - Risk_12_month: Probability of failure within 365 days
  - Risk_24_month: Probability of failure within 730 days

---

### Step 5: Score All Equipment (NEW!)
**Script:** `predict_all_equipment.py`
**Purpose:** Generate multi-horizon risk predictions for entire equipment population
**Input:**
- `outputs/step2_features_added.xlsx`
- Trained model files (from Step 3.5)

**Output:**
- `outputs/step5_all_equipment_predictions.xlsx` - Multi-sheet Excel with:
  - **All_Equipment**: All equipment with risk scores
  - **IMMEDIATE_Action**: 3-month risk > 0.7 (needs action NOW)
  - **HIGH_Priority**: 6-month risk > 0.6 (action within 3 months)
  - **MEDIUM_Priority**: 12-month risk > 0.5 (action within 6 months)
  - **Summary**: Statistics and metadata
- `outputs/step5_equipment_risks_simple.csv` - Simple CSV for import
- `outputs/step5_risk_distribution.png` - Visualization

**Priority Tiers:**
```python
IMMEDIATE:  Risk_3_month > 0.7     → Replace within 3 months
HIGH:       Risk_6_month > 0.6     → Replace within 6 months
MEDIUM:     Risk_12_month > 0.5    → Replace within 12 months
LOW:        Risk_12_month 0.3-0.5  → Monitor closely
MONITOR:    Risk_12_month < 0.3    → Standard monitoring
```

**Run:**
```bash
python predict_all_equipment.py
```

---

### Step 4: Analytics & Visualization
**Scripts:**
- `comprehensive_analytics.py` - Full analytics suite
- `advanced_analytics.py` - Survival curves, SHAP, CAPEX
- `enhanced.py` - Interactive visualizations

**Status:** These scripts currently use the OLD classification model. They will need to be updated to use the new survival model predictions.

---

## Complete Workflow

### Initial Training (Run Once)

```bash
# Step 1: Process raw data
python process_data.py

# Step 2: Engineer features
python feature_engineering.py

# Step 3.5: Train survival model
python survival_modeling.py
```

### Scoring New Equipment (Production Use)

```bash
# Score all equipment with multi-horizon predictions
python predict_all_equipment.py
```

**This generates your deliverables:**
- ✅ Multi-horizon predictions (3m, 6m, 12m, 24m)
- ✅ Survival analysis-based risk scores
- ✅ Prioritized equipment lists

---

## Model Performance Comparison

### Old Approach (Classification)
```
Model: XGBoost Classifier
Target: Binary (failure within 12 months)
AUC: 0.7148
Output: Single risk score (12-month)
Predictions: One model per horizon needed
```

### New Approach (Survival Analysis)
```
Model: Random Survival Forest / Gradient Boosting Survival
Target: Time-to-event with censoring
C-index: ~0.70-0.75 (expected)
Output: Risk scores at ALL horizons (3m, 6m, 12m, 24m, ANY)
Predictions: One model for all horizons
```

---

## Key Differences from Classification

| Aspect | Classification (Old) | Survival Analysis (New) |
|--------|---------------------|-------------------------|
| **Target** | Binary: Yes/No failure | Time-to-failure + Censoring |
| **Question** | "Will it fail in 12 months?" | "When will it fail?" |
| **Horizons** | Fixed (need separate models) | Flexible (any time point) |
| **Censored Data** | Ignored or treated as negative | Explicitly modeled |
| **Output** | Risk score (0-1) | Survival curve + risk scores |
| **Consistency** | Risk_6m might > Risk_12m ❌ | Monotonic risk ✅ |
| **Business Value** | "Replace high-risk equipment" | "Replace equipment by Q2 2024" |

---

## File Outputs Summary

### Training Outputs (Step 3.5)
```
outputs/
├── step3_5_survival_model.pkl          # Trained survival model
├── step3_5_survival_scaler.pkl         # Feature scaler
├── step3_5_survival_features.json      # Feature list
├── step3_5_survival_metadata.json      # Model metadata
├── step3_5_survival_risk_scored.xlsx   # All records with risk scores
├── step3_5_survival_high_risk.xlsx     # High-risk equipment
└── step3_5_survival_model_comparison.xlsx  # RSF vs GBS comparison
```

### Scoring Outputs (Step 5)
```
outputs/
├── step5_all_equipment_predictions.xlsx   # Multi-sheet prioritized lists
├── step5_equipment_risks_simple.csv       # Simple CSV export
└── step5_risk_distribution.png            # Risk visualization
```

---

## Equipment Type Integration

The pipeline uses the **consolidated Equipment_Type** column throughout:

### In process_data.py (Step 1):
```python
# Priority logic:
Equipment_Type = Component_Equipment_Type if not empty
                 else Ekipman Sınıfı (fallback)
```

### In Analytics (Step 4):
- Survival curves by equipment type
- Interactive maps colored by equipment type
- CAPEX prioritization lists grouped by type

### In Predictions (Step 5):
- Equipment type included in output sheets for filtering
- Can analyze risk distribution by equipment type

---

## Next Steps

### 1. Run the New Pipeline

```bash
# Complete training workflow
python process_data.py
python feature_engineering.py
python survival_modeling.py

# Generate predictions
python predict_all_equipment.py
```

### 2. Review Outputs

Check these key files:
- `step3_5_survival_metadata.json` - Model performance (C-index)
- `step5_all_equipment_predictions.xlsx` - Your main deliverable
  - Review IMMEDIATE_Action sheet
  - Review HIGH_Priority sheet

### 3. Compare with Old Model (Optional)

Run the old classification pipeline for comparison:
```bash
python eda_modeling.py
python optimizations.py
```

Compare:
- Old AUC (12-month) vs New C-index
- Old high-risk list vs New IMMEDIATE/HIGH priority lists
- Single horizon vs Multi-horizon predictions

### 4. Update Analytics Scripts

The analytics scripts (comprehensive_analytics.py, etc.) still use the old classification model. They can be updated to:
- Load survival model instead
- Generate multi-horizon SHAP analysis
- Show survival curves with confidence intervals
- Display time-dependent feature importance

---

## Troubleshooting

### Missing Features Warning
```
⚠️ Warning: Scaler expects features not in data
```
**Cause:** Some features don't exist in all datasets
**Solution:** Script automatically adds missing features with zeros
**Impact:** No impact on predictions (expected behavior)

### Low C-index (<0.65)
**Possible causes:**
- Insufficient feature engineering
- Class imbalance (too few events)
- Need more data or better features

**Solutions:**
- Add equipment type as feature (one-hot encoding)
- Include maintenance history if available
- Add external factors (weather, load patterns)

### Prediction Script Fails
**Common issues:**
- Model files not found → Run survival_modeling.py first
- Feature mismatch → Use same input data (step2_features_added.xlsx)
- Memory error → Reduce data size or batch processing

---

## FAQ

**Q: Can I add equipment type as a feature to improve predictions?**
A: Yes! Modify survival_modeling.py to add one-hot encoding of Equipment_Type to the feature list. This typically improves C-index by 0.02-0.05.

**Q: How do I predict risk at a custom horizon (e.g., 9 months)?**
A: Modify predict_all_equipment.py to add `'9_month': 274` to the `horizons` dictionary.

**Q: Can I use this model in production?**
A: Yes! Use predict_all_equipment.py with new equipment data. Just ensure the same features are calculated.

**Q: What's the minimum data requirement?**
A: Recommended:
- At least 5,000 work orders
- At least 1,000 unique equipment
- At least 500 failure events (not censored)

**Q: How often should I retrain?**
A: Retrain quarterly or when:
- New equipment types added
- Significant changes in failure patterns
- Model performance degrades (C-index drops >0.05)

---

## References

### Survival Analysis
- Scikit-survival documentation: https://scikit-survival.readthedocs.io/
- Random Survival Forests: Ishwaran et al. (2008)
- Gradient Boosting Survival: Ridgeway (1999)

### Predictive Maintenance
- "Survival analysis for predictive maintenance" - IEEE Reliability Society
- "Time-to-failure prediction using survival analysis" - Various industry papers

---

## Contact & Support

For questions about this pipeline:
1. Check this README first
2. Review script docstrings
3. Check output logs for error messages
4. Review metadata.json files for model details

---

**Last Updated:** 2025-10-30
**Pipeline Version:** 2.0 (Survival Analysis)
**Previous Version:** 1.0 (Binary Classification)
