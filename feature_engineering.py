"""
STEP 2: Feature Engineering
============================
Amaç:
- Son bakım tarihi hesaplama (Checklist/İş Emri verilerinden)
- Arıza geçmişi agregasyonu (12ay, 6ay, 3ay)
- Proxy yüklenme değişkenleri (bağlı müşteri, fider yoğunluğu)
- Marka/Model extraction
- Coğrafi faktörler (kentsel/kırsal)
- Temporal features (mevsim, tatil, hava durumu proxy)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = 'outputs/step1_processed_data.xlsx'
OUTPUT_DIR = 'outputs/'
REFERENCE_DATE = datetime.now()

print("=" * 80)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 80)
print(f"Reference Date: {REFERENCE_DATE.strftime('%Y-%m-%d')}\n")

# ============================================================================
# 1. LOAD PROCESSED DATA FROM STEP 1
# ============================================================================
print("[1/8] Loading processed data from STEP 1...")
try:
    df = pd.read_excel(INPUT_FILE, engine='openpyxl')
    print(f"✓ Data loaded: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
except Exception as e:
    print(f"✗ Error: {e}")
    print("Trying CSV format...")
    df = pd.read_csv(INPUT_FILE.replace('.xlsx', '.csv'))
    print(f"✓ Data loaded: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

# Parse dates if needed
if df['Arıza_Tarihi'].dtype == 'object':
    df['Arıza_Tarihi'] = pd.to_datetime(df['Arıza_Tarihi'])
if 'Tamamlanma_Tarihi' in df.columns:
    df['Tamamlanma_Tarihi_parsed'] = pd.to_datetime(df['Tamamlanma_Tarihi'], errors='coerce')

# ============================================================================
# 2. ARIZA GEÇMİŞİ FEATURES (Fault History)
# ============================================================================
print("[2/8] Calculating Fault History Features...")

# Sort by equipment and date
df_sorted = df.sort_values(['Ekipman Kodu', 'Arıza_Tarihi'])

# Calculate fault counts for each equipment
fault_history = []

for idx, row in df_sorted.iterrows():
    ekipman = row['Ekipman Kodu']
    ariza_tarihi = row['Arıza_Tarihi']
    
    # Get all previous faults for this equipment
    prev_faults = df_sorted[
        (df_sorted['Ekipman Kodu'] == ekipman) & 
        (df_sorted['Arıza_Tarihi'] < ariza_tarihi)
    ]
    
    # Count faults in different time windows
    date_12m = ariza_tarihi - timedelta(days=365)
    date_6m = ariza_tarihi - timedelta(days=182)
    date_3m = ariza_tarihi - timedelta(days=91)
    
    faults_12m = len(prev_faults[prev_faults['Arıza_Tarihi'] >= date_12m])
    faults_6m = len(prev_faults[prev_faults['Arıza_Tarihi'] >= date_6m])
    faults_3m = len(prev_faults[prev_faults['Arıza_Tarihi'] >= date_3m])
    faults_total = len(prev_faults)
    
    # Calculate days since last fault
    if len(prev_faults) > 0:
        last_fault_date = prev_faults['Arıza_Tarihi'].max()
        days_since_last = (ariza_tarihi - last_fault_date).days
    else:
        days_since_last = None
    
    # Calculate MTBF (Mean Time Between Failures)
    if faults_total >= 2:
        time_span = (ariza_tarihi - prev_faults['Arıza_Tarihi'].min()).days
        mtbf = time_span / faults_total
    else:
        mtbf = None
    
    fault_history.append({
        'Arıza_Sayısı_12ay': faults_12m,
        'Arıza_Sayısı_6ay': faults_6m,
        'Arıza_Sayısı_3ay': faults_3m,
        'Toplam_Arıza_Sayısı': faults_total,
        'Son_Arızadan_Gün': days_since_last,
        'MTBF_Gün': mtbf,
        'Tekrarlayan_Arıza_Flag': 1 if faults_3m >= 2 else 0
    })

# Add to dataframe
fault_df = pd.DataFrame(fault_history)
df = pd.concat([df.reset_index(drop=True), fault_df], axis=1)

print(f"✓ Fault history features created:")
print(f"  - Arıza_Sayısı_12ay: Mean = {df['Arıza_Sayısı_12ay'].mean():.2f}")
print(f"  - Arıza_Sayısı_6ay: Mean = {df['Arıza_Sayısı_6ay'].mean():.2f}")
print(f"  - Arıza_Sayısı_3ay: Mean = {df['Arıza_Sayısı_3ay'].mean():.2f}")
print(f"  - Tekrarlayan Arıza: {df['Tekrarlayan_Arıza_Flag'].sum():,} cases ({df['Tekrarlayan_Arıza_Flag'].sum()/len(df)*100:.1f}%)\n")

# ============================================================================
# 3. SON BAKIM TARİHİ (Last Maintenance Date)
# ============================================================================
print("[3/8] Calculating Last Maintenance Date...")

# Group by equipment and find last maintenance-related work order
# Looking for work orders with maintenance keywords
maintenance_keywords = ['Bakım', 'bakım', 'BAKIM', 'Periyodik', 'Önleyici', 'Koruyucu']

if 'İş Emri Tipi' in df.columns:
    # Create maintenance flag
    df['Is_Maintenance'] = df['İş Emri Tipi'].str.contains('|'.join(maintenance_keywords), na=False, case=False)
    
    # For each row, find last maintenance before this fault
    last_maintenance = []
    
    for idx, row in df.iterrows():
        ekipman = row['Ekipman Kodu']
        ariza_tarihi = row['Arıza_Tarihi']
        
        # Find last maintenance for this equipment before fault
        prev_maintenance = df[
            (df['Ekipman Kodu'] == ekipman) & 
            (df['Arıza_Tarihi'] < ariza_tarihi) &
            (df['Is_Maintenance'] == True)
        ]
        
        if len(prev_maintenance) > 0:
            last_maint_date = prev_maintenance['Arıza_Tarihi'].max()
            days_since_maint = (ariza_tarihi - last_maint_date).days
        else:
            days_since_maint = None
        
        last_maintenance.append(days_since_maint)
    
    df['Son_Bakım_Gün_Sayısı'] = last_maintenance
    
    valid_maint = df['Son_Bakım_Gün_Sayısı'].notna().sum()
    print(f"✓ Last maintenance calculated for {valid_maint:,} records ({valid_maint/len(df)*100:.1f}%)")
    if valid_maint > 0:
        print(f"  - Mean days since maintenance: {df['Son_Bakım_Gün_Sayısı'].mean():.1f}")
        print(f"  - Median: {df['Son_Bakım_Gün_Sayısı'].median():.1f}\n")
else:
    df['Son_Bakım_Gün_Sayısı'] = None
    print("⚠ İş Emri Tipi column not found, skipping maintenance calculation\n")

# ============================================================================
# 4. PROXY YÜKLENME DEĞİŞKENLERİ (Customer-based Loading Proxies)
# ============================================================================
print("[4/8] Creating Customer-based Loading Variables...")

# Customer count columns (already in data!)
customer_columns = {
    'total customer count': 'Toplam_Müşteri_Sayısı',
    'urban mv': 'Kentsel_OG_Müşteri',
    'urban lv': 'Kentsel_AG_Müşteri',
    'suburban mv': 'Kentaltı_OG_Müşteri',
    'suburban lv': 'Kentaltı_AG_Müşteri',
    'rural mv': 'Kırsal_OG_Müşteri',
    'rural lv': 'Kırsal_AG_Müşteri'
}

for orig_col, new_col in customer_columns.items():
    if orig_col in df.columns:
        df[new_col] = pd.to_numeric(df[orig_col], errors='coerce').fillna(0)
    else:
        df[new_col] = 0
        print(f"  ⚠️ {orig_col} not found, using 0")

# Calculate customer type ratios
df['Kentsel_Müşteri_Oranı'] = (df['Kentsel_OG_Müşteri'] + df['Kentsel_AG_Müşteri']) / (df['Toplam_Müşteri_Sayısı'] + 1)
df['Kırsal_Müşteri_Oranı'] = (df['Kırsal_OG_Müşteri'] + df['Kırsal_AG_Müşteri']) / (df['Toplam_Müşteri_Sayısı'] + 1)
df['OG_Müşteri_Oranı'] = (df['Kentsel_OG_Müşteri'] + df['Kentaltı_OG_Müşteri'] + df['Kırsal_OG_Müşteri']) / (df['Toplam_Müşteri_Sayısı'] + 1)

print(f"✓ Customer count features created:")
print(f"  - Total customers: Mean = {df['Toplam_Müşteri_Sayısı'].mean():.1f}, Max = {df['Toplam_Müşteri_Sayısı'].max():.0f}")
print(f"  - Urban ratio: Mean = {df['Kentsel_Müşteri_Oranı'].mean():.2%}")
print(f"  - Rural ratio: Mean = {df['Kırsal_Müşteri_Oranı'].mean():.2%}")

# Calculate fider density (faults per fider)
if 'Fider ID' in df.columns:
    fider_stats = df.groupby('Fider ID').agg({
        'id': 'count',  # Number of faults
        'Toplam_Müşteri_Sayısı': 'sum'  # Total customers
    }).rename(columns={'id': 'Fider_Arıza_Sayısı', 'Toplam_Müşteri_Sayısı': 'Fider_Toplam_Müşteri'})
    
    df = df.merge(fider_stats, left_on='Fider ID', right_index=True, how='left', suffixes=('', '_fider'))
    
    print(f"✓ Fider-based features created:")
    print(f"  - Unique fiders: {df['Fider ID'].nunique()}")
    print(f"  - Fider mean customers: {df['Fider_Toplam_Müşteri'].mean():.1f}")
else:
    print("⚠️ Fider ID not found")

# Equipment load proxy (based on fault frequency and customer count)
df['Ekipman_Yoğunluk_Skoru'] = df['Arıza_Sayısı_12ay'] / (df['Ekipman_Yaşı_Yıl'] + 1)
df['Müşteri_Başına_Arıza'] = df['Arıza_Sayısı_12ay'] / (df['Toplam_Müşteri_Sayısı'] + 1)

print(f"  - Ekipman_Yoğunluk_Skoru: Mean = {df['Ekipman_Yoğunluk_Skoru'].mean():.3f}")
print(f"  - Müşteri_Başına_Arıza: Mean = {df['Müşteri_Başına_Arıza'].mean():.4f}\n")

# ============================================================================
# 5. MARKA & MODEL EXTRACTION
# ============================================================================
print("[5/8] Extracting Equipment Brand & Model...")

def extract_brand_model(equipment_desc):
    """Extract brand and model from equipment description"""
    if pd.isna(equipment_desc):
        return None, None, None
    
    desc = str(equipment_desc).upper()
    
    # Common brands
    brands = ['SIEMENS', 'ABB', 'SCHNEIDER', 'AREVA', 'EATON', 'BEST', 'ORMAZABAL', 
              'TAMINI', 'TRAFOMAK', 'ÇESAN', 'ERMAK', 'AKSA']
    
    brand = None
    for b in brands:
        if b in desc:
            brand = b
            break
    
    # Extract kVA rating
    import re
    kva_match = re.search(r'(\d+)\s*KVA', desc)
    kva_rating = kva_match.group(1) if kva_match else None
    
    # Extract voltage
    voltage_match = re.search(r'(\d+\.?\d*)/(\d+\.?\d*)\s*KV', desc)
    voltage = f"{voltage_match.group(1)}/{voltage_match.group(2)}" if voltage_match else None
    
    return brand, kva_rating, voltage

# Apply extraction
if 'Ekipman Tanımı' in df.columns:
    extraction = df['Ekipman Tanımı'].apply(extract_brand_model)
    df['Marka'] = extraction.apply(lambda x: x[0])
    df['kVA_Rating'] = extraction.apply(lambda x: x[1])
    df['Voltaj'] = extraction.apply(lambda x: x[2])
    
    brand_count = df['Marka'].notna().sum()
    print(f"✓ Brand extraction: {brand_count:,} records ({brand_count/len(df)*100:.1f}%)")
    print(f"  - Unique brands: {df['Marka'].nunique()}")
    if df['Marka'].notna().sum() > 0:
        print(f"  - Top brands: {df['Marka'].value_counts().head(3).to_dict()}")
else:
    print("⚠ Ekipman Tanımı not found\n")

print()

# ============================================================================
# 6. COĞRAFİ FAKTÖRLER (Geographic Factors)
# ============================================================================
print("[6/8] Creating Geographic Features...")

# Urban vs Rural classification
urban_districts = [
    'ALAŞEHİR', 'SALIHLI', 'SALİHLİ',  # Büyük/küçük harf varyasyonları
    'GÖRDES', 'GORDES',
    'Alaşehir', 'Salihli', 'Gördes'  # Title case
]
if 'İlçe' in df.columns:
    df['Bölge_Tipi'] = df['İlçe'].apply(
        lambda x: 'Kentsel' if str(x).upper() in urban_districts else 'Kırsal'
    )
    
    print(f"✓ Geographic classification:")
    print(f"  - Kentsel: {(df['Bölge_Tipi'] == 'Kentsel').sum():,} ({(df['Bölge_Tipi'] == 'Kentsel').sum()/len(df)*100:.1f}%)")
    print(f"  - Kırsal: {(df['Bölge_Tipi'] == 'Kırsal').sum():,} ({(df['Bölge_Tipi'] == 'Kırsal').sum()/len(df)*100:.1f}%)\n")

# Coordinate-based features (if available)
if 'KOORDINAT_X' in df.columns and 'KOORDINAT_Y' in df.columns:
    # Create grid zones for spatial analysis
    df['Coord_Zone'] = (
        df['KOORDINAT_X'].astype(str).str[:4] + '_' + 
        df['KOORDINAT_Y'].astype(str).str[:4]
    )
    print(f"  - Coordinate zones created: {df['Coord_Zone'].nunique()} unique zones\n")

# ============================================================================
# 7. TEMPORAL FEATURES
# ============================================================================
print("[7/8] Creating Temporal Features...")

df['Yıl'] = df['Arıza_Tarihi'].dt.year
df['Ay'] = df['Arıza_Tarihi'].dt.month
df['Gün'] = df['Arıza_Tarihi'].dt.day
df['Haftanın_Günü'] = df['Arıza_Tarihi'].dt.dayofweek  # 0=Monday, 6=Sunday
df['Hafta_İçi'] = df['Haftanın_Günü'].apply(lambda x: 1 if x < 5 else 0)

# Season
def get_season(month):
    if month in [12, 1, 2]:
        return 'Kış'
    elif month in [3, 4, 5]:
        return 'İlkbahar'
    elif month in [6, 7, 8]:
        return 'Yaz'
    else:
        return 'Sonbahar'

df['Mevsim'] = df['Ay'].apply(get_season)

# Hour (if time available)
if df['Arıza_Tarihi'].dt.hour.notna().sum() > 0:
    df['Saat'] = df['Arıza_Tarihi'].dt.hour
    df['Gece_Gündüz'] = df['Saat'].apply(lambda x: 'Gündüz' if 6 <= x < 18 else 'Gece')
    print(f"✓ Temporal features created (including hour)")
else:
    print(f"✓ Temporal features created (date only)")

print(f"  - Year range: {df['Yıl'].min()} - {df['Yıl'].max()}")
print(f"  - Mevsim distribution:")
for season, count in df['Mevsim'].value_counts().items():
    print(f"    {season}: {count:,} ({count/len(df)*100:.1f}%)\n")

# ============================================================================
# 8. SUMMARY STATISTICS
# ============================================================================
print("=" * 80)
print("[8/8] FEATURE ENGINEERING SUMMARY")
print("=" * 80)

new_features = [
    'Arıza_Sayısı_12ay', 'Arıza_Sayısı_6ay', 'Arıza_Sayısı_3ay',
    'Son_Arızadan_Gün', 'MTBF_Gün', 'Tekrarlayan_Arıza_Flag',
    'Son_Bakım_Gün_Sayısı', 
    'Toplam_Müşteri_Sayısı', 'Kentsel_Müşteri_Oranı', 'Kırsal_Müşteri_Oranı',
    'Ekipman_Yoğunluk_Skoru', 'Müşteri_Başına_Arıza',
    'Marka', 'kVA_Rating', 'Bölge_Tipi',
    'Mevsim', 'Hafta_İçi', 'Yıl', 'Ay'
]

print("\n📊 NEW FEATURES CREATED:")
for i, feature in enumerate(new_features, 1):
    if feature in df.columns:
        non_null = df[feature].notna().sum()
        print(f"  {i:2d}. {feature:30s}: {non_null:6,} non-null ({non_null/len(df)*100:5.1f}%)")

print("\n📈 KEY STATISTICS:")
print(f"  Total rows: {len(df):,}")
print(f"  Total features: {len(df.columns)}")
print(f"  New features: {len([f for f in new_features if f in df.columns])}")

print("\n⭐ KESICI EQUIPMENT FOCUS:")
kesici_mask = df['Ekipman Sınıfı'].str.contains('Kesici', na=False, case=False)
kesici_df = df[kesici_mask]
print(f"  Kesici count: {len(kesici_df):,}")
print(f"  Mean faults (12m): {kesici_df['Arıza_Sayısı_12ay'].mean():.2f}")
print(f"  Mean age: {kesici_df['Ekipman_Yaşı_Yıl'].mean():.1f} years")
print(f"  Repeating faults: {kesici_df['Tekrarlayan_Arıza_Flag'].sum():,}")

# ============================================================================
# 9. SAVE PROCESSED DATA
# ============================================================================
print("\n" + "=" * 80)
print("SAVING PROCESSED DATA")
print("=" * 80)

output_file = OUTPUT_DIR + 'step2_feature_engineered_data.xlsx'
try:
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"✓ Saved to: {output_file}")
except Exception as e:
    print(f"Warning: Could not save as Excel: {e}")
    output_file = OUTPUT_DIR + 'step2_feature_engineered_data.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✓ Saved to: {output_file}")

# Save feature summary
summary_file = OUTPUT_DIR + 'step2_feature_summary.txt'
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("STEP 2: FEATURE ENGINEERING SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Records: {len(df):,}\n")
    f.write(f"Total Features: {len(df.columns)}\n\n")
    f.write("New Features:\n")
    for feature in new_features:
        if feature in df.columns:
            f.write(f"  - {feature}\n")
    f.write("\nKey Statistics:\n")
    f.write(f"  Mean faults (12m): {df['Arıza_Sayısı_12ay'].mean():.2f}\n")
    f.write(f"  Repeating faults: {df['Tekrarlayan_Arıza_Flag'].sum():,}\n")
    f.write(f"  Kesici equipment: {kesici_mask.sum():,}\n")

print(f"✓ Summary saved to: {summary_file}")

print("\n" + "=" * 80)
print("✓ STEP 2 COMPLETED SUCCESSFULLY")
print("=" * 80)
print("\n✅ Ready for STEP 3: Exploratory Data Analysis & Modeling\n")