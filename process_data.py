"""
STEP 1: Veri Hazırlama ve Ekipman Yaşı Hesaplama (IMPROVED)
=============================================================
İyileştirmeler:
- 1900 tarihlerini invalid olarak ele al
- "First work order" yerine "Oluşturma Tarihi Sıralama" kullan
- Daha sağlam tarih validasyonu
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = 'data/combined_data_cbs.xlsx'
REFERENCE_DATE = datetime.now()
OUTPUT_DIR = 'outputs/'

# Date validation parameters
MIN_VALID_YEAR = 1950  # 1950'den eski tarihler invalid
MAX_VALID_YEAR = datetime.now().year + 1

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("STEP 1: DATA PREPARATION & EQUIPMENT AGE CALCULATION (IMPROVED)")
print("=" * 80)
print(f"Reference Date: {REFERENCE_DATE.strftime('%Y-%m-%d')}")
print(f"Valid Year Range: {MIN_VALID_YEAR} - {MAX_VALID_YEAR}\n")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("[1/6] Loading data...")
try:
    df = pd.read_excel(INPUT_FILE, engine='openpyxl')
    print(f"✓ Data loaded successfully: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit(1)

# ============================================================================
# 2. FILTER: Kesintili Arıza İE only
# ============================================================================
print("[2/6] Filtering for 'Kesintili Arıza İE'...")
original_count = len(df)
df = df[df['İş Emri Tipi'] == 'Kesintili Arıza İE'].copy()
print(f"✓ Filtered: {len(df):,} rows (from {original_count:,})")
print(f"  Filtered out: {original_count - len(df):,} rows\n")

# ============================================================================
# 3. PARSE AND VALIDATE DATES
# ============================================================================
print("[3/6] Parsing and validating date columns...")

def parse_and_validate_date(date_series, column_name):
    """Parse dates and mark invalid ones (1900, future dates, etc.)"""
    # Parse dates
    parsed = pd.to_datetime(date_series, errors='coerce', dayfirst=True)
    
    # Validation
    valid_mask = (
        parsed.notna() & 
        (parsed.dt.year >= MIN_VALID_YEAR) & 
        (parsed.dt.year <= MAX_VALID_YEAR)
    )
    
    # Count invalid 1900s
    invalid_1900 = (parsed.notna() & (parsed.dt.year < MIN_VALID_YEAR)).sum()
    
    # Set invalid dates to NaT
    parsed[~valid_mask] = pd.NaT
    
    # Statistics
    total = len(date_series)
    valid = valid_mask.sum()
    
    print(f"  {column_name:30s}:")
    print(f"    Valid: {valid:6,} ({valid/total*100:5.1f}%)")
    if invalid_1900 > 0:
        print(f"    Invalid (1900s): {invalid_1900:6,} ⚠️")
    
    return parsed

date_columns = {
    'started at': 'Arıza_Tarihi',
    'TESIS_TARIHI': 'TESIS_TARIHI_parsed',
    'EDBS_IDATE': 'EDBS_IDATE_parsed',
    'ended at': 'Arıza_Bitiş_Tarihi',
    'Oluşturulma_Tarihi': 'Oluşturulma_Tarihi'
}

for col, new_col in date_columns.items():
    if col in df.columns:
        df[new_col] = parse_and_validate_date(df[col], col)
    else:
        print(f"  {col:30s}: ⚠️ Column not found!")
        df[new_col] = pd.NaT

print()


# ============================================================================
# 4. CALCULATE EQUIPMENT AGE (IMPROVED)
# ============================================================================
print("[3B] Deriving first Oluşturulma_Tarihi per equipment...")

if 'Ekipman Kodu' in df.columns and 'Oluşturulma_Tarihi' in df.columns:
    first_creation = df.groupby('Ekipman Kodu')['Oluşturulma_Tarihi'].min().rename('First_Oluşturulma_Tarihi')
    df = df.merge(first_creation, on='Ekipman Kodu', how='left')
else:
    df['First_Oluşturulma_Tarihi'] = pd.NaT
    
    
print("[4/6] Calculating Equipment Age with improved logic...")

def calculate_equipment_age_improved(row):
    """
    Improved Priority: 
    1. TESIS_TARIHI (if valid and >= 1950)
    2. EDBS_IDATE (if valid and >= 1950)
    3. Oluşturma Tarihi (as proxy)
    Returns: (age_in_days, source_used, install_date)
    """
    ref_date = REFERENCE_DATE
    
    # Option 1: TESIS_TARIHI
    if pd.notna(row['TESIS_TARIHI_parsed']):
        install_date = row['TESIS_TARIHI_parsed']
        if install_date < ref_date:
            age_days = (ref_date - install_date).days
            return age_days, 'TESIS_TARIHI', install_date
    
    # Option 2: EDBS_IDATE
    if pd.notna(row['EDBS_IDATE_parsed']):
        install_date = row['EDBS_IDATE_parsed']
        if install_date < ref_date:
            age_days = (ref_date - install_date).days
            return age_days, 'EDBS_IDATE', install_date
    
    # Option 3: Oluşturma Tarihi Sıralama (as proxy - equipment might be older!)
    if pd.notna(row['First_Oluşturulma_Tarihi']):
        install_date = row['First_Oluşturulma_Tarihi']
        if install_date < ref_date:
            age_days = (ref_date - install_date).days
            return age_days, 'FIRST_WORKORDER_PROXY', install_date

    
    return None, 'MISSING', None

# Apply calculation
results = df.apply(calculate_equipment_age_improved, axis=1)
df['Ekipman_Yaşı_Gün'] = results.apply(lambda x: x[0])
df['Yaş_Kaynak'] = results.apply(lambda x: x[1])
df['Ekipman_Kurulum_Tarihi'] = results.apply(lambda x: x[2])

# Convert to years
df['Ekipman_Yaşı_Yıl'] = df['Ekipman_Yaşı_Gün'] / 365.25

# Statistics
print("\n📊 Equipment Age Source Distribution:")
source_counts = df['Yaş_Kaynak'].value_counts()
for source, count in source_counts.items():
    print(f"  {source:25s}: {count:6,} ({count/len(df)*100:5.1f}%)")

print("\n" + "=" * 80)

# ============================================================================
# 5. HANDLE MISSING AGES - PER EQUIPMENT BASIS
# ============================================================================
print("[5/6] Handling MISSING equipment ages using first work order per equipment...")

missing_mask = df['Yaş_Kaynak'] == 'MISSING'
missing_count_before = missing_mask.sum()

if missing_count_before > 0:
    print(f"\n⚠️ Still {missing_count_before:,} records with MISSING age")
    print("   Using earliest Oluşturma Tarihi Sıralama per equipment as proxy...\n")
    
    # Group by equipment and find first work order date
    first_wo_dates = df.groupby('Ekipman Kodu')['Oluşturulma_Tarihi'].min()
    
    filled_count = 0
    for idx in df[missing_mask].index:
        ekipman_kodu = df.loc[idx, 'Ekipman Kodu']
        if pd.notna(ekipman_kodu) and ekipman_kodu in first_wo_dates.index:
            first_date = first_wo_dates[ekipman_kodu]
            if pd.notna(first_date):
                age_days = (REFERENCE_DATE - first_date).days
                if age_days >= 0:
                    df.loc[idx, 'Ekipman_Yaşı_Gün'] = age_days
                    df.loc[idx, 'Ekipman_Yaşı_Yıl'] = age_days / 365.25
                    df.loc[idx, 'Yaş_Kaynak'] = 'FIRST_WORKORDER_PROXY'
                    df.loc[idx, 'Ekipman_Kurulum_Tarihi'] = first_date
                    filled_count += 1

    missing_count_after = (df['Yaş_Kaynak'] == 'MISSING').sum()
    print(f"✓ Filled: {filled_count:,} using first work order proxy")
    print(f"  Remaining MISSING: {missing_count_after:,}\n")

# ============================================================================
# 6. DATA QUALITY REPORT
# ============================================================================
print("=" * 80)
print("[6/6] DATA QUALITY REPORT")
print("=" * 80)

print("\n📊 OVERALL STATISTICS:")
print(f"  Total Rows:              {len(df):,}")
print(f"  Unique Equipment Codes:  {df['Ekipman Kodu'].nunique():,}")
if 'GIS_ID' in df.columns:
    print(f"  Unique GIS IDs:          {df['GIS_ID'].nunique():,}")
print(f"  Date Range:              {df['Arıza_Tarihi'].min():%Y-%m-%d} to {df['Arıza_Tarihi'].max():%Y-%m-%d}")

print("\n⚙️ EQUIPMENT AGE STATISTICS:")
age_stats = df['Ekipman_Yaşı_Yıl'].describe()
print(f"  Count:      {age_stats['count']:,.0f}")
print(f"  Mean:       {age_stats['mean']:,.1f} years")
print(f"  Median:     {age_stats['50%']:,.1f} years")
print(f"  Min:        {age_stats['min']:,.1f} years")
print(f"  Max:        {age_stats['max']:,.1f} years")
print(f"  Std Dev:    {age_stats['std']:,.1f} years")

# Age distribution
print("\n  Age Distribution:")
age_ranges = [
    (0, 5, "0-5 years (New)"),
    (5, 10, "5-10 years"),
    (10, 20, "10-20 years"),
    (20, 30, "20-30 years"),
    (30, 50, "30-50 years"),
    (50, 9999, "50+ years ⚠️")
]
for min_age, max_age, label in age_ranges:
    count = ((df['Ekipman_Yaşı_Yıl'] >= min_age) & (df['Ekipman_Yaşı_Yıl'] < max_age)).sum()
    print(f"    {label:25s}: {count:6,} ({count/len(df)*100:5.1f}%)")

print("\n🎯 EQUIPMENT CLASS DISTRIBUTION:")
eq_class_dist = df['Ekipman Sınıfı'].value_counts().head(10)
for eq_class, count in eq_class_dist.items():
    print(f"  {str(eq_class):30s}: {count:6,} ({count/len(df)*100:5.1f}%)")

print("\n🔧 KESICI EQUIPMENT CLASS (Target Equipment):")
kesici_mask = df['Ekipman Sınıfı'].str.contains('Kesici|kesici', na=False, case=False)
kesici_count = kesici_mask.sum()
print(f"  Kesici Equipment Count:  {kesici_count:,} ({kesici_count/len(df)*100:.1f}%)")
if kesici_count > 0:
    kesici_age_mean = df[kesici_mask]['Ekipman_Yaşı_Yıl'].mean()
    print(f"  Kesici Mean Age:         {kesici_age_mean:.1f} years")

print("\n📍 GEOGRAPHIC DISTRIBUTION:")
print("  Top 5 İlçe:")
for ilce, count in df['İlçe'].value_counts().head(5).items():
    print(f"    {str(ilce):20s}: {count:6,}")

print("\n⚠️ DATA QUALITY ISSUES:")
print(f"  Missing Equipment Age:       {df['Ekipman_Yaşı_Gün'].isna().sum():,} ({df['Ekipman_Yaşı_Gün'].isna().sum()/len(df)*100:.1f}%)")
print(f"  Missing Arıza_Tarihi:        {df['Arıza_Tarihi'].isna().sum():,}")
print(f"  Missing Ekipman Kodu:        {df['Ekipman Kodu'].isna().sum():,}")
if 'KOORDINAT_X' in df.columns:
    print(f"  Missing Coordinates:         {df['KOORDINAT_X'].isna().sum():,}")
print(f"  Negative Equipment Age:      {(df['Ekipman_Yaşı_Gün'] < 0).sum():,}")
print(f"  Equipment Age > 50 years:    {(df['Ekipman_Yaşı_Yıl'] > 50).sum():,}")
print(f"    → Using FIRST_WORKORDER:   {((df['Ekipman_Yaşı_Yıl'] > 50) & (df['Yaş_Kaynak'] == 'FIRST_WORKORDER_PROXY')).sum():,}")

print("\n🕐 TEMPORAL ANALYSIS:")
df['Year'] = df['Arıza_Tarihi'].dt.year
df['Month'] = df['Arıza_Tarihi'].dt.month
print("  Faults by Year:")
for year, count in df['Year'].value_counts().sort_index().items():
    print(f"    {year}: {count:6,}")

print("\n⚙️ INSTALLATION DATE ANALYSIS:")
print("  Equipment with known installation dates:")
for source in ['TESIS_TARIHI', 'EDBS_IDATE', 'FIRST_WORKORDER_PROXY']:
    count = (df['Yaş_Kaynak'] == source).sum()
    if count > 0:
        avg_age = df[df['Yaş_Kaynak'] == source]['Ekipman_Yaşı_Yıl'].mean()
        print(f"    {source:25s}: {count:6,} (avg age: {avg_age:5.1f} years)")

print("\n" + "=" * 80)
print("✓ STEP 1 COMPLETED SUCCESSFULLY")
print("=" * 80)

# ============================================================================
# 7. SAVE PROCESSED DATA
# ============================================================================
print("\nSaving processed data...")
output_file = OUTPUT_DIR + 'step1_processed_data.xlsx'
try:
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"✓ Saved to: {output_file}")
except Exception as e:
    print(f"Warning: Could not save as Excel: {e}")
    output_file = OUTPUT_DIR + 'step1_processed_data.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✓ Saved to: {output_file}")

# Save detailed summary report
summary_file = OUTPUT_DIR + 'step1_quality_report.txt'
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("STEP 1: DATA QUALITY REPORT (IMPROVED VERSION)\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Records: {len(df):,}\n\n")
    
    f.write("AGE SOURCE DISTRIBUTION:\n")
    for source, count in df['Yaş_Kaynak'].value_counts().items():
        avg_age = df[df['Yaş_Kaynak'] == source]['Ekipman_Yaşı_Yıl'].mean()
        f.write(f"  {source}: {count:,} ({count/len(df)*100:.1f}%) - Avg age: {avg_age:.1f} yrs\n")
    
    f.write(f"\nEQUIPMENT AGE STATISTICS:\n")
    f.write(f"  Mean: {df['Ekipman_Yaşı_Yıl'].mean():.1f} years\n")
    f.write(f"  Median: {df['Ekipman_Yaşı_Yıl'].median():.1f} years\n")
    f.write(f"  Max: {df['Ekipman_Yaşı_Yıl'].max():.1f} years\n")
    f.write(f"  Age > 50 years: {(df['Ekipman_Yaşı_Yıl'] > 50).sum():,}\n")
    
    f.write("\nAGE DISTRIBUTION:\n")
    for min_age, max_age, label in age_ranges:
        count = ((df['Ekipman_Yaşı_Yıl'] >= min_age) & (df['Ekipman_Yaşı_Yıl'] < max_age)).sum()
        f.write(f"  {label}: {count:,} ({count/len(df)*100:.1f}%)\n")
    
    f.write("\nEQUIPMENT CLASS DISTRIBUTION:\n")
    f.write(df['Ekipman Sınıfı'].value_counts().head(15).to_string())

print(f"✓ Detailed summary saved to: {summary_file}")

print("\n" + "=" * 80)
print("📌 IMPORTANT NOTES:")
print("=" * 80)
print("1. ✅ Date Validation: Dates before 1950 are marked as invalid")
print("2. ✅ FIRST_WORKORDER_PROXY: Uses 'Oluşturulma_Tarihi'")
print("   ⚠️ This is a LOWER BOUND - equipment may be older!")
print(f"3. Equipment Age > 50 years: {(df['Ekipman_Yaşı_Yıl'] > 50).sum():,} cases")
print(f"   → Using proxy dates: {((df['Ekipman_Yaşı_Yıl'] > 50) & (df['Yaş_Kaynak'] == 'FIRST_WORKORDER_PROXY')).sum():,}")
print("   → Consider reviewing these manually")
print("\n✅ Ready for STEP 2: Feature Engineering\n")