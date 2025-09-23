# 📊 Data Quality Checker Guide

## 🎯 Overview

The Data Quality Checker analyzes your DIODE dataset to identify and flag problematic samples that could cause training issues. It addresses all the problems we encountered during training setup.

## 🔍 **Issues Detected:**

### **Depth Issues:**
- ✅ **Negative depth values** - Values < 0 that cause NaN in loss
- ✅ **Zero depth values** - Invalid depth measurements  
- ✅ **Extreme depth values** - Unreasonably large (>100m) or small (<0.001m)
- ✅ **Data type issues** - Incorrect numpy dtypes

### **Mask Issues:**
- ✅ **Empty masks** - No valid pixels (causes empty tensor errors)
- ✅ **Sparse masks** - Very few valid pixels (<1% by default)
- ✅ **Non-binary masks** - Values other than 0/1
- ✅ **Data type issues** - Incorrect mask dtypes

### **File Issues:**
- ✅ **Missing files** - Incomplete triplets (image/depth/mask)
- ✅ **Corrupted files** - Files that can't be loaded
- ✅ **Dimension mismatches** - Image/depth/mask size inconsistencies
- ✅ **File read errors** - Permission or format issues

## 🚀 **Usage Commands:**

### **Quick Check:**
```bash
# Check data quality using existing splits
./dataset_manager.sh check-data-quality
```

### **Advanced Usage:**
```bash
# Run quality checker directly with custom parameters
docker compose run --rm depth-anything python /app/data_quality_checker.py \
  --data_root=/data \
  --splits_dir=/app/dataset_splits \
  --output_dir=/app/data_quality_reports \
  --max_depth=50.0 \
  --min_depth=0.01 \
  --min_mask_ratio=0.05
```

## 📊 **Generated Reports:**

### **1. Summary Report** (`data_quality_summary_YYYYMMDD_HHMMSS.json`)
```json
{
  "statistics": {
    "total_samples": 69588,
    "valid_samples": 65432,
    "problematic_samples": 4156
  },
  "issue_counts": {
    "negative_depths": 1523,
    "empty_masks": 234,
    "extreme_depths": 445,
    "sparse_masks": 1954
  }
}
```

### **2. Detailed Report** (`data_quality_detailed_YYYYMMDD_HHMMSS.json`)
Complete breakdown of each issue with specific file paths and statistics.

### **3. Problematic Files List** (`problematic_files_YYYYMMDD_HHMMSS.txt`)
```
# Problematic files found on 20250923_143052
# Total problematic files: 4156
/data/deployment_1001/scan_000001/1001_L_000000.png
/data/deployment_1001/scan_000002/1001_L_000001.png
...
```

### **4. Clean Files List** (`clean_files_YYYYMMDD_HHMMSS.txt`)
```
# Clean files found on 20250923_143052  
# Total clean files: 65432
/data/deployment_1001/scan_000010/1001_L_000010.png
/data/deployment_1002/scan_000001/1002_L_000000.png
...
```

## 🔧 **Configurable Thresholds:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | 100.0m | Maximum reasonable depth |
| `min_depth` | 0.001m | Minimum reasonable depth (1mm) |
| `min_mask_ratio` | 0.01 | Minimum 1% valid pixels required |
| `extreme_depth_ratio` | 0.05 | Flag if >5% pixels are extreme |

## 📈 **Example Quality Check Results:**

```bash
$ ./dataset_manager.sh check-data-quality

==== Checking Data Quality ====
[INFO] Analyzing DIODE dataset for quality issues...
[INFO] This may take several minutes for large datasets.

🔍 Starting data quality check for: /data
📊 Output directory: /app/data_quality_reports
📁 Found 69588 samples to check
Checking data quality: 100%|██████████| 69588/69588 [15:23<00:00, 75.32it/s]

📝 Generating quality reports...

✅ Data quality check completed!
📊 Total samples checked: 69588
✅ Valid samples: 65432  
⚠️  Problematic samples: 4156

📄 Reports generated:
  📋 Summary: /app/data_quality_reports/data_quality_summary_20250923_143052.json
  📝 Detailed: /app/data_quality_reports/data_quality_detailed_20250923_143052.json
  🚫 Problematic files: /app/data_quality_reports/problematic_files_20250923_143052.txt
  ✅ Clean files: /app/data_quality_reports/clean_files_20250923_143052.txt

🔍 Issue breakdown:
  Negative Depths: 1523
  Empty Masks: 234
  Extreme Depths: 445
  Sparse Masks: 1954

[INFO] ✅ Data quality check completed!
[INFO] Reports saved to: data_quality_reports/
```

## 🛠️ **Using Quality Results:**

### **1. Filter Training Data:**
```bash
# Create clean dataset splits using only good samples
python create_clean_splits.py \
  --clean_files=data_quality_reports/clean_files_20250923_143052.txt \
  --output_dir=clean_dataset_splits
```

### **2. Monitor Training Issues:**
- Review problematic files list before training
- Use clean files list for critical training runs
- Check depth statistics for model configuration

### **3. Data Preprocessing:**
```python
# Example: Load only clean samples
with open('data_quality_reports/clean_files_20250923_143052.txt', 'r') as f:
    clean_files = [line.strip() for line in f if not line.startswith('#')]

# Filter your dataset
clean_dataset = [sample for sample in dataset if sample['image'] in clean_files]
```

## 🚨 **Common Issues Found:**

### **1. Negative Depths (Fixed in our training)**
- **Problem**: Depth values < 0 cause NaN in logarithmic loss
- **Our Fix**: `depth = np.maximum(depth, 0.001)` in data loader
- **Quality Check**: Identifies which files have this issue

### **2. Empty Masks (Fixed in our training)**  
- **Problem**: All pixels masked out → empty tensors → training crash
- **Our Fix**: Return small loss value when mask is empty
- **Quality Check**: Identifies samples with no valid pixels

### **3. Extreme Depth Values**
- **Problem**: Unrealistic depths (>100m underwater, <1mm)
- **Solution**: Filter these samples or clamp values
- **Quality Check**: Flags based on configurable thresholds

### **4. Sparse Masks**
- **Problem**: Very few valid pixels provide poor training signal
- **Solution**: Consider removing or giving lower weight
- **Quality Check**: Identifies samples with <1% valid pixels

## 🎯 **Quality-Based Training Strategy:**

### **Conservative Approach:**
```bash
# Train only on high-quality samples
./dataset_manager.sh train 120 2 production_clean_$(date +%Y%m%d)
# (modify splits to use only clean files)
```

### **Robust Approach:**
```bash  
# Train on all data with our fixes handling problematic samples
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./dataset_manager.sh train 120 2 production_robust_$(date +%Y%m%d)
```

## 📋 **Quality Checklist:**

Before training:
- [ ] Run `./dataset_manager.sh check-data-quality`
- [ ] Review issue breakdown 
- [ ] Check if >90% samples are valid
- [ ] Identify major issue categories
- [ ] Decide on filtering vs. robust handling strategy
- [ ] Monitor training for correlation with data quality issues

Your data quality checker will help prevent the training issues we encountered and ensure robust, reliable training! 🚀
