"""
Full dataset analysis for Fine-grained Painted Pottery Recognition.
Outputs to file to avoid shell encoding issues.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
import os

SRC = Path(r'f:\考古\cc_pottery\pottery_dataset_index.json')
OUT = Path(r'f:\考古\cc_pottery\dataset_analysis_report.txt')

with open(SRC, 'r', encoding='utf-8') as f:
    data = json.load(f)

lines = []

def p(s=""):
    lines.append(s)

p(f"{'='*80}")
p(f"DATASET OVERVIEW")
p(f"{'='*80}")
p(f"Total artifacts: {len(data)}")

# ---- Image statistics ----
img_counts = [len(item['images']) for item in data]
total_imgs = sum(img_counts)
p(f"\nTotal image references: {total_imgs}")
p(f"  Items with 1 image:  {img_counts.count(1)} ({100*img_counts.count(1)/len(data):.1f}%)")
p(f"  Items with 2 images: {img_counts.count(2)} ({100*img_counts.count(2)/len(data):.1f}%)")
p(f"  Items with 3+ images:{sum(1 for x in img_counts if x>=3)} ({100*sum(1 for x in img_counts if x>=3)/len(data):.1f}%)")

# Verify image existence
existing_imgs = 0
missing_imgs = 0
sample_missing = []
for item in data:
    for img_path in item['images']:
        if os.path.exists(img_path):
            existing_imgs += 1
        else:
            missing_imgs += 1
            if len(sample_missing) < 5:
                sample_missing.append(img_path)

p(f"\nImage file verification:")
p(f"  Existing: {existing_imgs}")
p(f"  Missing:  {missing_imgs}")
if sample_missing:
    p(f"  Sample missing paths:")
    for path in sample_missing:
        p(f"    {path}")

# ---- Volume distribution ----
vols = Counter(item['vol'] for item in data)
p(f"\n{'='*80}")
p(f"VOLUME DISTRIBUTION ({len(vols)} volumes)")
p(f"{'='*80}")
for v, c in vols.most_common():
    p(f"  [{c:4d}] {v}")

# ---- Culture analysis ----
cultures = Counter(item['culture'] for item in data)
p(f"\n{'='*80}")
p(f"CULTURE ANALYSIS")
p(f"{'='*80}")
p(f"Unique cultures: {len(cultures)}")

p(f"\n  Top 30 cultures:")
for c, n in cultures.most_common(30):
    nan_tag = " [NO LABEL]" if c == "nan" else ""
    p(f"    {c}: {n}{nan_tag}")

sorted_cultures = cultures.most_common()
p(f"\n  Long-tail analysis:")
heads = sum(n for _, n in sorted_cultures if n >= 100)
mids = sum(n for _, n in sorted_cultures if 10 <= n < 100)
tails = sum(n for _, n in sorted_cultures if n < 10)
p(f"    Head (>=100 samples): {sum(1 for _,n in sorted_cultures if n>=100)} cultures, {heads} samples ({100*heads/len(data):.1f}%)")
p(f"    Mid  (10-99 samples): {sum(1 for _,n in sorted_cultures if 10<=n<100)} cultures, {mids} samples ({100*mids/len(data):.1f}%)")
p(f"    Tail (<10 samples):   {sum(1 for _,n in sorted_cultures if n<10)} cultures, {tails} samples ({100*tails/len(data):.1f}%)")

# ---- Artifact type analysis ----
names = Counter(item['name'] for item in data)
p(f"\n{'='*80}")
p(f"ARTIFACT TYPE ANALYSIS")
p(f"{'='*80}")
p(f"Unique types: {len(names)}")
for n, c in names.most_common():
    p(f"  {n}: {c}")

# ---- Era analysis ----
eras = Counter(item['era'] for item in data)
p(f"\n{'='*80}")
p(f"ERA ANALYSIS")
p(f"{'='*80}")
for e, c in eras.most_common():
    p(f"  {e}: {c}")

# ---- Cross-analysis: culture x artifact type ----
culture_type = Counter((item['culture'], item['name']) for item in data)
p(f"\n{'='*80}")
p(f"CROSS-ANALYSIS: Top Culture x Type combinations")
p(f"{'='*80}")
for (c, t), n in culture_type.most_common(30):
    p(f"  {c} + {t}: {n}")

# ---- Description text analysis ----
desc_lengths = [len(item['description']) for item in data]
p(f"\n{'='*80}")
p(f"DESCRIPTION TEXT ANALYSIS")
p(f"{'='*80}")
p(f"  Mean length: {sum(desc_lengths)/len(desc_lengths):.0f} chars")
p(f"  Median length: {sorted(desc_lengths)[len(desc_lengths)//2]:.0f} chars")
p(f"  Min/Max: {min(desc_lengths)}/{max(desc_lengths)}")
p(f"  Samples >= 100 chars: {sum(1 for d in desc_lengths if d>=100)}")
p(f"  Samples >= 200 chars: {sum(1 for d in desc_lengths if d>=200)}")

# Sample diverse descriptions
p(f"\n  Sample descriptions (diverse cultures & types):")
seen_cultures = set()
for item in data:
    c = item['culture']
    if c not in seen_cultures and len(seen_cultures) < 5:
        seen_cultures.add(c)
        p(f"\n  [{c}] {item['name']} ({item['era']})")
        p(f"  Dimensions: {item['dimensions']}")
        p(f"  Description: {item['description'][:200]}...")

# ---- Potential fine-grained tasks ----
p(f"\n{'='*80}")
p(f"FINE-GRAINED TASK ANALYSIS")
p(f"{'='*80}")

p(f"\n  Task 1: Culture Classification")
p(f"    Classes: {len(cultures)} (with 'nan' = {cultures.get('nan', 0)} unlabeled)")
p(f"    Challenge: Long-tail distribution, open-set cultures")

p(f"\n  Task 2: Artifact Type Classification")
p(f"    Classes: {len(names)}")
p(f"    Challenge: Fine-grained visual differences")

p(f"\n  Task 3: Cross-Culture Artifact Recognition")
multi_culture_types = sum(1 for t in names
    if sum(1 for c,_ in culture_type if _==t) > 2)
p(f"    Types appearing in >2 cultures: {multi_culture_types}")
p(f"    Challenge: Style variation across regions within same artifact type")

p(f"\n  Task 4: Era Prediction")
p(f"    Classes: {len(eras)}")

p(f"\n  Task 5: Text-to-Image Retrieval (Archaeological)")
p(f"    Query: textual description -> matching artifact image")

# ---- Data quality issues ----
p(f"\n{'='*80}")
p(f"DATA QUALITY ISSUES")
p(f"{'='*80}")
empty_desc = sum(1 for item in data if len(item['description'].strip()) < 10)
p(f"  Very short descriptions (<10 chars): {empty_desc}")
nan_cultures = sum(1 for item in data if item['culture'] == 'nan')
p(f"  Unlabeled cultures (nan): {nan_cultures}")
uids = [item['uid'] for item in data]
dup_uids = len(uids) - len(set(uids))
p(f"  Duplicate UIDs: {dup_uids}")

p(f"\n{'='*80}")
p(f"ANALYSIS COMPLETE")
p(f"{'='*80}")

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("Report written to:", str(OUT))
