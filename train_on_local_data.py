import os
import shutil
import cv2
import hashlib
from pathlib import Path
from tqdm import tqdm
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ==========================================
# 1. LOCAL PATHS & DIRECTORY SETUP
# ==========================================
# Update these paths if your Desktop username is different!
STAGING_DIR = Path(r"C:\Users\SAKSHAM PASWAN\Desktop\Waste_Project\2_staging_area")
MASTER_DIR = Path(r"C:\Users\SAKSHAM PASWAN\Desktop\Waste_Project\3_master_dataset")

# The exact 20 Leaf Nodes from your hierarchical taxonomy
TARGET_CLASSES = [
    # Hazardous
    'hw_ewaste_boards', 'hw_ewaste_cables', 'hw_ewaste_devices',
    'hw_chem_aerosol', 'hw_chem_bottles', 'hw_med_syringes', 'hw_med_blisters',
    # Compostable
    'nh_comp_org_food', 'nh_comp_org_coffee', 'nh_comp_dry_yard', 
    'nh_comp_dry_paper', 'nh_comp_inorg_shells', 'nh_comp_inorg_bioplastic',
    # Recyclable
    'nh_noncomp_rec_plastic_rigid', 'nh_noncomp_rec_glass', 
    'nh_noncomp_rec_metal', 'nh_noncomp_rec_cardboard',
    # Landfill
    'nh_noncomp_nonrec_plastic_soft', 'nh_noncomp_nonrec_styrofoam', 'nh_noncomp_nonrec_soiled'
]

# Generate the physical folder structure
for category in TARGET_CLASSES:
    os.makedirs(MASTER_DIR / category, exist_ok=True)

# ==========================================
# 2. THE UNIVERSAL TRANSLATION DICTIONARY
# ==========================================
CLASS_MAPPING = {
    # Hazardous
    'e-waste': 'hw_ewaste_boards', 'e_waste': 'hw_ewaste_boards', 'pcb': 'hw_ewaste_boards',
    'cables': 'hw_ewaste_cables', 'wires': 'hw_ewaste_cables',
    'small_electronics': 'hw_ewaste_devices', 'phones': 'hw_ewaste_devices',
    'aerosol': 'hw_chem_aerosol', 'spray_cans': 'hw_chem_aerosol',
    'chemicals': 'hw_chem_bottles',
    'medical_waste': 'hw_med_syringes', 'medical': 'hw_med_syringes', 'syringes': 'hw_med_syringes',
    'blister_packs': 'hw_med_blisters', 'pills': 'hw_med_blisters',

    # Compostable
    'organic': 'nh_comp_org_food', 'organic waste': 'nh_comp_org_food', 'food_waste': 'nh_comp_org_food',
    'coffee': 'nh_comp_org_coffee', 'tea_bags': 'nh_comp_org_coffee',
    'yard_waste': 'nh_comp_dry_yard', 'leaves': 'nh_comp_dry_yard',
    'paper': 'nh_comp_dry_paper', 'newspaper': 'nh_comp_dry_paper',
    'eggshells': 'nh_comp_inorg_shells',
    'bioplastic': 'nh_comp_inorg_bioplastic', 'pla': 'nh_comp_inorg_bioplastic',

    # Recyclable
    'plastic': 'nh_noncomp_rec_plastic_rigid', 'bottles': 'nh_noncomp_rec_plastic_rigid', 'pet': 'nh_noncomp_rec_plastic_rigid',
    'glass': 'nh_noncomp_rec_glass', 'brown-glass': 'nh_noncomp_rec_glass', 'green-glass': 'nh_noncomp_rec_glass', 'white-glass': 'nh_noncomp_rec_glass',
    'metal': 'nh_noncomp_rec_metal', 'cans': 'nh_noncomp_rec_metal',
    'cardboard': 'nh_noncomp_rec_cardboard', 'boxes': 'nh_noncomp_rec_cardboard', 'corrugated': 'nh_noncomp_rec_cardboard',

    # Landfill
    'plastic_bags': 'nh_noncomp_nonrec_plastic_soft', 'wrappers': 'nh_noncomp_nonrec_plastic_soft',
    'styrofoam': 'nh_noncomp_nonrec_styrofoam', 'foam': 'nh_noncomp_nonrec_styrofoam',
    'trash': 'nh_noncomp_nonrec_soiled', 'garbage': 'nh_noncomp_nonrec_soiled', 'soiled_paper': 'nh_noncomp_nonrec_soiled',
    'clothes': 'nh_noncomp_nonrec_soiled', 'shoes': 'nh_noncomp_nonrec_soiled', 'textile': 'nh_noncomp_nonrec_soiled'
}

# ==========================================
# 3. THE CLEANING ENGINE
# ==========================================
logging.info("🚀 Starting Local ETL Pipeline...")
seen_hashes = set()
stats = {"copied": 0, "corrupted": 0, "duplicates": 0, "unmapped": 0}

for root, dirs, files in os.walk(STAGING_DIR):
    current_folder = Path(root).name.lower().strip().replace(' ', '_').replace('-', '_')
    
    if current_folder in CLASS_MAPPING:
        target_folder = CLASS_MAPPING[current_folder]
        target_path = MASTER_DIR / target_folder
        
        for file in tqdm(files, desc=f"Processing -> {target_folder}"):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')): 
                continue
            
            source_file = Path(root) / file
            
            try:
                # 1. Deduplicate
                with open(source_file, "rb") as f:
                    img_hash = hashlib.sha256(f.read()).hexdigest()
                    
                if img_hash in seen_hashes: 
                    stats["duplicates"] += 1
                    continue
                
                # 2. Integrity Check
                img = cv2.imread(str(source_file))
                if img is None: 
                    stats["corrupted"] += 1
                    continue
                    
                # 3. Copy & Rename
                seen_hashes.add(img_hash)
                new_filename = f"{img_hash[:12]}{source_file.suffix}"
                shutil.copy2(source_file, target_path / new_filename)
                stats["copied"] += 1
                
            except Exception:
                stats["corrupted"] += 1
                continue
    else:
        stats["unmapped"] += len(files)

# ==========================================
# 4. REPORT of the Dataset
# ==========================================
logging.info("\n" + "="*50)
logging.info("✅ LOCAL MASTER DATASET BUILT!")
logging.info("="*50)
logging.info(f"🌟 Pristine Images:      {stats['copied']:,}")
logging.info(f"🚨 Duplicates Blocked:  {stats['duplicates']:,}")
logging.info(f"🗑️ Corrupted Blocked:   {stats['corrupted']:,}")
logging.info("="*50)
