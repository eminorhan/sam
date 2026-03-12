import os
import subprocess
import tempfile
from pathlib import Path
from multiprocessing import Pool

# --- Configuration ---
RAW_DIR = "/lustre/polis/stf218/scratch/emin/sa1b/raw"
SORTED_DIR = "/lustre/polis/stf218/scratch/emin/sa1b/sorted"
# NEW: Define a massive temp directory right on Lustre
TEMP_LUSTRE_DIR = "/lustre/polis/stf218/scratch/emin/sa1b/temp_extraction"

# Set this to the number of CPU cores on your node (e.g., 32, 64)
# Warning: 64 workers extracting millions of files simultaneously might hammer 
# the Lustre Metadata Server. If the filesystem gets sluggish, dial this down to 16 or 32.
NUM_WORKERS = 64 

def repack_tar(tar_path):
    tar_path = Path(tar_path)
    out_tar_path = Path(SORTED_DIR) / tar_path.name
    
    # Skip if already processed (great for resuming if the script gets interrupted)
    if out_tar_path.exists():
        return f"Skipped {tar_path.name} (already exists)"

    # Extract to the Lustre scratch space to completely bypass the /tmp limit
    with tempfile.TemporaryDirectory(dir=TEMP_LUSTRE_DIR) as tmpdir:
        try:
            # 1. Extract the scrambled tar to Lustre
            subprocess.run(["tar", "-xf", str(tar_path), "-C", tmpdir], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            
            # 2. Re-pack it directly back to Lustre using the crucial --sort=name flag
            subprocess.run(["tar", "--sort=name", "-cf", str(out_tar_path), "-C", tmpdir, "."], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            
            return f"Success: Repacked {tar_path.name}"
            
        except subprocess.CalledProcessError as e:
            # Capture and return the ACTUAL error message from tar
            return f"❌ Error processing {tar_path.name}:\n{e.stderr}"

if __name__ == "__main__":
    os.makedirs(SORTED_DIR, exist_ok=True)
    os.makedirs(TEMP_LUSTRE_DIR, exist_ok=True) # Ensure our new Lustre temp dir exists
    
    tar_files = list(Path(RAW_DIR).glob("*.tar"))
    print(f"Found {len(tar_files)} tar files. Starting highly-parallel repack on Lustre...")
    
    # Fire up the worker pool
    with Pool(NUM_WORKERS) as p:
        for result in p.imap_unordered(repack_tar, tar_files):
            print(result)
            
    print("\nAll done! Update your training script to point to the sorted/ directory.")