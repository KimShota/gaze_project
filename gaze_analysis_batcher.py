import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import glob 
import os
import argparse
import cv2 
from typing import Dict, Tuple, Optional

GRID_N = 16
CELLS_PER_SIDE = 256

def id_to_grid_position(cellID: int) -> Tuple[int, int]:
    sid = conversion(cellID)
    actual_sid = sid - 1
    row = actual_sid // GRID_N
    col = actual_sid % GRID_N
    return int(row), int(col)

def build_grid(percent_map):
    grids = np.zeros((GRID_N, GRID_N), dtype=np.float32)

    for cellID, pct in percent_map.items(): 
        row, col = id_to_grid_position(cellID)
        grids[row, col] = float(pct)
    return grids

def find_image(image_folder: str, pair: str, side: str) -> Optional[str]:
    """Return the image path for this pair/side if found, else None."""
    pair = str(pair).upper().strip()
    side = side.capitalize().strip()  # "Left"/"Right"
    num  = '1' if side == 'Left' else '2'

    # Try a few patterns; add more if needed
    patterns = [
        os.path.join(image_folder, f"{pair}{num}.*"),        # A1.jpg/png/jpeg
        os.path.join(image_folder, f"{pair}_{side}.*"),      # A_Left.*
        os.path.join(image_folder, f"Pair_{pair}_{side}.*"), # Pair_A_Left.*
        os.path.join(image_folder, f"{pair}-{side}.*"),      # A-Left.*
    ]
    exts = (".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG")

    for pat in patterns:
        for cand in glob.glob(pat):
            if cand.endswith(exts) and os.path.exists(cand):
                return cand
    return None

def overlay_heatmap_on_image(
    base_img_path: str,
    grid: np.ndarray,
    out_path: str,
    *,
    percentile: float = 95.0,
    blur_kernel: int = 13,
    alpha_heat: float = 0.6,
    alpha_img: float  = 0.4
) -> None:
    """
    Convert a 16x16 grid (0..100) to a smooth colored overlay and blend it with the base image.
    Steps:
    1) clip by percentile to stabilize contrast
    2) normalize 0..vmax -> 0..255
    3) resize to image size (bicubic), gaussian blur
    4) colorize with JET, alpha-blend onto image
    """
    img = cv2.imread(base_img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"I could not load image: {base_img_path}")

    H, W = img.shape[:2] #extract the original height and width of the image 

    #Only extract non-zero values 
    nz = grid[grid > 0]
    if nz.size > 0:
        vmax = np.percentile(nz, percentile)
        if vmax <= 0:
            vmax = nz.max() if nz.max() > 0 else 1.0
    else:
        vmax = 1.0

    norm = np.clip(grid / vmax, 0.0, 1.0) * 255.0
    norm = norm.astype(np.uint8)

    # Upscale to image resolution
    big = cv2.resize(norm, (W, H), interpolation=cv2.INTER_CUBIC)

    # Gentle blur for smoothness (kernel size must be odd)
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    big = cv2.GaussianBlur(big, (blur_kernel, blur_kernel), 0)

    # Colorize
    heat = cv2.applyColorMap(big, cv2.COLORMAP_JET)

    # Blend
    overlay = cv2.addWeighted(heat, alpha_heat, img, alpha_img, 0)

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ok = cv2.imwrite(out_path, overlay)
    if not ok:
        raise RuntimeError(f"Failed to write {out_path}")

def save_heatmap_only_png(grid: np.ndarray, out_path: str):
    """
    Optional: save a heatmap-only image (no base image) in the same resolution as the base,
    mainly for debugging or documentation. Uses OpenCV coloring (no seaborn).
    """
    nz = grid[grid > 0]
    vmax = np.percentile(nz, 95.0) if nz.size > 0 else 1.0
    vmax = max(vmax, 1e-6)

    norm = np.clip(grid / vmax, 0.0, 1.0) * 255.0
    norm = norm.astype(np.uint8)
    # Scale up to a nice size (e.g., 768×768)
    big = cv2.resize(norm, (768, 768), interpolation=cv2.INTER_NEAREST)
    heat = cv2.applyColorMap(big, cv2.COLORMAP_JET)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, heat)


#helper funciton 
def parse_cellID_column (pandas_series: pd.Series) -> pd.Series: 
    s = pandas_series.astype(str)
    out = s.str.extract(r'cellIDGrid_(\d+)', expand=False)
    return out.astype('Int64')

def parse_imagePair_column(pandas_series : pd.Series) -> pd.Series: 
    return pandas_series.astype(str).str.extract(r'imagePair([A-Z])', expand=False)

#score filteration 
def category(score): 
    if 0 <= score <= 46:
        return 'Low'
    elif 61 <= score <= 100: 
        return 'High'
     
#side filteration 
def Left(cellID: int) -> bool: 
    if pd.isna(cellID): 
        return False
    return (1 <= cellID <= 256) or (513 <= cellID <= 768)

def Right(cellID: int) -> bool:
    if pd.isna(cellID): 
        return False
    return (257 <= cellID <= 512) or (769 <= cellID <= 1024) 

#frequency logic
def frequency(seq): 
    counts = {}
    prevID = None
    totalFre = 0

    for cellID in seq: 
        if cellID != prevID: 
            counts[cellID] = counts.get(cellID, 0) + 1
            totalFre += 1
        
        prevID = cellID
    
    return counts, totalFre

#duration logic
def duration(seq): 
    counts = {}
    totalDur = 0
    prevID = None
    currentStreak = 0

    for cellID in seq: 
        if cellID == prevID: 
            currentStreak += 1
        else: 
            if prevID is not None: 
                counts[prevID] = counts.get(prevID, 0) + currentStreak
                totalDur += currentStreak
            currentStreak = 1
            prevID = cellID
    
    if prevID is not None and currentStreak > 0: 
        counts[prevID] = counts.get(prevID, 0) + currentStreak
        totalDur += currentStreak

    return counts, totalDur

#function to convert cellIDs to fit into the grid 
def conversion(cellID: int) -> int: 
    cellID = int(cellID)
    return ((cellID - 1) % 256) + 1

#function to analyze each file 
def gaze_analysis_batcher (filepath, filtering): 

    file = {}

    #read the csv file and sorting out the data 
    df = pd.read_csv(filepath)

    #extracting significant parts
    df['ID'] = parse_cellID_column(df['cellID'])
    df['pair'] = parse_imagePair_column(df['imagePair'])
    df = df.dropna(subset=['pair', 'ID', 'score'])

    #categorizing 
    df['Stress Group'] = df['score'].apply(category)

    #Filtering 
    if filtering.lower() == 'high': 
        df = df[df['Stress Group'] == 'High']
    elif filtering.lower() == 'low': 
        df = df[df['Stress Group'] == 'Low']

    for pair, pair_df in df.groupby('pair', sort=True): 
        #only extract cellIDs with each row, depending on whether it is left or right
        left_side = pair_df['ID'][pair_df['ID'].apply(Left)].tolist()
        right_side = pair_df['ID'][pair_df['ID'].apply(Right)].tolist()

        if left_side: 
            freL, totFreL = frequency(left_side)
            durL, totDurL = duration(left_side)
            file[pair, 'Left', 'Frequency'] = {k: v / totFreL for k, v in freL.items()} if totFreL > 0 else {}
            file[pair, 'Left', 'Duration'] = {k: v / totDurL for k, v in durL.items()} if totDurL > 0 else {}
        else: 
            file[pair, 'Left', 'Frequency'] = {}
            file[pair, 'Left', 'Duration'] = {}

        if right_side: 
            freR, totFreR = frequency(right_side)
            durR, totDurR = duration(right_side)
            file[pair, 'Right', 'Frequency'] = {k: v / totFreR for k, v in freR.items()} if totFreR > 0 else {}
            file[pair, 'Right', 'Duration'] = {k: v / totDurR for k, v in durR.items()}  if totDurR > 0 else {}
        else: 
            file[pair, 'Right', 'Frequency'] = {}
            file[pair, 'Right', 'Duration'] = {}

        
    return file

#function to create a heatmap
# def create_heatmap(img, percent_map, cmap, title, out_file):
#     grids = np.zeros((16, 16), dtype=float)

#     for cellID, avg_percent in percent_map.items(): 
#         converted_id = conversion(cellID)
#         row = (converted_id - 1) // 16
#         col = (converted_id - 1) % 16
#         grids[row, col] = avg_percent

#     plt.figure()
#     heatmap_img = sns.heatmap(grids, annot=True, cmap=cmap, fmt='.1f', cbar=True, vmin=0, vmax=15)
#     plt.title(title)
#     plt.xlabel('Column')
#     plt.ylabel('Row')
#     plt.savefig(out_file)
#     plt.close()

#function to access each image 
home = os.path.expanduser('~')
image_folder = os.path.join(home, 'Downloads', 'gaze_project', 'Images')
os.makedirs(image_folder, exist_ok=True)

#function to batch-analyze files 
def batcher(input_folder, filtering):
    allfiles = glob.glob(os.path.join(input_folder, 'VR*/**/*.csv'), recursive=True)

    totalPercentages = {}
    participantCounts = {}

    for file in allfiles: 
        try:
            per_file = gaze_analysis_batcher(file, filtering)
        except Exception as e: 
            print(f'There is an error with the file {file}: {e}')
            continue
        
        for key, cell_map in per_file.items(): 
            if key not in totalPercentages:
                totalPercentages[key] = {}
                participantCounts[key] = {}
            for cellID, percentage in cell_map.items(): 
                totalPercentages[key][cellID] = totalPercentages[key].get(cellID, 0.0) + percentage
                participantCounts[key][cellID] = participantCounts[key].get(cellID, 0) + 1

    #file / folder definition 
    home = os.path.expanduser('~')
    heatmap_folder = os.path.join(home, 'Downloads', 'gaze_project', 'Heatmaps')
    os.makedirs(heatmap_folder, exist_ok=True)

    for (pair, side, mode), sums in totalPercentages.items(): 
        counts = participantCounts[(pair, side, mode)]
        avg = {cellID: round(100 * sums[cellID] / counts[cellID]) for cellID in sums if counts[cellID] > 0} #dictionary where key is cellID and value is an averaged percentage

        # Build grid
        grid = build_grid(avg)

        # Output folders
        out_folder = os.path.join(heatmap_folder, f'Pair_{pair}')
        os.makedirs(out_folder, exist_ok=True)

        # Try to find base image for the given pair and side
        base_img_path = find_image(image_folder, pair, side)

        # Always write a heatmap-only PNG (optional, great for QA)
        debug_heat_path = os.path.join(out_folder, f'{side}_{mode}_HEATMAP.png')
        save_heatmap_only_png(grid, debug_heat_path)

        if base_img_path is None:
            print(f"[WARN] No base image for Pair {pair} {side}. Saved heatmap-only at {debug_heat_path}.")
            continue

        # Overlay on the base image with good quality defaults
        out_file = os.path.join(out_folder, f'{side}_{mode}_OVERLAY.png')
        overlay_heatmap_on_image(
            base_img_path,
            grid,
            out_file,
            percentile=95.0,     # tune: 90–98 is common
            blur_kernel=13,      # tune: 11–17 for smoothness
            alpha_heat=0.60,     # heatmap strength
            alpha_img=0.40       # base-image visibility
        )
        print(f"[OK] {pair} {side} {mode} -> {out_file}")

    print('Heatmap generation is done!')

#main function 
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Batch-analyze files based on group')
    parser.add_argument(
        '--group', 
        choices=['High', 'Low'], 
        required=True, 
        help='Choose Score Range: High or Low'
    )
    parser.add_argument(
        '--input', 
        default='Health-O', 
        help='Choose your folder with csv files'
    )

    args = parser.parse_args()

    batcher(args.input, filtering=args.group)
