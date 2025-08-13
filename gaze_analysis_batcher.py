import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import glob 
import os
import argparse 
import cv2

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

#heatmap generator
def heatmap_generator(pair, side, mode, out_file, percent_map):

    image_name = os.path.join(image_folder, f"{pair}{1 if side == 'Left' else 2}.JPG")

    if not os.path.exists(image_name): 
        raise FileNotFoundError(f'{image_name} does not exist. Please try it again.')

    img = cv2.imread(image_name)
    if img is None: 
        raise ValueError(f'We failed to load the image')

    grids = np.zeros((16, 16), dtype=np.float32)
    for cellID, pct in percent_map.items(): 
        converted_id = conversion(cellID)
        row = (converted_id - 1) // 16
        col = (converted_id - 1) % 16
        grids[row, col] = float(pct)

    grid_255 = np.clip((grids / 100.0) * 255.0, 0, 255).astype(np.uint8)

    h, w = img.shape[:2] #take only the first two values 
    mask = cv2.resize(grid_255, (w, h), interpolation=cv2.INTER_NEAREST)

    mask = cv2.GaussianBlur(mask, (13, 13), 11)
    heatmap_img = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
    cv2.imwrite(out_file, super_imposed_img)
    
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
        cmap = 'Blues' if mode == 'Frequency' else 'Reds'
        title = f'{filtering} Pair {pair} {side} {mode.capitalize()}'
        out_folder = os.path.join(heatmap_folder, f'Pair {pair}') 
        os.makedirs(out_folder, exist_ok=True)
        out_file = os.path.join(out_folder, f'{side} {mode}.png')
        heatmap_generator(pair, side, mode, out_file, percent_map=avg)
        
        #create_heatmap(avg, cmap, title, out_file)

    print('Heatmap gerenation is done!')

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
