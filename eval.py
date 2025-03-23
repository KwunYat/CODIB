import os
import cv2
from tqdm import tqdm
# Assume py_sod_metrics is already installed
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

methods = 'MINet'.split(', ')
for method in methods:  # Loop through each method
    for _data_name in ['CAMO', 'COD10K', 'NC4K']:
        mask_root = f'./data/TestDataset/{_data_name}/GT'
        pred_root = f'./results/{method}/{_data_name}/'  # Path modified to include method
        mask_name_list = sorted(os.listdir(mask_root))
        
        FM = Fmeasure()
        WFM = WeightedFmeasure()
        SM = Smeasure()
        EM = Emeasure()
        M = MAE()

        for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
            mask_path = os.path.join(mask_root, mask_name)
            pred_path = os.path.join(pred_root, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            FM.step(pred=pred, gt=mask)
            WFM.step(pred=pred, gt=mask)
            SM.step(pred=pred, gt=mask)
            EM.step(pred=pred, gt=mask)
            M.step(pred=pred, gt=mask)

        fm = FM.get_results()["fm"]
        wfm = WFM.get_results()["wfm"]
        sm = SM.get_results()["sm"]
        em = EM.get_results()["em"]
        mae = M.get_results()["mae"]

        results = {
            "MAE": round(mae, 3),
            "adpEm": round(em["curve"].mean(), 3),
            "Smeasure": round(sm, 3),
            "adpFm": round(fm["curve"].mean(), 3),
            "wFmeasure": round(wfm, 3),
        }

        print(f"{method} {results}")
        with open("evalresults.txt", "a") as file:
            file.write(f"{method} {_data_name} {results}\n")
