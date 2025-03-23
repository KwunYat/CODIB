import cv2
import numpy as np
import os
import pywt
from PIL import Image

def WaveTrans(image, original_size):
    if not isinstance(image, np.ndarray):
        image_array = np.array(image)
    else:
        image_array = image
    

    hf_components_channels = []
    
    for channel in range(image_array.shape[2]):  
        channel_data = image_array[:, :, channel]
        LL, (LH, HL, HH) = pywt.dwt2(channel_data, wavelet='haar')        
        combined_hf = (LH + HL + HH) / 3  
        combined_hf_normalized = (combined_hf - combined_hf.min()) / (combined_hf.max() - combined_hf.min()) * 255
        hf_components_channels.append(combined_hf_normalized)

    hf_image_array = np.stack(hf_components_channels, axis=2)

    hf_resized = cv2.resize(hf_image_array, (original_size[0], original_size[1]), interpolation=cv2.INTER_LINEAR)
    
    HF_image = Image.fromarray(np.uint8(hf_resized))
    
    return HF_image

def process_labels(path):
    dir = path
    savepath = "data/TrainDataset/Wavelet_gt"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    pathlists = os.listdir(dir)
    for path in pathlists:
        name = path.split('.')[0]
        img = Image.open(os.path.join(dir, path)).convert('RGB') 
        original_size = img.size  
        HF = WaveTrans(img, original_size)

        HF.save(os.path.join(savepath, name + '.png'), format='PNG')

path_train = "./data/TrainDataset/Mask"
process_labels(path_train)
