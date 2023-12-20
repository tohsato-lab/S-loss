import os
import numpy as np


from PIL import Image
import matplotlib.pyplot as plt

def make_pallet(config):
    area_id_dict = config.mouse_area_data
    color_list = []
    for area_name in area_id_dict.keys():
        color_list.append(area_id_dict[area_name]['color'])
    pallet = []
    for color in color_list:
        for i in color:
            pallet.append(i)
    return pallet


def create_class_image(config, original_image, label_image):
    pallet = make_pallet(config)
    if original_image.shape[0] == 3:
        original_image = np.transpose(original_image, (1, 2, 0))
    if np.max(original_image) < 2:
        original_image = np.array(original_image * 255, dtype="uint8")
    else:
        original_image = np.array(original_image, dtype="uint8")
    original_image = Image.fromarray(original_image)
    original_image = original_image.convert("RGB")
    label_image = Image.fromarray(label_image.astype(np.uint8))
    label_image = label_image.convert("P")
    label_image.putpalette(pallet)
    label_image = label_image.convert("RGB")
    original_and_class = Image.blend(original_image, label_image, 0.5)
    return original_and_class




def save_overlap_image(config, image, label, name):
    batch_size = image.shape[0]
    for batch_n in range(batch_size):
        img = image[batch_n]
        lbl = label[batch_n]
        nam = name[batch_n]
        overlap_image = create_class_image(config, img, lbl)
        save_path = os.path.join(config.Log_ViewFolder, nam)
        overlap_image.save(save_path)


def create_label_image(config, label, name, save_folder):
    pallet = make_pallet(config)
    batch_num = label.shape[0]
    for batcn_n in range(batch_num):
        label_image = label[batcn_n]
        label_image = Image.fromarray(label_image.astype(np.uint8))
        label_image = label_image.convert("P")
        label_image.putpalette(pallet)
        label_image = label_image.convert("RGB")
        label_image.save(os.path.join(save_folder, name[batcn_n]))

def create_pred_one_image(config, label):
    pallet = make_pallet(config)
    label_image = label.convert("P")
    label_image.putpalette(pallet)
    label_image = label_image.convert("RGB")
    return label_image

def create_hist_image(config, label, name, save_folder, percent):
    label_image = np.array(label, dtype=int)
    batch_num = label.shape[0]
    for batcn_n in range(batch_num):
        plt.figure()
        for key in config.mouse_area_data.keys():
            c = config.mouse_area_data[key]['id']
            color = config.mouse_area_data[key]['color']
            color = np.array(color)/255
            plt.hist(label_image[batcn_n][c].flatten(), bins=20, color=color, alpha=0.5)
        plt.savefig(os.path.join(save_folder, name[batcn_n].replace('.png', f'_hit_{percent[batcn_n][0,0,0]}.png')))
