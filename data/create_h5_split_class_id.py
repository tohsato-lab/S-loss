import os
import cv2
import h5py
import numpy as np
from PIL import Image
import copy
import pandas as pd

def sort_image_name(image_folser):
    png_name_list = os.listdir(image_folser)
    data_dict = dict()
    for image_name in png_name_list:
        data_tag_list = image_name.split("_")
        Gen, TM, Z, _, _ = data_tag_list
        GL = data_tag_list[3][0]

        if f"{Gen}_{TM}_{GL}" not in data_dict.keys():
            data_dict[f"{Gen}_{TM}_{GL}"] = [image_name]
        else:
            data_dict[f"{Gen}_{TM}_{GL}"].append(image_name)
    for key in data_dict.keys():
        data_dict[key] = sorted(data_dict[key])
    return data_dict


class Png2bd5:
    def __init__(self, image_folder, save_folder):
        self.image_folder = image_folder
        self.save_folder = save_folder
        self.name_dict = sort_image_name(self.image_folder)
        os.makedirs(self.save_folder, exist_ok=True)

    def set_units(self, gen_id):
        print(gen_id)
        self.ID = 0
        su_type = np.dtype({'names': ['dimension', 'xScale', 'yScale', 'zScale', 'sUnit'],
                            'formats': ['S8', 'f8', 'f8', 'f8', 'S16']})
        scaleunit = np.array([('3D', 1, 1, 20, 'micrometer')], dtype=su_type)
        bd5scaleunit = self.bd5f.create_dataset('/data/scaleUnit', data=scaleunit)

        # setting up datatype for scaleUnit
        objdf_type = np.dtype({'names': ['oID', 'name'],
                               'formats': ['i4', 'S128']})
        objectdef = np.array([[(1, 'Isocortex')],
                              [(2, 'Olfactory area')],
                              [(3, 'Hippocampal formation')],
                              [(4, 'Cerebral nuclei')],
                              [(5, 'Interbrain')],
                              [(6, 'Midbrain')],
                              [(7, 'Hindbrain')],
                              [(8, 'Cerebellum')]], dtype=objdf_type)
        bd5objdef = self.bd5f.create_dataset('/data/objectDef', data=objectdef)

        # Setting up a numpy array with the correct data type for storing line data
        self.obj_type = np.dtype({'names': ['ID', 't', 'entity', 'sID', 'x', 'y', 'z', 'label'],
                                  'formats': ['S10', 'f8', 'S6', 'int', 'f8', 'f8', 'f8', 'S20']})
        object0 = self.bd5f.create_group('/data/0/object')
        object_data = self.run(gen_id)
        for class_id in range(9):
            if class_id == 0:
                continue
            else:
                object_data_clas_id = self.split_table_by_classid(class_id, object_data)
                object_id = self.bd5f.create_dataset(f'/data/0/object/{class_id}', data=object_data_clas_id)

    def split_table_by_classid(self, class_id, object_data):
        class_id = str(class_id)
        data_table = []
        for data in object_data:
            data_table.append(list(data[0]))
        data_table = np.array(data_table)
        object_data_clas_id = object_data[np.where(data_table[:, -1] == class_id.encode('ascii'))]
        return object_data_clas_id

    def mask_to_polygons(self, contours, hierarchy):
        cnt_children = dict()
        for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
            if parent_idx != -1:
                if parent_idx not in cnt_children:
                    cnt_children.setdefault(parent_idx, [])
                cnt_children[parent_idx].extend(contours[idx])
            else:
                if idx not in cnt_children:
                    cnt_children.setdefault(idx, [])
                cnt_children[idx].extend(contours[idx])

        return cnt_children

    def png2bd5(self, image_path):
        image_name = image_path.split("/")[-1]
        data_tag_list = image_path.split("/")[-1].split("_")
        Gen, TM, Z, _, _ = data_tag_list

        image: np.ndarray = np.asarray(Image.open(image_path))

        print(image_name)
        x_correction, y_correction = 0, 0
        if os.path.isfile(os.path.join("contours", f"{image_name}.csv")):
            image_y, image_x = image.shape
            data_table = pd.read_csv(os.path.join("contours", f"{image_name}.csv"))
            x_correction = (int(data_table.iloc[0]['Padding_x']) - image_x) // 2 + int(data_table.iloc[0]['Center_x'])
            y_correction = (int(data_table.iloc[0]['Padding_y']) - image_x) // 2 + int(data_table.iloc[0]['Center_y'])
        else:
            with open("error.txt", "a") as f:
                f.write(f"{image_name}\n")
        object_data_list = []

        for class_id in np.unique(image):
            # 背景は除外
            if class_id == 0:
                continue
            mask_image = np.where(image == class_id, 255, 0).astype(np.uint8)
            # ここには輪郭を計算してるよ。輪郭データを持っていたら、直接変換すればいいよ
            find_result = cv2.findContours(mask_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            polygons = self.mask_to_polygons(find_result[-2], find_result[-1])
            for obj_id, i in enumerate(polygons.keys()):
                first_data = None
                for polygon_id, polygon in enumerate(polygons[i]):
                    if polygon_id == 0:
                        first_data = np.array(
                            #  ['ID','t','entity', 'sID','x', 'y', 'z', 'label'], 'label'の調整必要
                            [(self.ID, 0.0, 'line', self.ID * 10 + class_id, polygon[0][1] + y_correction, polygon[0][0] + x_correction, int(Z),
                              str(class_id))],
                            dtype=self.obj_type
                        )
                    object_data_list.append(
                        np.array(
                            #  ['ID','t','entity', 'sID','x', 'y', 'z', 'label'], 'label'の調整必要
                            [(self.ID, 0.0, 'line', self.ID * 10 + class_id, polygon[0][1] + y_correction, polygon[0][0] + x_correction, int(Z),
                              str(class_id))],
                            dtype=self.obj_type
                        )
                    )
                object_data_list.append(first_data)
                self.ID += 1
        return np.array(object_data_list)

    def creata_all_h5(self):
        for gen_id in self.name_dict.keys():
            self.bd5f = h5py.File(os.path.join(self.save_folder, f"{gen_id}_bd5.h5"), "w")
            self.data = self.bd5f.create_group('/data')
            self.set_units(gen_id)
            # self.bd5f.close()

    def run(self, gen_id):
        name_list = self.name_dict[gen_id]
        data_list = None
        for image_name in name_list:
            object_data_list = self.png2bd5(os.path.join(self.image_folder, image_name))
            if data_list is None:
                data_list = object_data_list
            else:
                data_list = np.vstack((data_list, object_data_list))
        return data_list


def main():
    png2bd5 = Png2bd5(
        image_folder="./Prediction_resize/",
        save_folder="./h5_predictions/",
    )
    png2bd5.creata_all_h5()


if __name__ == '__main__':
    main()
