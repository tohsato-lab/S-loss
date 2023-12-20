import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class IoU:
    def __init__(self, n_classes, ignore_area=[0]):
        self.n_class = n_classes
        self.ignore_area = ignore_area
        self.id_iou_dict = {}
        self.id_and_or_dict = {}
        for area_id in range(n_classes):
            self.id_and_or_dict[area_id] = {"AND": 0, "OR": 0}
            self.id_iou_dict[area_id] = []
        self.all_data_dict = {}

    def cal_iou_from_img(self, outputs_data: np.array, label_data: np.array, name=None):
        """
        画像からIoUを計算し、self.all_data_dictに”AND”と”OR”のピクセル数をためておく
        """

        # count 'and' 'or' pix
        for b_n in range(outputs_data.shape[0]):
            outputs = outputs_data[b_n]
            label = label_data[b_n]
            self.all_data_dict[name[b_n]] = {}
            for area_id in range(self.n_class):
                data = {"AND": 0, "OR": 0, 'True': 0}
                area_outputs = outputs == area_id
                area_label = label == area_id
                and_area = np.sum(np.logical_and(area_outputs, area_label))
                or_area = np.sum(np.logical_or(area_outputs, area_label))
                data['AND'] = and_area.item()
                data['OR'] = or_area.item()
                data['True'] = np.sum(area_label).item()
                self.all_data_dict[name[b_n]][area_id] = data
                self.id_and_or_dict[area_id]["AND"] += and_area.item()
                self.id_and_or_dict[area_id]["OR"] += or_area.item()

    def get_m_iou(self):
        iou_dict = {}
        for image_name in self.all_data_dict.keys():
            for area_id in self.all_data_dict[image_name].keys():
                and_pix = self.all_data_dict[image_name][area_id]['AND']
                or_pix = self.all_data_dict[image_name][area_id]['OR']
                true_pix = self.all_data_dict[image_name][area_id]['True']
                if true_pix > 0:
                    if area_id in iou_dict:
                        IoU = and_pix / or_pix
                        iou_dict[area_id].append(IoU)
                    else:
                        IoU = and_pix / or_pix
                        iou_dict[area_id] = [IoU]

        all_data = []
        for area_id in iou_dict:
            if area_id == 0:
                continue
            else:
                iou_list = iou_dict[area_id]
                all_data.extend(iou_list)
        mIoU = np.array(all_data).mean()
        return mIoU

    def _get_m_iou(self):
        mIoU_list = []
        for image_name in self.all_data_dict.keys():
            img_IoU = []
            for area_id in self.all_data_dict[image_name].keys():
                and_pix = self.all_data_dict[image_name][area_id]['AND']
                or_pix = self.all_data_dict[image_name][area_id]['OR']
                true_pix = self.all_data_dict[image_name][area_id]['True']
                if true_pix > 0:
                    IoU = and_pix / or_pix
                    img_IoU.append(IoU)
            img_mIoU = np.array(img_IoU).mean().item()
            mIoU_list.append(img_mIoU)
        mIoU = np.array(mIoU_list).mean()
        return mIoU

    def get_area_iou_dict(self):
        """
        ”AND”と”OR”のピクセル数が保存されたself.id_and_or_dictをもとにｍIoUを計算する。
        その後、各領域のIoUが保存されたdictを出力する。
        """
        # get each class IOU
        for area_id in range(self.n_class):
            if self.id_and_or_dict[area_id]["OR"] == 0:
                iou = 0
            else:
                iou = self.id_and_or_dict[area_id]["AND"] / self.id_and_or_dict[area_id]["OR"]
            self.id_iou_dict[area_id] = iou
        return self.id_iou_dict

    def get_all_iou_dict(self):
        """
        画像に対する各領域のピクセル数が入ったdictを返す。
        self.all_data_dict{ImageName_1:{AreaName_1:{"AND": ***, "OR":***, "True": ***},
                                        AreaName_2:{"AND": ***, "OR":***, "True": ***}},・・・
                           ImageName_2:{AreaName_1:{"AND": ***, "OR":***, "True": ***},
                                        AreaName_2:{"AND": ***, "OR":***, "True": ***}},・・・
                          }

        """
        return self.all_data_dict

    def save_all_dict(self, save_dict_path):
        """
        画像に対する各領域のピクセル数が入ったdictをyaml形式で保存する。
        """
        iou_data = self.get_all_iou_dict()
        with open(save_dict_path, "w") as f:
            yaml.dump(iou_data, f)


    def cal_gen_iou_from_all_dict(self, save_path):
        """
        GEN_GL_TM(G2AN_TM115_L など)ごとにIoUを計算し、
        ファイルに保存する。
        """

        all_iou_dict = self.get_all_iou_dict()
        gen_GL_TM_dict = {}
        for keys in sorted(all_iou_dict.keys()):
            image_name = keys
            name_split = image_name.split('_')
            GenType = name_split[0]
            TM = name_split[1].replace('TM', '')
            POS = name_split[2]
            GL = name_split[3][0]
            one_img_iou_data = all_iou_dict[keys]

            gen_GL_TM = f"{GenType}_{GL}_{TM}"
            if gen_GL_TM in gen_GL_TM_dict:
                gen_GL_TM_dict[gen_GL_TM]["NUM"] += 1
            else:
                gen_GL_TM_dict[gen_GL_TM] = {i: {"AND": 0, "OR": 0} for i in range(self.n_class)}
                gen_GL_TM_dict[gen_GL_TM]["NUM"] = 1

            for area_id in one_img_iou_data.keys():
                if area_id == 'NUM':
                    continue
                gen_GL_TM_dict[gen_GL_TM][area_id]["AND"] += one_img_iou_data[area_id]["AND"]
                gen_GL_TM_dict[gen_GL_TM][area_id]["OR"] += one_img_iou_data[area_id]["OR"]

        save_data = {}
        for gen_GL_TM in sorted(gen_GL_TM_dict.keys()):
            iou_list = []
            for area_id in gen_GL_TM_dict[gen_GL_TM].keys():
                if area_id == 'NUM':
                    image_counter = gen_GL_TM_dict[gen_GL_TM][area_id]
                elif area_id == 0:
                    continue
                else:
                    AND = gen_GL_TM_dict[gen_GL_TM][area_id]['AND']
                    OR = gen_GL_TM_dict[gen_GL_TM][area_id]['OR']
                    if OR == 0 or AND == 0:
                        iou = 0
                    else:
                        iou = AND / OR
                    iou_list.append(iou)
            if len(iou_list) == 0:
                m_iou = 0
            else:
                m_iou = np.mean(np.array(iou_list))
            save_data[gen_GL_TM] = {'Name': gen_GL_TM, "len": image_counter, "miou": m_iou}

        with open(save_path, "w") as f:
            for keys in save_data.keys():
                f.write("{: >13},{: >4},   {},\n".format(keys, save_data[keys]['len'], save_data[keys]['miou']))

    def save_csv_data(self, csv_path):
        """
        すべてのデータを「画像名、各領域（９つある）、mIoU（画像ごと）」でCSV形式で保存する
        """
        data_list = []
        for keys in sorted(self.all_data_dict.keys()):

            image_name = keys
            name_split = image_name.split('_')
            GenType = name_split[0]
            TM = name_split[1].replace('TM', '')
            POS = '{:0>4}'.format(name_split[2])
            GL = name_split[3][0]

            column_list = [image_name, GenType, TM, POS, GL]

            column = self.all_data_dict[keys]
            iou_list = []
            for area_id in sorted(column.keys()):
                if area_id == 0:
                    iou = column[area_id]['AND'] / column[area_id]['OR']
                    column_list.append(iou)
                else:
                    if column[area_id]['OR'] == 0 or column[area_id]['AND'] == 0:
                        iou = 0
                    else:
                        iou = column[area_id]['AND'] / column[area_id]['OR']
                        iou_list.append(iou)
                    column_list.append(iou)

            if len(iou_list) == 0:
                m_iou = 0
            else:
                iou_array = np.array(iou_list)
                m_iou = np.mean(iou_array)

            column_list.append(m_iou)
            data_list.append(column_list)

        data_table = pd.DataFrame(data_list, columns=['Name', 'GenType', 'TM', 'Pos', 'GL',
                                                      'BG', 'Isocortex', 'Olfactory area', 'Hippocampal formation',
                                                      'Cerebral nuclei', 'Interbrain', 'Midbrain', 'Hindbrain',
                                                      'Cerebellum', 'mIoU'])
        data_table.to_csv(csv_path)



def cal_from_csv(csv_path, save_path):
    data_table = pd.read_csv(csv_path, index_col=0)
    iou_data = {}
    for gen in data_table['GenType'].unique():
        data_table_gen = data_table.loc[data_table['GenType'].str.match(f"{gen}")]
        for GL in data_table_gen['GL'].unique():
            data_table_gen_GL = data_table_gen.loc[data_table_gen['GL'].str.match(f"{GL}")]
            for TM in data_table_gen_GL['TM'].unique():
                data_table_gen_GL_TL = data_table_gen_GL.loc[data_table_gen_GL['TM'].str.match(f"{TM}")]
                iou_list = data_table_gen_GL_TL['mIoU'].tolist()
                miou = np.mean(np.array(iou_list)[1:])
                iou_data[f"{gen}_{GL}_{TM}"] = {"miou": miou, "len": len(iou_list)}

    with open(save_path, "w") as f:
        for keys in iou_data.keys():
            f.write("{: >13},{: >4},   {},\n".format(keys, iou_data[keys]['len'], iou_data[keys]['miou']))