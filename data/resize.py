import os
from PIL import Image
import pandas as pd


def main():
    table = pd.read_csv('ImageDataTable.csv')
    os.makedirs('Prediction_resize', exist_ok=True)
    for i in range(len(table)):
        table_i = table.iloc[i]
        print(table_i['Name'])
        img = Image.open(table_i['Name'])
        img = img.resize([table_i['Y_size'], table_i['X_size']], Image.NEAREST)
        img.save(os.path.join('Prediction_resize', table_i['Name']))


if __name__ == '__main__':
    main()
