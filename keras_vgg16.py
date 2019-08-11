from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import generate_cnn_data as gcd

dirname = 'data_flp'

max_room_count = 224


def dataset(shape, path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            vect = line[1:len(line)-2]
            vect = vect.split(' ')
            mtrx = []
            for v in vect:
                val = v.strip()
                if len(val) > 0:
                    value = int(float(val + '0') * 100)
                    mtrx.append(value)
            np_mtrx = np.array(mtrx).reshape((shape, shape))
            full_mtrx = []
            for i in range(0, max_room_count):
                row = []
                for j in range(0, max_room_count):
                    try:
                        if np_mtrx[i][j] != 0:
                            full_value = np_mtrx[i][j]
                            row.append([full_value, full_value, full_value])
                        else:
                            row.append([0, 0, 0])
                    except:
                        row.append([0, 0, 0])
                full_mtrx.append(row)
            data.append(full_mtrx)
    return data


model = VGG16(weights='imagenet', include_top=False)


def get_attrs(case):
    x = np.expand_dims(case, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    # print(len(features))            # 1 # features
    # print(len(features[0]))         # 7 # f1
    # print(len(features[0][0]))      # 7 # f2
    # print(len(features[0][0][0]))   # 512 # f3
    attrs = []
    for i1 in range(len(features)):
        f1 = features[i1]
        for i2 in range(len(f1)):
            f2 = f1[i2]
            sums2 = []
            for i3 in range(len(f2)):
                f3 = f2[i3]
                sum3 = sum(f3)
                sums2.append(sum3)
            sum2 = sum(sums2)
            attrs.append(sum2)
    return attrs


# case_data = dataset(gcd.max_room_count, path=gcd.current_path() + '/data_flp/case_data/cases/0_extended.txt')
# file = open(gcd.current_path() + '/data_flp/case_data/cases/0_attrs.txt', 'a+')
# for i in range(len(case_data)):
#     case = case_data[i]
#     file.write(str(i) + ':' + str(get_attrs(case)) + '\n')

