import operator
import keras_vgg16 as vgg
import generate_cnn_data as gcd


def similarity(val1, val2):
    return min(val1, val2) / max(val1, val2)


def get_sims(path, room_count):  # path to query_{id}.map
    query_data = vgg.dataset(room_count, path=path)
    assert len(query_data[0]) == vgg.max_room_count
    sims = []
    for i in range(len(query_data)):
        query_features = vgg.get_attrs(query_data[i])
        sims_case = {}
        with open(gcd.current_path() + '/' + vgg.dirname + '/case_data/cases/0_attrs.txt', 'r', encoding='utf-8')\
                as cases_file:
            cases = cases_file.readlines()
            for case in cases:
                cs = case.split(':')
                num = cs[0]
                attrs = cs[1]
                case_features = attrs[1:len(attrs)-2].split(',')
                local_sims = []
                if len(query_features) == len(case_features):
                    for j in range(len(query_features)):
                        qf = query_features[j]
                        cf = float(case_features[j].strip())
                        local_sim = similarity(qf, cf)
                        local_sims.append(local_sim)
                global_sim = sum(local_sims) / len(local_sims)
                sims_case[num] = global_sim
        sorted_sims = sorted(sims_case.items(), key=operator.itemgetter(1))
        sims.append(sorted_sims[-10:])
    return sims
