from keras import models
import retrieval
import replace
import numpy as np
import generate_cnn_data as gcd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-i", "--query-map-id", dest="query_map_id", required=True)
args = parser.parse_args()

query_path = gcd.current_path() + '/data_flp/queries/query_' + args.query_map_id + '.map'
query_triples_path = gcd.current_path() + '/data_flp/queries/query_triples_' + args.query_map_id + '.map'

classificator_model = models.load_model(gcd.current_path() + '/data_flp/classificator/saves/classificator.h5')

query_triples = gcd.get_triples(query_triples_path)
room_count = len(query_triples[0])
query_data = gcd.dataset(num_classes=1, convnet_type=None, dataset_type='queries', shape=room_count,
                         query_id=args.query_map_id)
assert len(query_data) == 1
query_connmap = query_data[0]

query_data.clear()
query_data.append(gcd.get_extened_connmap(query_connmap, room_count))

max_room_count = gcd.max_room_count
assert len(query_data[0]) == max_room_count * max_room_count

pred_classes_classificator = []
for e in query_data:
    pred_classes_classificator.append(
        classificator_model.predict(np.array(e).reshape(1, max_room_count, max_room_count, 1)).argmax())
assert len(pred_classes_classificator) == 1

all_sims = retrieval.get_sims(query_path, room_count)
assert len(all_sims) == 1

query_data_file = open(query_path).readlines()
case_data_file = open(gcd.current_path() + '/data_flp/case_data/cases/0.txt').readlines()
case_triples_file = open(gcd.current_path() + '/data_flp/case_data/cases/0_uuids.txt').readlines()
# save adapted maps
file_adapted = open(gcd.current_path() + '/data_flp/results/adapted_' + args.query_map_id + '.map', 'a+')
file_adapted_extended = open(gcd.current_path() + '/data_flp/results/adapted_extended_' + args.query_map_id + '.map',
                             'a+')
file_adapted_triples = open(gcd.current_path() + '/data_flp/results/adapted_triples_' + args.query_map_id + '.map', 'a+')
query_connmap_1d = query_data_file[0]
mode = pred_classes_classificator[0]  # e.g. 0, 1, or 2
for case_num, sim in all_sims[0]:
    case_connmap_1d = case_data_file[int(case_num)]
    query_map = replace.mrx(query_connmap_1d, room_count)
    case_triples = gcd.get_triples(case_triples_file[int(case_num)])
    case_room_count = len(case_triples[0])
    case_map = replace.mrx(case_connmap_1d, case_room_count)
    try:
        query_map_replaced, query_triples_replaced = replace.replace(query_map,
                                                                     query_triples[0],
                                                                     case_map,
                                                                     case_triples[0],
                                                                     mode=mode)
        np_replaced = np.array(query_map_replaced).flatten()
        file_adapted.write(str(np_replaced).replace('\n', '') + '\n')
        query_map_replaced_extended = gcd.get_extened_connmap(query_map_replaced, len(query_map_replaced))
        file_adapted_extended.write(str(query_map_replaced_extended).replace(',', '') + '\n')
        file_adapted_triples.write(str(query_triples_replaced) + '\n')
    except TypeError:
        # TODO check why it sometimes gives 'TypeError: cannot unpack non-iterable NoneType object'
        pass
