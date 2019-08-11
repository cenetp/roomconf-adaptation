from argparse import ArgumentParser
from generate_cnn_data import current_path
from keras import models
import generate_cnn_data as gcd

parser = ArgumentParser()
parser.add_argument("-i", "--query-map-id", dest="query_map_id", required=True)
args = parser.parse_args()

discriminator_model = models.load_model(current_path() + '/data_flp/discriminator/saves/discriminator.h5')

results_data = gcd.dataset(num_classes=1, convnet_type=None, dataset_type='results', shape=gcd.max_room_count,
                           query_id='extended_'+args.query_map_id)
pred_classes_discriminator = []
for res in results_data:
    pred_classes_discriminator.append(
        discriminator_model.predict(res.reshape(1, gcd.max_room_count, gcd.max_room_count, 1)).argmax())

assert len(pred_classes_discriminator) == len(results_data)

with open(current_path() + '/data_flp/results/truefalse_' + args.query_map_id + '.txt', 'w') as f:
    result = ''
    for cls in pred_classes_discriminator:
        result += (str(cls) + '\n')
    f.write(result)
