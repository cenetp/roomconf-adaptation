import xml.etree.ElementTree as eT
from argparse import ArgumentParser

import numpy as np

import generate_cnn_data as gcd

parser = ArgumentParser()
parser.add_argument("-i", "--query-map-id", dest="query_map_id", required=True)
args = parser.parse_args()

namespace = '{http://graphml.graphdrawing.org/xmlns}'
tree = eT.parse(gcd.current_path() + '/data_flp/queries/agraphml_' + args.query_map_id + '.agraphml')
root = tree.getroot()
graph = root[0]


def find_room_type(room_id):
    for room in graph.findall(namespace + 'node'):
        if room_id == room.get('id'):
            data = room.findall(namespace + 'data')
            for d in data:
                if d.get('key') == 'roomType':
                    return d.text


room_ids = [room.get('id') for room in graph.findall(namespace + 'node')]
assert len(room_ids) <= gcd.max_room_count

length = len(room_ids)
connmap = []
triples = []
for i in range(0, length):
    triple_row = []
    id_from = ''
    try:
        id_from = room_ids[i]
    except IndexError:
        pass
    # no rows for roomcodes, reshape will be done in generator/classificator
    for j in range(0, length):
        # initialize with "no connection"
        conn = 0.0
        triple = ['', '', None]
        id_to = ''
        try:
            id_to = room_ids[j]
        except IndexError:
            pass
        if id_from != id_to:
            for edge in graph.findall(namespace + 'edge'):
                source_id = edge.get('source')
                target_id = edge.get('target')
                if id_from == source_id and id_to == target_id:
                    source_number_code = gcd.room_types[gcd.room_type_codes[find_room_type(room_ids[i])]]
                    target_number_code = gcd.room_types[gcd.room_type_codes[find_room_type(target_id)]]
                    edge_number_code = gcd.edge_types[gcd.edge_type_codes[edge.find(namespace + 'data').text]]
                    conn = float('0.' + source_number_code + target_number_code + edge_number_code)
                    triple = [id_from, id_to, edge_number_code]
        connmap.append(conn)
        triple_row.append(triple)
    assert len(triple_row) == length
    triples.append(triple_row)

assert len(connmap) == length * length
assert len(triples) == length

with open(gcd.current_path() + '/data_flp/queries/query_triples_' + args.query_map_id + '.map',
          'w') as query_triples_file:
    query_triples_file.write(str(triples))

with open(gcd.current_path() + '/data_flp/queries/query_' + args.query_map_id + '.map', 'w') as query_map_file:
    query_map_file.write(np.array2string(np.array(connmap)).replace('\n', ''))
