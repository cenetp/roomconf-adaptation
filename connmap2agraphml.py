import re
import uuid
from argparse import ArgumentParser
import numpy as np
import generate_cnn_data as gcd

parser = ArgumentParser()
parser.add_argument("-i", "--query-map-id", dest="query_map_id", required=True)
args = parser.parse_args()

lines = open(gcd.current_path() + '/data_flp/results/adapted_' + args.query_map_id + '.map', 'r').readlines()
all_triples = gcd.get_triples(gcd.current_path() + '/data_flp/results/adapted_triples_' + args.query_map_id + '.map')

truefalse = []
try:
    truefalse = open(gcd.current_path() + '/data_flp/results/truefalse_' + args.query_map_id + '.txt', 'r').readlines()
except FileNotFoundError:
    pass

for i in range(len(lines)):
    if len(truefalse) == 0 or int(truefalse[i]) == 1:
        line = lines[i]
        connmap_string_raw = re.sub('\[|\]|\n', '', line)
        connmap_string = re.sub('\s+', ',', connmap_string_raw).strip(',')
        connmap_1d = np.array(connmap_string.split(','), dtype='float')
        triples = all_triples[i]

        room_count = len(triples[0])
        # TODO check why sometimes the size of connmap_1d is wrong
        try:
            assert len(connmap_1d) == room_count * room_count
        except AssertionError:
            continue

        connmap = connmap_1d.reshape((room_count, room_count))

        connections = []
        rooms = {}

        for j in range(len(connmap)):
            row = connmap[j]
            row_triples = triples[j]
            for k in range(len(row)):
                conn = row[k]
                triple = row_triples[k]
                if conn != 0.0:
                    conn = str(conn)[2:]
                    edge_code = 0
                    try:
                        edge_code = int(conn[2])
                    except IndexError:
                        pass
                    if edge_code != int(triple[2]):
                        raise ValueError
                    else:
                        # edge info
                        edge_type = gcd.edge_type_codes_inverted[gcd.edge_types_inverted[edge_code]]
                        # source info
                        code_from = int(conn[0])
                        type_from = gcd.room_type_codes_inverted[gcd.room_types_inverted[code_from]]
                        id_from = triple[0]
                        # target info
                        code_to = int(conn[1])
                        type_to = gcd.room_type_codes_inverted[gcd.room_types_inverted[code_to]]
                        id_to = triple[1]
                        # add to corresponding collections
                        rooms[id_from] = type_from
                        rooms[id_to] = type_to
                        connections.append([id_from, id_to, edge_type])

        head = '<graphml xmlns="http://graphml.graphdrawing.org/xmlns" ' \
               'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ' \
               'xsi:schemalocation="http://graphml.graphdrawing.org/xmlns     ' \
               'http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">' \
               '<graph id="searchGraph1" edgedefault="undirected">' \
               '<key id="imageUri" for="graph" attr.name="imageUri" attr.type="string"></key>' \
               '<key id="imageMD5" for="graph" attr.name="imageMD5" attr.type="string"></key>' \
               '<key id="validatedManually" for="graph" attr.name="validatedManually" attr.type="boolean"></key>' \
               '<key id="floorLevel" for="graph" attr.name="floorLevel" attr.type="float"></key>' \
               '<key id="buildingId" for="graph" attr.name="buildingId" attr.type="string"></key>' \
               '<key id="ifcUri" for="graph" attr.name="ifcUri" attr.type="string"></key>' \
               '<key id="bimServerPoid" for="graph" attr.name="bimServerPoid" attr.type="long"></key>' \
               '<key id="alignmentNorth" for="graph" attr.name="alignmentNorth" attr.type="float"></key>' \
               '<key id="geoReference" for="graph" attr.name="geoReference" attr.type="string"></key>' \
               '<key id="name" for="node" attr.name="name" attr.type="string"></key>' \
               '<key id="roomType" for="node" attr.name="roomType" attr.type="string"></key>' \
               '<key id="center" for="node" attr.name="center" attr.type="string"></key>' \
               '<key id="corners" for="node" attr.name="corners" attr.type="string"></key>' \
               '<key id="windowExist" for="node" attr.name="windowExist" attr.type="boolean"></key>' \
               '<key id="enclosedRoom" for="node" attr.name="enclosedRoom" attr.type="boolean"></key>' \
               '<key id="area" for="node" attr.name="area" attr.type="float"></key>' \
               '<key id="sourceConnector" for="edge" attr.name="sourceConnector" attr.type="string"></key>' \
               '<key id="targetConnector" for="edge" attr.name="targetConnector" attr.type="string"></key>' \
               '<key id="hinge" for="edge" attr.name="hinge" attr.type="string"></key>' \
               '<key id="edgeType" for="edge" attr.name="edgeType" attr.type="string"></key>' \
               '<data key="imageUri"></data>' \
               '<data key="imageMD5"></data>' \
               '<data key="validatedManually">false</data>' \
               '<data key="floorLevel">0.0</data>' \
               '<data key="buildingId">0</data>' \
               '<data key="ifcUri"></data>' \
               '<data key="bimServerPoid">0</data>' \
               '<data key="alignmentNorth">0.0</data>' \
               '<data key="geoReference">null</data>'
        foot = '</graph></graphml>'

        # add nodes (rooms)
        # combine connmap register with hidden rooms
        agraphml_rooms = ''
        room_ids = rooms.keys()
        for room_id in room_ids:
            agraphml_rooms += '<node id="' + room_id + '"><data key="roomType">' + rooms[room_id] + '</data></node>'

        # add edges (connections)
        agraphml_edges = ''
        for final_conn in connections:
            source = final_conn[0]
            target = final_conn[1]
            edge_type = final_conn[2]
            edge_id = str(uuid.uuid4())
            agraphml_edges += '<edge id="' + edge_id + '" source="' + source + '" target="' + target + '">' \
                '<data key="edgeType">' + edge_type + '</data></edge>'

        final_agraphml = head + agraphml_rooms + agraphml_edges + foot
        ida = args.query_map_id + '_' + str(i)
        with open(gcd.current_path() + '/data_flp/results/agraphml_' + ida + '.agraphml',
                  'w') as agraphml_file:
            agraphml_file.write(final_agraphml)
