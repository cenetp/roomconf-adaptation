import os
import sys
import uuid
import numpy as np
import random

max_room_count = 24
dirname = 'data_flp'

room_type_codes = {
    'ROOM': 'r',
    'LIVING': 'l',
    'SLEEPING': 's',
    'WORKING': 'w',
    'KITCHEN': 'k',
    'CORRIDOR': 'c',
    'BATH': 'b',
    'TOILET': 't',
    'CHILDREN': 'h',
    'STORAGE': 'g'
}

room_type_codes_inverted = {
    'r': 'ROOM',
    'l': 'LIVING',
    's': 'SLEEPING',
    'w': 'WORKING',
    'k': 'KITCHEN',
    'c': 'CORRIDOR',
    'b': 'BATH',
    't': 'TOILET',
    'h': 'CHILDREN',
    'g': 'STORAGE'
}

room_types = {
    'r': '0',  # room
    'l': '1',  # living
    's': '2',  # sleeping
    'w': '3',  # working
    'k': '4',  # kitchen
    'c': '5',  # corridor
    'b': '6',  # bath
    't': '7',  # toilet
    'h': '8',  # children
    'g': '9'  # storage
}

room_types_inverted = {
    0: 'r',
    1: 'l',
    2: 's',
    3: 'w',
    4: 'k',
    5: 'c',
    6: 'b',
    7: 't',
    8: 'h',
    9: 'g'
}

edge_type_codes = {
    'EDGE': 'e',
    'DOOR': 'd',
    'PASSAGE': 'p',
    'WALL': 'w',
    'ENTRANCE': 'r',
    'SLAB': 'b',
    'STAIRS': 's',
    'WINDOW': 'n'
}

edge_type_codes_inverted = {
    'e': 'EDGE',
    'd': 'DOOR',
    'p': 'PASSAGE',
    'w': 'WALL',
    'r': 'ENTRANCE',
    'b': 'SLAB',
    's': 'STAIRS',
    'n': 'WINDOW'
}

edge_types = {
    'e': '0',  # edge
    'd': '1',  # door
    'p': '2',  # passage
    'w': '3',  # wall
    'r': '4',  # ENTRANCE
    'b': '5',  # SLAB
    's': '6',  # STAIRS
    'n': '7'  # WINDOW
}

edge_types_inverted = {
    0: 'e',  # edge
    1: 'd',  # door
    2: 'p',  # passage
    3: 'w',  # wall
    4: 'r',  # ENTRANCE
    5: 'b',  # SLAB
    6: 's',  # STAIRS
    7: 'n'  # WINDOW
}


def current_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def get_triples(path_or_connmap):
    triples = []
    lines = []
    if str(path_or_connmap).startswith('['):
        lines.append(path_or_connmap)
    else:
        lines = open(path_or_connmap, 'r', encoding='utf-8').readlines()
    for line in lines:
        line = line[1:len(line) - 1]
        splitted_1 = line.split(']], [[')
        t1 = []
        for s1 in splitted_1:
            splitted_2 = s1.split('], [')
            t2 = []
            for s2 in splitted_2:
                splitted_3 = s2.strip('[[').strip(']]]').split(', ')
                t3 = []
                for s3 in splitted_3:
                    s4 = s3.strip('\'')
                    if s4 == 'None':
                        s4 = None
                    t3.append(s4)
                t2.append(t3)
            t1.append(t2)
        triples.append(t1)
    return triples


def generate_connmap(room_count, freq):
    connmap = []
    connmap_triples = []  # we use connection triples in the form [id_from, id_to, edge_type]
    rooms = []
    rooms_uuids = []
    # 1. create rooms list
    for i1 in range(room_count):
        room_num = random.randrange(1, 10)  # no use of anonymous "room", it will only be used for decoding
        rooms.append(room_types_inverted[room_num])
        rooms_uuids.append(str(uuid.uuid4()))
    # rooms.sort()
    # rooms = [room_types_inverted[room_num] for room_num in rooms]
    # 2. initialize with 0.0s and empty triples
    for i2 in range(1, len(rooms) + 1):
        row = []
        row_triples = []
        for j2 in range(1, len(rooms) + 1):
            row.append(0.0)
            row_triples.append(['', '', None])
        connmap.append(row)
        connmap_triples.append(row_triples)
    # 3. fill with connections
    for i3 in range(len(rooms)):
        row = connmap[i3]
        for j3 in range(len(row)):
            if i3 != j3:
                add_conn = random.randrange(0, freq)
                if add_conn == 0:  # 0 = add connection
                    edge_num = random.randrange(1, 8)  # no use of "edge" as conn, it will only be used for decoding
                    conn = room_types[rooms[i3]] + room_types[rooms[j3]] + str(edge_num)
                    connmap[i3][j3] = float('0.' + conn)
                    connmap_triples[i3][j3] = [rooms_uuids[i3], rooms_uuids[j3], edge_num]
    return connmap, connmap_triples


def get_extened_connmap(connmap, room_count):
    connmap_extended = []
    for n in range(len(connmap)):
        row = connmap[n]
        new_row = []
        for p in range(room_count):
            new_row.append(row[p])
        for k in range(room_count, max_room_count):
            new_row.append(0.0)
        connmap_extended.extend(new_row)
    for l in range(room_count, max_room_count):
        new_row = []
        for m in range(max_room_count):
            new_row.append(0.0)
        connmap_extended.extend(new_row)
    return connmap_extended


def generate_data(convnet_type, dataset_type, num_classes, amount, mode):
    for i in range(num_classes):
        r = 5  # default random
        if mode == 'no_default_random':
            r = random.randrange(3, 7)
        file_connmap = open(current_path() + '/' + dirname + '/' + convnet_type + '/' + dataset_type + '/' + str(i)
                            + '.txt', 'a+', encoding='utf-8')
        file_connmap_extended = open(
            current_path() + '/' + dirname + '/' + convnet_type + '/' + dataset_type + '/' + str(i) + '_extended.txt',
            'a+', encoding='utf-8')
        file_room_uuids = open(current_path() + '/' + dirname + '/' + convnet_type + '/' + dataset_type + '/' + str(i)
                               + '_uuids.txt', 'a+', encoding='utf-8')
        for j in range(amount):
            if mode != 'no_default_random':
                r = random.randrange(2, 8)
            room_count = random.randrange(10, max_room_count)
            connmap, connmap_triples = generate_connmap(room_count, r)
            if room_count < max_room_count:
                connmap_extended = get_extened_connmap(connmap, room_count)
                file_connmap_extended.write(
                    np.array2string(np.array(connmap_extended).flatten(), max_line_width=5000) + '\n')
            else:
                file_connmap_extended.write(np.array2string(np.array(connmap).flatten(), max_line_width=5000) + '\n')
            file_connmap.write(np.array2string(np.array(connmap).flatten(), max_line_width=5000) + '\n')
            file_room_uuids.write(str(connmap_triples) + '\n')


# generate_data('case_data', 'cases', 1, 30, 'no_default_random')


def dataset(num_classes, convnet_type, dataset_type, shape, query_id):
    data = []
    cnn = ''
    if convnet_type is not None:
        cnn = '/' + convnet_type
    for i in range(num_classes):
        filename = str(i) + '_extended.txt'
        if num_classes == 1:
            if dataset_type == 'queries':
                filename = 'query_' + query_id + '.map'
            elif dataset_type == 'results':
                filename = 'adapted_' + query_id + '.map'
        with open(current_path() + '/' + dirname + cnn + '/' + dataset_type + '/' + filename, 'r', encoding='utf-8') \
                as file:
            lines = file.readlines()
            for j in range(len(lines)):
                line = lines[j]
                vect = line[1:len(line) - 2]
                vect = vect.split(' ')
                mtrx = []
                for v in vect:
                    val = v.strip()
                    if len(val) > 0:
                        mtrx.append(float(val + '0'))
                np_mtrx = None
                if shape is not None:
                    np_mtrx = np.array(mtrx).reshape((shape, shape))
                else:
                    np_mtrx = np.array(mtrx).reshape((max_room_count, max_room_count))
                data.append(np_mtrx)
    return data


def classes(num_classes, convnet_type, dataset_type):
    data = []
    for i in range(num_classes):
        with open(current_path() + '/' + dirname + '/' + convnet_type + '/' + dataset_type + '/' + str(i) + '.txt', 'r',
                  encoding='utf-8') as file:
            for j in range(len(file.readlines())):
                data.append(i)
    return data
