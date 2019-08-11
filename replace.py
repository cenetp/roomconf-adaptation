import random
import numpy as np


def replace(query_map, query_triples, case_map, case_triples, mode):
    # as a rule of thumb: case map size must be equal or bigger than query map size
    if len(case_map) >= len(query_map):
        # cache the current query map size
        length_old = len(query_map)
        rnd = 2
        if mode == 0:
            rnd = 4
        # should rooms from the case be appended to the existing ones in the query?
        append_rooms = random.randrange(0, rnd)
        if append_rooms == 0:  # 0 = append new rooms
            for r in range(0, len(query_map)):
                q_row = query_map[r]
                qt_row = query_triples[r]
                ct_row = case_triples[r]
                for n in range(len(q_row), len(case_map)):
                    q_row.append(0.0)
                    qt_row.append(ct_row[n])
                query_map[r] = q_row
                query_triples[r] = qt_row
        # replace the rooms in query with rooms from the case
        for i in range(0, len(case_map)):
            case_map_row = case_map[i]
            case_triples_row = case_triples[i]
            # find the room from the current map row
            case_row_room = ''
            case_triples_row_room_id = ''
            for triple in case_triples_row:
                if triple[0] != '':
                    case_triples_row_room_id = triple[0]
                    break
            if case_triples_row_room_id == '':
                for triple_row in case_triples:
                    triple_check = triple_row[i]
                    if triple_check[1] != '':
                        case_triples_row_room_id = triple_check[1]
                        break
            for conn in case_map_row:
                if conn > 0:
                    case_row_room = str(conn)[2]
                    break
            if case_row_room == '':
                for row in case_map:
                    conn_check = row[i]
                    if conn_check > 0:
                        case_row_room = str(conn_check)[3]
                        break
            if case_row_room != '' and case_triples_row_room_id != '':
                # should the room be replaced?
                replace_room = random.randrange(0, rnd)
                try:
                    query_map_row = query_map[i]
                    query_triples_row = query_triples[i]
                    if replace_room == 0:
                        for j in range(0, len(case_map_row)):
                            try:
                                query_map_row_conn = query_map_row[j]
                                query_triples_row_triple = query_triples_row[j]
                                if query_map_row_conn > 0:
                                    query_conn = list(str(query_map_row_conn))
                                    query_conn[2] = case_row_room
                                    query_map[i][j] = float(''.join(query_conn))
                                    query_triples_row_triple[0] = case_triples_row_room_id
                                    query_triples[i][j] = query_triples_row_triple
                            except IndexError:
                                if append_rooms == 0:
                                    query_map[i][j] = case_map_row[j]
                                    query_triples[i][j] = case_triples_row[j]
                            for k in range(0, len(query_map)):
                                query_row = query_map[k]
                                query_tr_row = query_triples[k]
                                if query_row[i] > 0:
                                    conn_for_replace = list(str(query_row[i]))
                                    conn_for_replace[3] = case_row_room
                                    query_map[k][i] = float(''.join(conn_for_replace))
                                    triple_for_replace = query_tr_row[i]
                                    triple_for_replace[1] = case_triples_row_room_id
                                    query_triples[k][i] = triple_for_replace
                except IndexError:
                    if append_rooms == 0:
                        query_map.append(case_map_row)
                        query_triples.append(case_triples_row)
        # post-processing: replace "to" rooms of appended rows with their respective new codes and new IDs
        # new codes and new IDs will be taken from replaced (or old) rooms
        if append_rooms == 0:
            room_list = []
            triple_list = []
            for m in range(0, length_old):
                query_row_room = ''
                query_row_triple_room_id = ''
                row = query_map[m]
                row_triples = query_triples[m]
                for conn in row:
                    if conn > 0:
                        query_row_room = str(conn)[2]
                        break
                if query_row_room == '':
                    for row in query_map:
                        conn_check = row[m]
                        if conn_check > 0:
                            query_row_room = str(conn_check)[3]
                            break
                for tr in row_triples:
                    if tr[0] != '':
                        query_row_triple_room_id = tr[0]
                        break
                if query_row_triple_room_id == '':
                    for row_trs in query_triples:
                        triple_check = row_trs[m]
                        if triple_check[1] != '':
                            query_row_triple_room_id = triple_check[1]
                            break
                room_list.append(query_row_room)
                triple_list.append(query_row_triple_room_id)
            for l in range(length_old, len(query_map)):
                row = query_map[l]
                row_triples = query_triples[l]
                for x in range(0, length_old):
                    conn_current = row[x]
                    triple_final = row_triples[x]
                    if conn_current > 0:
                        if room_list[x] != str(conn_current)[3]:
                            conn_final = list(str(conn_current))
                            conn_final[3] = room_list[x]
                            query_map[l][x] = float(''.join(conn_final))
                            triple_final[1] = triple_list[x]
                            query_triples[l][x] = triple_final
        return query_map, query_triples
    return None


def mrx(line, shape):
    line = line[1:len(line) - 2]
    temp = line.split(' ')
    mm = []
    for t in temp:
        try:
            mm.append(float(str(t).strip()))
        except ValueError:
            pass
    connmap = []
    for row in np.array(mm).reshape((shape, shape)):
        new_row = []
        for value in row:
            new_row.append(value)
        connmap.append(new_row)
    return connmap
