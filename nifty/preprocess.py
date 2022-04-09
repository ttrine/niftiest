import json
import logging
import networkx as nx
import pandas as pd
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In production these will be environment variables
DATA_FILENAME = 'user_behaviour.json'
DATA_DIR = 'data'


def load_json(path):
    records = list()
    with open(path, 'r') as f:
        for line in f.readlines():
            records.append(json.loads(line))
    return records


def to_tabular(documents):
    """ Parse input documents into tables. """
    df = pd.DataFrame(documents)

    # Put each event into its own table
    tables = dict()
    tables['state'] = df.loc[df.event == 'endUserState'][['timestamp', 'userId', 'islandId', 'state', 'lengthSec']]
    tables['level_up'] = df.loc[df.event == 'levelUp'][['timestamp', 'userId', 'islandId', 'level']]
    tables['gain_xp'] = df.loc[df.event == 'gainXp'][['timestamp', 'userId', 'islandId', 'totalXp']]
    tables['quest'] = df.loc[df.event == 'questStepCompleted'][['timestamp', 'userId', 'islandId', 'questId',
                                                                'questName', 'questStep', 'questLastStep']]

    # Note: These two events were switched in the data. We switch them back here
    tables['visit_abroad'] = df.loc[df.event == 'visitHomeIsland'][['timestamp', 'userId',
                                                                    'islandId', 'visitorId']]
    tables['visit_home'] = df.loc[df.event == 'visitIsland'][['timestamp', 'userId', 'islandId', 'visitorId']]

    # This derived table provides a one-to-one mapping from island IDs
    island_user = tables['visit_home'][['userId', 'islandId']].drop_duplicates().set_index('islandId')

    return tables, island_user


def island_to_user(tables, island_user):
    """ Map island IDs to the associated user ID in each table. """
    key_cols = ['timestamp', 'guestId', 'hostId']

    # In these tables, 'userId' is always the *owner* of the island
    tables['visit_abroad'] = tables['visit_abroad'].rename(
        columns={'userId': 'hostId', 'visitorId': 'guestId'})[key_cols]

    tables['visit_home'] = tables['visit_home'].rename(
        columns={'userId': 'hostId'})[['timestamp', 'hostId']]

    # In the remaining tables, 'userId' is not necessarily the island owner (but it can be)
    rename_cols = {'userId': 'guestId', 'userId_island': 'hostId'}

    tables['state'] = tables['state'].join(island_user, on='islandId', rsuffix='_island').rename(
        columns=rename_cols)[['timestamp', 'guestId', 'hostId', 'state', 'lengthSec']]

    tables['level_up'] = tables['level_up'].join(island_user, on='islandId', rsuffix='_island').rename(
        columns=rename_cols)[key_cols + ['level']]

    tables['gain_xp'] = tables['gain_xp'].join(island_user, on='islandId', rsuffix='_island').rename(
        columns=rename_cols)[key_cols + ['totalXp']]

    tables['quest'] = tables['quest'].join(island_user, on='islandId', rsuffix='_island').rename(
        columns=rename_cols)[key_cols + ['questId', 'questName', 'questStep', 'questLastStep']]


def separate_home_from_abroad(**tables):
    """ This is a prerequisite step before we build our attributed graph. """
    separated = dict()

    for name in tables:
        if 'visit' in name:
            continue
        separated[name] = dict()

    # Visits came separated in the original data, so just need to index appropriately
    separated['visit'] = dict()
    separated['visit']['home'] = tables['visit_home']
    separated['visit']['abroad'] = tables['visit_abroad']

    for name, table in tables.items():
        if 'visit' in name:
            continue
        separated[name]['home'] = table.loc[table.guestId == table.hostId]
        separated[name]['abroad'] = table.loc[table.guestId != table.hostId]

    return separated


def build_graph(**tables):
    """ Construct graphical representation of dataset. """
    graph = nx.DiGraph()

    # Build graph structure (nodes and edges)
    adjacency_list = tables['visit']['abroad'][['guestId', 'hostId']].drop_duplicates()
    graph.add_edges_from(adjacency_list.values)

    # Assign data to appropriate part of graph
    drop_cols = ['guestId', 'hostId']

    # Store home activity in node attributes
    for node, data in graph.nodes(data=True):
        for name, table in tables.items():
            table = table['home']
            data[name] = table.loc[table.hostId == node]
            if name != 'visit':
                data[name].drop(columns=drop_cols)
            data[name].reset_index(drop=True)

    # Store abroad activity in edge attributes
    for node1, node2, data in graph.edges(data=True):
        for name, table in tables.items():
            table = table['abroad']
            data[name] = table.loc[table.guestId == node1].loc[table.hostId == node2].drop(
                columns=drop_cols).reset_index(drop=True)

    # Calculate total time spent on each edge
    for _, _, data in graph.edges(data=True):
        data['totalTime'] = data['state'].lengthSec.sum()

    return graph


def preprocess(serialize=False):
    """ Build lossless graphical representation of the dataset and optionally write it to a file. """
    path = f'{DATA_DIR}/{DATA_FILENAME}'
    try:
        data = load_json(path)
    except FileNotFoundError as e:
        logger.error(f"Unable to find raw data at {path}")
        sys.exit(1)

    logger.info("Parsing documents into tables")
    data, island_user = to_tabular(data)
    island_to_user(data, island_user)
    data = separate_home_from_abroad(**data)

    logger.info("Building graph representation (may take a few minutes)")
    data = build_graph(**data)

    if serialize:
        out_path = f"{DATA_DIR}/graph.pkl"
        logger.info(f"Writing graph to {out_path}")
        nx.write_gpickle(data, out_path)

    return data


if __name__ == '__main__':
    preprocess(serialize=True)
