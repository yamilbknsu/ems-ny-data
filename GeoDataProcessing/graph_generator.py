# Import statements
import uuid
import pickle
import igraph
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString

"""
Take the 'raw' data from OSM and make it
a usable igraph

Assumptions:
- if attribute 'oneway' exists and it's no, then the arc should
  be duplicated inversed
- if instead 'oneway' is null, then the road it's supposed to be
  bidirectional, so it's also duplicated.
"""

DATA_DIR = 'C://Users//Yamil//Proyectos//Proyectos en Git//' \
    + 'Memoria Ambulancias//ems-ny-data//Generated Shapefiles//'

if __name__ == '__main__':
    # Load the edges and nodes geojson
    print('Loading shps...')
    graph_edges = gpd.read_file(DATA_DIR + 'NYC_edges_base.geojson', driver='geojson')  # noqa: E501
    graph_nodes = gpd.read_file(DATA_DIR + 'NYC_nodes.geojson', driver='geojson')       # noqa: E501

    # Duplacting edges when necessary
    print('Initial edges amount: ', len(graph_edges))
    new_edges: dict = {column: [] for column in graph_edges.columns}
    i: int = 0
    total: int = len(graph_edges.index[graph_edges['oneway'] != 'yes'])
    for idx in graph_edges.index[graph_edges['oneway'] != 'yes']:
        # Copy all attributes except id, u, v and geometry
        for column in list(set(graph_edges.columns) - set(['edgeid', 'u', 'v', 'geometry'])):  # noqa: E501
            new_edges[column].append(graph_edges[column][idx])

        new_edges['v'].append(graph_edges['u'][idx])
        new_edges['u'].append(graph_edges['v'][idx])
        new_edges['edgeid'].append(uuid.uuid1().hex)
        new_edges['geometry'].append(LineString(graph_edges['geometry'][idx].coords[::-1]))     # noqa: E501

        i += 1
        if i % 100 == 0:
            print(i, '/', total)

    new_df: pd.DataFrame = pd.DataFrame(new_edges, columns=new_edges.keys())
    graph_edges = graph_edges.append(new_df, ignore_index=True)
    graph_edges = graph_edges.replace(b'Louis Ni\xf1\xe9 Boulevard', 'Francis Lewis Boulevard')  # noqa: E501
    graph_edges = gpd.GeoDataFrame(graph_edges, geometry='geometry')
    graph_edges.crs = "EPSG:4326"
    graph_edges.to_file(DATA_DIR + 'NYC_edges.geojson', driver='GeoJSON')
    print('Final edges amount: ', len(graph_edges))

    # Initialize the graph object
    graph = igraph.Graph()
    graph.to_directed(mutual=False)
    print('Is the graph directed? (Should be)', graph.is_directed())

    # Add the nodes and properties
    print('Add vertices...')
    graph.add_vertices(graph_nodes['osmid'])
    graph.vs['lat'] = graph_nodes['lat']
    graph.vs['lon'] = graph_nodes['lon']

    # Add edges
    print('Add edges...')

    i = 0                      # Disposable
    total = len(graph_edges)   # Disposable

    for _, edge in graph_edges.iterrows():
        u: str = edge['u']
        v: str = edge['v']
        graph.add_edge(u, v)

        i += 1
        if i % 100 == 0:
            print(i, '/', total)

        # Add edge attributes
        graph.es[i-1]['osmid'] = edge['osmid']
        graph.es[i-1]['edgeid'] = edge['edgeid']
        graph.es[i-1]['u'] = u
        graph.es[i-1]['v'] = v
        graph.es[i-1]['length'] = edge['length']
        graph.es[i-1]['name'] = edge['name']
        graph.es[i-1]['oneway'] = 'yes'
        graph.es[i-1]['lanes'] = edge['lanes']
        graph.es[i-1]['highway'] = edge['highway']
        graph.es[i-1]['maxspeed'] = edge['maxspeed']

    # Saving final graph
    with open(DATA_DIR + 'NYC_graph.pickle', 'wb') as file:
        pickle.dump(graph, file)

    print('Process completed successfully!')
