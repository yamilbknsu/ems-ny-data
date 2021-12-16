import pickle
# import igraph
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Internal imports
from dispatchAnalysis import nearest_neighbor

"""
In this script we correct some issues with the graph,
particularly the fact that there are several (around 1000) nodes
that are disconnected from the main graph.
"""


def fix_disconnected_nodes(graph):
    # Get the disconnected set of nodes
    graph_decomposition = graph.decompose()
    main_index = np.argmax([len(g.vs) for g in graph_decomposition])
    print('Found main graph with ', len(graph_decomposition[main_index].vs), ' nodes.')         # noqa E501

    # Delete the nodes that are not in the main graph
    for idx, dec in enumerate(graph_decomposition):
        if idx != main_index:
            graph.delete_vertices(dec.vs['name'])

    with open(DATA_DIR + 'NYC Graph//NYC_graph_revised.pickle', 'wb') as file:
        pickle.dump(graph, file)

    original_nodes = gpd.read_file(DATA_DIR +
                                   "NYC Graph//Before Revision//NYC_nodes.geojson")             # noqa E501

    original_nodes[original_nodes['osmid'].isin(graph.vs['name'])].to_file(DATA_DIR + "NYC Graph//NYC_nodes_revised.geojson", driver='GeoJSON') # noqa E501


def fix_speeds_shp(gdf, correct_nodes):
    u_missing_index = (gdf['u_osmid'] == -2147483648)
    v_missing_index = (gdf['v_osmid'] == -2147483648)

    u_points = []
    v_points = []

    for _, row in gdf[u_missing_index].iterrows():
        u_points.append(Point(row.geometry.coords[0]))

    for _, row in gdf[v_missing_index].iterrows():
        v_points.append(Point(row.geometry.coords[-1]))

    u_NN = nearest_neighbor(gpd.GeoDataFrame(geometry=u_points), correct_nodes)
    v_NN = nearest_neighbor(gpd.GeoDataFrame(geometry=v_points), correct_nodes)

    gdf.loc[u_missing_index, 'u_osmid'] = list(u_NN.osmid)
    gdf.loc[v_missing_index, 'v_osmid'] = list(v_NN.osmid)

    return gdf


def join_speed_edge(speeds, edges, u_nodes=None, v_nodes=None, save_dir=None):

    u_list = list(edges.geometry.apply(lambda x: Point(x.coords[0])))
    u_NN = nearest_neighbor(gpd.GeoDataFrame(geometry=u_list), u_nodes) # noqa E501

    new_columns = []
    i = 0
    total = len(edges)
    for r, row in edges.iterrows():
        u_index = (speeds['u_osmid'] == row['u'])
        v_index = (speeds['v_osmid'] == row['v'])
        speeds_sub_df = speeds[u_index & v_index]

        if len(speeds_sub_df) == 0:
            start_point = Point(row.geometry.coords[0])
            final_point = Point(row.geometry.coords[-1])

            if len(speeds[u_index]) != 0:
                coord_options = [Point(l.coords[-1]) for l in speeds[u_index].geometry]             # noqa E501
                final_distances = [final_point.distance(pt) for pt in coord_options]                # noqa E501
                selected_edge = speeds.loc[speeds[u_index].index[np.argmin(final_distances)]]       # noqa E501
                new_columns.append([row.edgeid] + list(selected_edge[['p'+str(x)+'n' for x in range(1, 6)]]))       # noqa E501
            elif len(speeds[v_index]) != 0:
                coord_options = [Point(l.coords[0]) for l in speeds[v_index].geometry]              # noqa E501
                final_distances = [start_point.distance(pt) for pt in coord_options]                # noqa E501
                selected_edge = speeds.loc[speeds[v_index].index[np.argmin(final_distances)]]       # noqa E501
                new_columns.append([row.edgeid] + list(selected_edge[['p'+str(x)+'n' for x in range(1, 6)]]))       # noqa E501
            else:
                u_index = (speeds['u_osmid'] == u_NN.loc[r, 'u_osmid'])
                coord_options = [Point(l.coords[-1]) for l in speeds[u_index].geometry]             # noqa E501
                final_distances = [final_point.distance(pt) for pt in coord_options]                # noqa E501
                selected_edge = speeds.loc[speeds[u_index].index[np.argmin(final_distances)]]       # noqa E501
                new_columns.append([row.edgeid] + list(selected_edge[['p'+str(x)+'n' for x in range(1, 6)]]))       # noqa E501

            coord_options = [Point(l.coords[-1]) for l in speeds[u_index].geometry]                 # noqa E501
            final_distances = [final_point.distance(pt) for pt in coord_options]                    # noqa E501

        else:
            new_columns.append([row.edgeid] + list(np.mean(speeds_sub_df[['p'+str(x)+'n' for x in range(1, 6)]], axis=0)))    # noqa E501

        if i % 100 == 0:
            print(i, '/', total)
        i += 1

    with open(save_dir, 'wb') as f:
        pickle.dump(new_columns, f)


def get_streets_nodes(gdf):
    u_list = list(gdf.geometry.apply(lambda x: Point(x.coords[0])))
    v_list = list(gdf.geometry.apply(lambda x: Point(x.coords[-1])))

    return (gpd.GeoDataFrame(gdf.u_osmid, geometry=u_list),
            gpd.GeoDataFrame(gdf.v_osmid, geometry=v_list))


if __name__ == "__main__":
    # Graph importing
    DATA_DIR = 'C://Users//Yamil//Proyectos//Proyectos en Git//' \
        + 'Memoria Ambulancias//'

    #with open(DATA_DIR + 'NYC Graph//Before Revision//NYC_graph.pickle', 'rb') as file:             # noqa E501
    #    graph: igraph.Graph = pickle.load(file)

    #speeds_gdf = gpd.read_file(DATA_DIR + 'Archivos memoria Vicky//Velocidades//EdgesVelocUber.shp') # noqa E501
    #nodes_gdf = gpd.read_file(DATA_DIR + 'ems-ny-data//NYC Graph//NYC_nodes_revised.geojson')        # noqa E501

    # fixed_edges = fix_speeds_shp(speeds_gdf, nodes_gdf)
    # fixed_edges.to_file(DATA_DIR + 'ems-ny-data//Generated Shapefiles//NYC_edges_speed_vicky.geojson', driver ='GeoJSON') # noqa E501

    # speeds = gpd.read_file(DATA_DIR + 'ems-ny-data//Generated Shapefiles//' +
    #                       'NYC_edges_speed_vicky.geojson', driver='GeoJSON')
    edges = gpd.read_file(DATA_DIR + 'ems-ny-data//NYC Graph//NYC_edges.geojson')                      # noqa E501

    # u_nodes, v_nodes = get_streets_nodes(speeds)
    #join_speed_edge(speeds, edges, u_nodes, save_dir=DATA_DIR + 'ems-ny-data//NYC Graph//edge_speed_vicky.pickle')         # noqa E501

    with open(DATA_DIR + '//ems-ny-data//NYC Graph//edge_speed_vicky.pickle', 'rb') as f:               # noqa E501
        speeds = pickle.load(f)

    speeds = pd.DataFrame(speeds, columns=['edgeid'] + ['p'+str(x)+'n' for x in range(1,6)])            # noqa E501
    speeds.index = speeds['edgeid']
    speeds = speeds.loc[edges['edgeid'], :]

    for i in range(1, 6):
        edges['p'+str(i)+'n'] = list(speeds['p'+str(i)+'n'])

    edges.to_file(DATA_DIR + '//ems-ny-data//NYC Graph//edge_speed_vicky.shp')
