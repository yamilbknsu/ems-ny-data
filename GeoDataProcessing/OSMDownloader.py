# flake8: noqa
import uuid
import copy
import dill
import pickle
import igraph
import xml.sax
import os.path
import pandas as pd
import geopandas as gpd
from functools import partial
import pyproj
from shapely.ops import transform
from shapely.geometry import LineString, Point

"""
Function for downloading and processing overpass API data from:

https://gist.github.com/rajanski/ccf65d4f5106c2cdc70e

"""

class Node:
    def __init__(self, id, lon, lat):
        self.id = id
        self.lon = lon
        self.lat = lat
        self.tags = {}
        
class Way:
    def __init__(self, id, osm):
        self.osm = osm
        self.id = id
        self.nds = []
        self.tags = {}
        
    def split(self, dividers):
        # slice the node-array using this nifty recursive function
        def slice_array(ar, dividers):
            for i in range(1,len(ar)-1):
                if dividers[ar[i]]>1:
                    #print "slice at %s"%ar[i]
                    left = ar[:i+1]
                    right = ar[i:]
                    
                    rightsliced = slice_array(right, dividers)
                    
                    return [left]+rightsliced
            return [ar]
            


        slices = slice_array(self.nds, dividers)
        
        # create a way object for each node-array slice
        ret = []
        i=0
        for slice in slices:
            littleway = copy.copy( self )
            littleway.id += "-%d"%i
            littleway.nds = slice
            ret.append( littleway )
            i += 1
            
        return ret
        
class OSM:
    def __init__(self, filename_or_stream):
        """ File can be either a filename or stream/file object."""
        nodes = {}
        ways = {}
        
        superself = self
        
        class OSMHandler(xml.sax.ContentHandler):
            @classmethod
            def setDocumentLocator(self,loc):
                pass
            
            @classmethod
            def startDocument(self):
                pass
                
            @classmethod
            def endDocument(self):
                pass
                
            @classmethod
            def startElement(self, name, attrs):
                if name=='node':
                    self.currElem = Node(attrs['id'], float(attrs['lon']), float(attrs['lat']))
                elif name=='way':
                    self.currElem = Way(attrs['id'], superself)
                elif name=='tag':
                    self.currElem.tags[attrs['k']] = attrs['v']
                elif name=='nd':
                    self.currElem.nds.append( attrs['ref'] )            #pylint: disable=no-member
                
            @classmethod
            def endElement(self,name):
                if name=='node':
                    nodes[self.currElem.id] = self.currElem
                elif name=='way':
                    ways[self.currElem.id] = self.currElem
                
            @classmethod
            def characters(self, chars):
                pass
 
        xml.sax.parse(filename_or_stream, OSMHandler)
        
        self.nodes = nodes
        self.ways = ways
        #"""   
        #count times each node is used
        node_histogram = dict.fromkeys( self.nodes.keys(), 0 )
        for way in self.ways.values():
            if len(way.nds) < 2:       #if a way has only one node, delete it out of the osm collection
                del self.ways[way.id]
            else:
                for node in way.nds:
                    node_histogram[node] += 1
        
        #use that histogram to split all ways, replacing the member set of ways
        new_ways = {}
        for _, way in self.ways.items():
            split_ways = way.split(node_histogram)
            for split_way in split_ways:
                new_ways[split_way.id] = split_way
        self.ways = new_ways

def download_osm(left,bottom,right,top,highway_cat):
    """
    Downloads OSM street (only highway-tagged) Data using a BBOX, 
    plus a specification of highway tag values to use
    Parameters
    ----------
    left,bottom,right,top : BBOX of left,bottom,right,top coordinates in WGS84
    highway_cat : highway tag values to use, separated by pipes (|), for instance 'motorway|trunk|primary'
    Returns
    ----------
    stream object with osm xml data
    """

    #Return a filehandle to the downloaded data."""
    from urllib.request import urlopen
    print ("trying to download osm data from " + str(left),str(bottom),str(right),str(top)+" with highways of categories "+highway_cat)
    try:    
        print("downloading osm data from "+str(left),str(bottom),str(right),str(top)+" with highways of categories"+highway_cat)
        fp = urlopen( "http://www.overpass-api.de/api/xapi?way[highway=%s][bbox=%f,%f,%f,%f]"%(highway_cat,left,bottom,right,top) )
        print('Download succesful!')
        return fp
    except:
        print("osm data download unsuccessful")

def osm_to_shp(filename_or_stream, output_file, name, remove_intermediate_nodes = True, boundaries_shape = None):
    """
    Read graph in OSM format from file specified by name or by stream object.
    Parameters
    ----------
    filename_or_stream : filename or stream object
    """
    if os.path.exists('DL Data/'+name+'.pickle'):
        with open('DL Data/'+name+'.pickle', 'rb') as f:
            osm = dill.load(f)
    else:
        osm = OSM(filename_or_stream)
        with open('DL Data/'+name+'.pickle', 'wb') as f:
            dill.dump(osm, f)
    
    edges_data = {'osmid':[], 'edgeid':[], 'u':[], 'v':[], 'linestring':[], 'length':[]}
    used_nodes = []
    tags = []
    total = len(osm.ways)
    i = 0
    for _, edge in osm.ways.items():
        # Show count
        i += 1
        if i%1000 == 0:
            print(i,'/',total)

        # Retrieve data
        linestring = LineString([Point(osm.nodes[p].lon, osm.nodes[p].lat) for p in edge.nds])
        if boundaries_shape is None or linestring.within(boundaries_shape):
            edges_data['osmid'].append(edge.id)
            edges_data['edgeid'].append(uuid.uuid1().hex)
            edges_data['u'].append(edge.nds[0])
            edges_data['v'].append(edge.nds[-1])
            edges_data['linestring'].append(linestring)            
            project = partial(
                        pyproj.transform,
                        pyproj.Proj(init='epsg:4326'), # source coordinate system
                        pyproj.Proj(init='epsg:3857')) # destination coordinate system
            prj_linestring = transform(project, linestring)  # apply projection
            edges_data['length'].append(prj_linestring.length)

            if remove_intermediate_nodes:
                if edge.nds[0] not in used_nodes:
                    used_nodes.append(edge.nds[0])
                if edge.nds[-1] not in used_nodes:
                    used_nodes.append(edge.nds[-1])

            for tag in edge.tags.keys():
                if tag not in tags:
                    tags.append(tag)
                    edges_data[tag] = [None]*(len(edges_data['osmid']) - 1)
            
            for tag in tags:
                if tag in edge.tags.keys():
                    edges_data[tag].append(edge.tags[tag])
                else:
                    edges_data[tag].append(None)
    
    edges_df = pd.DataFrame(edges_data, columns = edges_data.keys())
    edges_df = edges_df[['osmid', 'edgeid', 'u', 'v', 'linestring', 'length', 'name', 'oneway', 'lanes', 'highway', 'maxspeed']]
    edges_gdf = gpd.GeoDataFrame(edges_df, geometry = 'linestring')
    edges_gdf.to_file(output_file+'_edges.geojson', driver = 'GeoJSON')

    if remove_intermediate_nodes:
        osm.nodes = {id: osm.nodes[id] for id in used_nodes }

    nodes_data = {'osmid':[], 'lat':[], 'lon':[]}
    tags = []
    total = len(osm.nodes)
    i = 0
    for _, node in osm.nodes.items():
        # Show count
        i += 1
        if i%1000 == 0:
            print(i,'/',total)

        # Retrieve data
        if boundaries_shape is None or Point(node.lon, node.lat).within(boundaries_shape):
            nodes_data['osmid'].append(node.id)
            nodes_data['lat'].append(node.lat)
            nodes_data['lon'].append(node.lon)

            for tag in node.tags.keys():
                if tag not in tags:
                    tags.append(tag)
                    nodes_data[tag] = [None]*(len(nodes_data['osmid']) - 1)
            
            for tag in tags:
                if tag in node.tags.keys():
                    nodes_data[tag].append(node.tags[tag])
                else:
                    nodes_data[tag].append(None)
    
    nodes_df = pd.DataFrame(nodes_data, columns = nodes_data.keys())
    nodes_df = nodes_df[['osmid', 'lat', 'lon']]
    nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry=gpd.points_from_xy(nodes_df.lon, nodes_df.lat))
    nodes_gdf.to_file(output_file+'_nodes.geojson', driver = 'GeoJSON')
    
def apply_speed_data(uber_dataset, df, base_hour = 5):
    uber_dataset = uber_dataset.drop(['year', 'month', 'utc_timestamp', 'segment_id',               # noqa: E501
        'start_junction_id', 'end_junction_id'], axis=1)

    #uber_dataset['hour_interval'] = pd.cut(uber_dataset.hour, [0, 6, 10, 13, 16, 19, 23])           # noqa: E501
    uber_dataset = uber_dataset.drop(['osm_start_node_id', 'osm_end_node_id'], axis=1)      # noqa: E501
    uber_dataset = uber_dataset.groupby(['osm_way_id', 'hour'], as_index=False).mean()     # noqa: E501
    uber_dataset = uber_dataset.drop('day', axis=1)

    mph_to_mps = 0.44704
    
    hour_intervals = uber_dataset['hour'].unique()
    interval_names = [str(h) for h in list(hour_intervals)]

    u = 0
    total = len(df) * len(interval_names)

    for i, interval in enumerate(hour_intervals):
        mean_values = []
        std_values = []
        sub_group = (uber_dataset['hour'] == interval)
        mean = uber_dataset[sub_group]['speed_mph_mean'].mean() * mph_to_mps
        std = uber_dataset[sub_group]['speed_mph_stddev'].mean() * mph_to_mps
        for _, row in df.iterrows():
            osmid = row['osmid'][:row['osmid'].index('-')]
            uber_row = uber_dataset[(uber_dataset['osm_way_id'] == int(osmid)) & sub_group]
            if len(uber_row) > 0 and not uber_row['speed_mph_mean'].isnull().any():
                mean_values.append(float(uber_row['speed_mph_mean']) * mph_to_mps)
                std_values.append(float(uber_row['speed_mph_stddev']) * mph_to_mps)
            else:
                uber_row = uber_dataset[uber_dataset['osm_way_id'] == int(osmid)].groupby('osm_way_id').mean()
                if len(uber_row) > 0 and not uber_row['speed_mph_mean'].isnull().any():
                    mean_values.append(float(uber_row['speed_mph_mean']) * mph_to_mps)
                    std_values.append(float(uber_row['speed_mph_stddev']) * mph_to_mps)
                else:
                    mean_values.append(mean)
                    std_values.append(std)
            
            u += 1
            if u%100 == 0:
                print(u,'/', total)

        df['speed_mean_' + str((int(interval_names[i]) + 5)%24)] = mean_values
        df['speed_std_' + str((int(interval_names[i]) + 5)%24)] = std_values

    return df


highway_cat = 'motorway|trunk|primary|secondary|tertiary|road|residential|motorway_link|trunk_link|primary_link|secondary_link|teriary_link'

top = 40.9503
bottom = 40.4762
left = -74.4756
right = -73.3976

#boundaries = gpd.read_file('data/NYcityBorders.gpkg')['geometry'][0]
#osm_to_shp(download_osm(left, bottom, right, top, highway_cat), 'Generated Shapefiles/NYC', 'NYC', boundaries_shape=boundaries)

# Load hourly speed data
DATA_DIR = 'C://Users//Yamil//Proyectos//Proyectos en Git//' \
    + 'Memoria Ambulancias//ems-ny-data//Generated Shapefiles//'

uber_dataset = pd.read_csv('C://Users//Yamil//Proyectos//' +
    'Proyectos en Git//Memoria Ambulancias//ems-ny-data//data//hourly_speeds.csv', sep=',')  # noqa: E501
graph_edges = gpd.read_file(DATA_DIR + 'NYC_edges.geojson', driver='geojson')  # noqa: E501

output = apply_speed_data(uber_dataset, graph_edges)

gdf = gpd.GeoDataFrame(output, geometry = 'geometry')
gdf.to_file(DATA_DIR + 'NYC_edges_with_speed.geojson', driver='GeoJSON')

pd.DataFrame(output.drop(['geometry', 'u','v','length','name','oneway','lanes','highway','maxspeed'], axis = 1)).to_csv('edge_speeds_.csv')