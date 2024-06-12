import networkx as nx
import utm
import json
from geopy.distance import distance


def make_metro_graph(edgesfile='edges.txt', coordsfile='metroCoords.geojson'):
    """Create a NetworkX graph corresponding to the Metro de Santiago Network.

    The file `metroCoords.geojson` was obtained from OpenStreetMap data, using the
    service https://overpass-turbo.eu/, with the query

    node
    [public_transport=station]
    [station=subway]
    ({{bbox}});
    out;

    Not that you need to have the city of Santiago in the map to use it as the
    bounding box.

    We can use this to get the lines, but is not clear at the moment how to
    programatically get the stations that are connected:

    [out:json][timeout:25];
    // gather results
    way["railway"="subway"]
    ["name"="Línea 5"]
    ({{bbox}});
    // print results
    out geom;
    """

    with open(coordsfile) as f:
        data = json.load(f)

    d = {x['properties']['name']: x['geometry']['coordinates'] for
         x in data['features']}

    # Manually add missing stations
    d['Irarrázaval'] = (-70.6283197, -33.4550371)
    d['Vicente Valdés'] = (-70.5967924, -33.5264186)

    G = nx.Graph()
    ref_station = 'San Pablo'
    for station, latlong in d.items():
        x, y, _, _ = utm.from_latlon(*latlong)
        G.add_node(station, latlong=latlong,
                   distance=distance(latlong, d[ref_station]).meters,
                   x=x, y=y)

    # G.add_edge('San Pablo', 'Neptuno')
    with open(edgesfile) as f:
        for e in f.readlines():
            if e[0] == '#' or len(e) < 2:
                continue
            u, v = e.split(',')
            G.add_edge(u.strip(), v.strip())
    return G
