# Required imports
import osmnx as ox # for analysing OpenStreet data
import geopandas as gpd # geospatial data analysis
import pandas as pd
import networkx as nx # graph theory
from shapely.geometry import Point
from itertools import combinations # get all pairs of items
import folium # iteractive maps
from sklearn.cluster import DBSCAN # Clustering algorithm
from geopy.distance import geodesic # fast geographic distance

def run_violation_analysis(city, radius, latitude, longtitude, violating_distance):
    print("Step 1: Getting graph")

    ''' 
    Using osmnx, a road map is made up of nodes (intersections) and edges (roads between nodes)
    The goal is to connect the intersection clusters to this real map G 

    Step 1: Set location and download road graph
    graph_from_point sets a 1.5km radius road network
    G is a graph of drivable streets
    graph_to_gdfs converts graphs nodes (intersections) into a GeoDataFrame
    '''

    place = "Brampton, Ontario, Canada"
    # Brampton, McVean Rd (43.804943, -79.730859)
    # Downtown Ottawa (45.4215, -75.6972)
    # Brampton, Ebenezer Rd (43.76963535984992, -79.66604852500859)
    coordinates = (latitude , longtitude)
    G = ox.graph_from_point(coordinates, dist=radius, network_type="drive")
    nodes, _ = ox.graph_to_gdfs(G)

    print("Step2")

    '''
    Step 2: Get traffic light features
    Query OpenStreet to find all points tagged as traffic lights 
    Convert traffic lights to UTMP projection (meters)
    Create new x and y columns. Extract longitude and latitude from geometry column
    DBSCAN doesn't understand geospatial objects but it does understand numerical coordinates 
    '''
    tags = {"highway": "traffic_signals"}
    traffic_lights = ox.features_from_point(coordinates, dist=radius, tags=tags)
    traffic_lights = traffic_lights.to_crs(epsg=32617)
    traffic_lights["x"] = traffic_lights.geometry.x
    traffic_lights["y"] = traffic_lights.geometry.y

    print("Step 3")

    '''
    # Step 3: Cluster traffic lights into intersection clusters
    Cluster traffic lights that are within 20 meters of each other
    Each cluster is assumed to represent one intersection
    '''
    coords = traffic_lights[["x", "y"]].values
    clustering = DBSCAN(eps=20, min_samples=1).fit(coords)
    traffic_lights["intersection_cluster"] = clustering.labels_

    print("Step 4")
    # Step 4: Compute centroids for each cluster

    '''
    Find the center point of each traffic light cluster
    This represents the entire intersection by a single location
    EPSG=4326 switches latitude and longitude 
    '''
    cluster_centroids = (
        traffic_lights.to_crs(epsg=4326)
        .groupby("intersection_cluster")
        .geometry.apply(lambda g: g.union_all().centroid)
        .reset_index()
    )
    '''
    Add longitude and latitude columns
    '''
    cluster_centroids["lat"] = cluster_centroids.geometry.y
    cluster_centroids["lon"] = cluster_centroids.geometry.x

    print("Snapping centroids to nearest graph nodes...")


    print("Step 5")
    # Step 5: Snap each cluster to the nearest graph node
    '''
    Given the lat and lon of an intersection cluster, find the nearest road intersection in G  
    Log if any centroids cannot be matched
    '''
    def safe_nearest_node(row):
        try:
            return ox.distance.nearest_nodes(G, row["lon"], row["lat"])
        except Exception as e:
            print(f"❌ Failed for cluster {row['intersection_cluster']}: {e}")
            return None

    cluster_centroids["nearest_node"] = cluster_centroids.apply(safe_nearest_node, axis=1)

    failed = cluster_centroids[cluster_centroids["nearest_node"].isna()]
    print(f"❌ Failed to snap {len(failed)} cluster(s):")
    print(failed)

    print("✅ Cluster centroids calculated.")
    print("Number of clusters:", len(cluster_centroids))
    print(cluster_centroids.head())


    print("Step 6")
    # Step 6: Compute route distances between cluster pairs
    """ cluster_pairs = list(combinations(cluster_centroids["intersection_cluster"], 2))
    node_lookup = dict(zip(cluster_centroids["intersection_cluster"], cluster_centroids["nearest_node"]))

    valid_route_violations = []
    for c1, c2 in cluster_pairs:
        try:
            node1 = node_lookup[c1]
            node2 = node_lookup[c2]
            if node1 == node2:
                continue
            distance = nx.shortest_path_length(G, node1, node2, weight="length")
            if distance < 200:
                valid_route_violations.append({
                    "from_cluster": c1,
                    "to_cluster": c2,
                    "road_distance_m": round(distance, 2)
                })
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    route_violation_df = pd.DataFrame(valid_route_violations)
    print("\u2705 Total route-based violating cluster pairs:", len(route_violation_df)) """

    '''
    # Step 6.1: Filter pairs using fast geodesic check 
    Determine if these two intersections are physically close in a straight line
    If yes, continue to the next step
    If no, don't check road distance
    '''
    latlon_lookup = cluster_centroids.set_index("intersection_cluster")[["lat", "lon"]].to_dict("index")

    cluster_ids = cluster_centroids["intersection_cluster"].tolist()
    close_pairs = []
    for c1, c2 in combinations(cluster_ids, 2):
        loc1 = latlon_lookup[c1]
        loc2 = latlon_lookup[c2]
        if geodesic((loc1["lat"], loc1["lon"]), (loc2["lat"], loc2["lon"])).meters < violating_distance:
            close_pairs.append((c1, c2))

    print(f"✅ Found {len(close_pairs)} pairs under 300m geodesic distance")

    '''
    Step 6.2: Compute road distances on filtered pairs
    For all geodesically-close pairs, compute actual road distance
    Using NetworkX and G, find the shortest drivable path between two nodes and add all the road segments 
    '''
    node_lookup = dict(zip(cluster_centroids["intersection_cluster"], cluster_centroids["nearest_node"]))
    valid_route_violations = []
    paths_by_pair = {}

    for c1, c2 in close_pairs:
        try:
            node1 = node_lookup[c1]
            node2 = node_lookup[c2]
            if node1 == node2:
                continue
            distance = nx.shortest_path_length(G, node1, node2, weight="length")
            if distance < violating_distance:
                valid_route_violations.append({
                    "from_cluster": c1,
                    "to_cluster": c2,
                    "road_distance_m": round(distance, 2)
                })
                path = nx.shortest_path(G, node1, node2, weight="length")
                paths_by_pair[(c1, c2)] = path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    route_violation_df = pd.DataFrame(valid_route_violations)
    print("✅ Finished route distance checks.")
    print("🚨 Total route-based violating cluster pairs:", len(route_violation_df))

    print("stp 7")
    '''
    Step 7: Add coordinates for mapping
    Add lat and long coordinates so we can then draw lines on a map
    '''
    def get_coords(cluster_id):
        row = cluster_centroids[cluster_centroids["intersection_cluster"] == cluster_id]
        return row["lat"].values[0], row["lon"].values[0]

    route_violation_df[["from_lat", "from_lon"]] = route_violation_df["from_cluster"].apply(lambda x: pd.Series(get_coords(x)))
    route_violation_df[["to_lat", "to_lon"]] = route_violation_df["to_cluster"].apply(lambda x: pd.Series(get_coords(x)))

    print("step 8")
    # Step 8: Plot the results on a folium map
    map_center = [cluster_centroids["lat"].mean(), cluster_centroids["lon"].mean()] # find the center of the data on the map 
    m = folium.Map(location=map_center, zoom_start=13)

    # Add traffic lights
    traffic_lights_latlon = traffic_lights.to_crs(epsg=4326)
    for _, row in traffic_lights_latlon.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=3,
            color="black",
            fill=True,
            fill_color="blue" if row["intersection_cluster"] in route_violation_df["from_cluster"].values or row["intersection_cluster"] in route_violation_df["to_cluster"].values else "gray",
            fill_opacity=0.7
        ).add_to(m)

    # Add violation lines
    for _, row in route_violation_df.iterrows():
        c1 = row["from_cluster"]
        c2 = row["to_cluster"]
        path_nodes = paths_by_pair.get((c1, c2))
        if path_nodes:
            path_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path_nodes]
            folium.PolyLine(
                path_coords,
                color="red",
                weight=3,
                opacity=0.7,
                tooltip=f"{row['road_distance_m']} m"
            ).add_to(m)
    # Save map
    m.save(city + ".html")
    print("Map saved")
