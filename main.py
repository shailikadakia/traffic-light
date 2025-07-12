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
import json 

print("Step 1: Getting graph")

place = "Ottawa, Ontario, Canada"
# Brampton, McVean Rd (43.804943, -79.730859)
# Downtown Ottawa (45.4215, -75.6972)
# Brampton, Ebenezer Rd (43.76963535984992, -79.66604852500859)
#coordinates = (45.4215, -75.6972)
# G = ox.graph_from_point(coordinates, dist=3000, network_type="drive")
G = ox.graph_from_place(place, network_type="drive")
nodes, _ = ox.graph_to_gdfs(G)

print("Step2")
tags = {"highway": "traffic_signals"}
#traffic_lights = ox.features_from_point(coordinates, dist=1500, tags=tags)
traffic_lights = ox.features_from_place(place, tags=tags)
traffic_lights = traffic_lights.to_crs(epsg=32617)
traffic_lights["x"] = traffic_lights.geometry.x
traffic_lights["y"] = traffic_lights.geometry.y
print("Traffic lights found:", len(traffic_lights))
traffic_lights.to_file("ottawa_traffic_lights.geojson", driver="GeoJSON")

print("Step 3")
coords = traffic_lights[["x", "y"]].values
clustering = DBSCAN(eps=20, min_samples=1).fit(coords)
traffic_lights["intersection_cluster"] = clustering.labels_

print("Step 4")
# Step 4: Compute centroids for each cluster

cluster_centroids = (
    traffic_lights.to_crs(epsg=4326)
    .groupby("intersection_cluster")
    .geometry.apply(lambda g: g.union_all().centroid)
    .reset_index()
)
cluster_centroids["lat"] = cluster_centroids.geometry.y
cluster_centroids["lon"] = cluster_centroids.geometry.x

cluster_centroids.to_csv("cluster_centroids.csv", index=False)

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
latlon_lookup = cluster_centroids.set_index("intersection_cluster")[["lat", "lon"]].to_dict("index")

cluster_ids = cluster_centroids["intersection_cluster"].tolist()
close_pairs = []
for c1, c2 in combinations(cluster_ids, 2):
    loc1 = latlon_lookup[c1]
    loc2 = latlon_lookup[c2]
    if geodesic((loc1["lat"], loc1["lon"]), (loc2["lat"], loc2["lon"])).meters < 200:
        close_pairs.append((c1, c2))

print(f"✅ Found {len(close_pairs)} pairs under 200m geodesic distance")

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
        if distance < 200:
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

# Step 6.3: Filter out very close cluster pairs (less than 50m)
filtered_route_violation_df = route_violation_df[route_violation_df["road_distance_m"] >= 50].copy()
print(f"✅ Route violations after filtering out <50m: {len(filtered_route_violation_df)}")

# Optional: Save for inspection
filtered_route_violation_df.to_csv("filtered_route_violations_50_200m.csv", index=False)

# print(f"⚠️ Total unique flagged intersections (after 50m filter): {len(flagged_intersections)}")

print("step 7")
def get_coords(cluster_id):
    row = cluster_centroids[cluster_centroids["intersection_cluster"] == cluster_id]
    return row["lat"].values[0], row["lon"].values[0]

filtered_route_violation_df[["from_lat", "from_lon"]] = filtered_route_violation_df["from_cluster"].apply(lambda x: pd.Series(get_coords(x)))
filtered_route_violation_df[["to_lat", "to_lon"]] = filtered_route_violation_df["to_cluster"].apply(lambda x: pd.Series(get_coords(x)))

########### no longer need this below section, it was to verify that the issues <50m are part of one intersection ############
#### assess only the intersections flagged as being less than 50m. This is as per the histogram showing the distribution ####
# Filter route violations for road distances < 50m
#close_violation_df = route_violation_df[route_violation_df["road_distance_m"] < 50].copy()

# Plot map of only <50m intersections
#map_center = [cluster_centroids["lat"].mean(), cluster_centroids["lon"].mean()]
#m_close = folium.Map(location=map_center, zoom_start=13)

# Add only flagged traffic light intersections in <50m pairs
#flagged_ids_close = pd.unique(
#    close_violation_df[["from_cluster", "to_cluster"]].values.ravel()
#)

#for _, row in traffic_lights_latlon.iterrows():
#    if row["intersection_cluster"] in flagged_ids_close:
#        folium.CircleMarker(
#            location=[row.geometry.y, row.geometry.x],
#            radius=3,
#            color="black",
#            fill=True,
#            fill_color="orange",
#            fill_opacity=0.8
#        ).add_to(m_close)

# Add red lines for <50m flagged intersections
# for _, row in close_violation_df.iterrows():
#    c1 = row["from_cluster"]
#    c2 = row["to_cluster"]
#    path_nodes = paths_by_pair.get((c1, c2))
#    if path_nodes:
#        path_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path_nodes]
#        folium.PolyLine(
#            path_coords,
#            color="red",
#            weight=3,
#            opacity=0.7,
#            tooltip=f"{row['road_distance_m']} m"
#        ).add_to(m_close)

# Save to HTML
#m_close.save("close_intersections_under_50m.html")
#print("✅ Saved map: close_intersections_under_50m.html")

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
        fill_color="blue" if row["intersection_cluster"] in filtered_route_violation_df["from_cluster"].values or row["intersection_cluster"] in route_violation_df["to_cluster"].values else "gray",
        fill_opacity=0.7
    ).add_to(m)

# Add violation lines
for _, row in filtered_route_violation_df.iterrows():
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
m.save("ottawa_map.html")
print("Map saved")

import os
print("Current working directory:", os.getcwd())

######### Accident data ##########
# Step 9: Extract all flagged intersections as a list of lat/lon points
print("Step 9: Extracting flagged intersection locations...")

# 1. Get unique cluster IDs from both 'from' and 'to' columns
flagged_cluster_ids = pd.unique(
    filtered_route_violation_df[["from_cluster", "to_cluster"]].values.ravel()
)

# 2. Filter the cluster_centroids for these IDs
flagged_intersections = cluster_centroids[
    cluster_centroids["intersection_cluster"].isin(flagged_cluster_ids)
].copy()

# 3. Preview and export
print("✅ Number of flagged intersections:", len(flagged_intersections))
print(flagged_intersections[["intersection_cluster", "lat", "lon"]].head())

# 4. Save for use with accident data
flagged_intersections[["intersection_cluster", "lat", "lon"]].to_csv(
    "flagged_intersections.csv", index=False
)
print("✅ Saved as 'flagged_intersections.csv'")

##### Matching flagged intersections with accident data #####
import pandas as pd

with open("config.json") as f:
    config = json.load(f)
# Load and filter only traffic-light-controlled accidents. This is after the realization that the dataset has a category of what the intersection has: stop sign, traffic light, etc #
# accidents = pd.read_csv("C:/Users/Shivani Kadakia/Documents/traffic-light-project/traffic-light/Ottawa_traffic_collision_data.csv")
accidents = pd.read_csv(config["accident_data_path"])
# Filter to only accidents at traffic-light-controlled intersections
accidents = accidents[accidents["Traffic_Control"] == "01 - Traffic signal"].copy()

print("✅ Accident records at traffic-signal-controlled intersections:", len(accidents))

# Load both datasets
flagged = pd.read_csv(config["flagged"])

# Round both to 7 decimal places for matching precision
flagged["lat_rounded"] = flagged["lat"].round(7)
flagged["lon_rounded"] = flagged["lon"].round(7)

accidents["lat_rounded"] = accidents["Lat"].round(7)
accidents["lon_rounded"] = accidents["Long"].round(7)

# Merge on rounded lat/lon
exact_matches = pd.merge(
    accidents,
    flagged,
    left_on=["lat_rounded", "lon_rounded"],
    right_on=["lat_rounded", "lon_rounded"],
    how="inner"
)

# Output matched accidents
print(f"✅ Exact matches found: {len(exact_matches)}")
exact_matches.to_csv("accidents_exactly_at_intersections.csv", index=False)

# Count accidents at each intersection
accident_counts = (
    exact_matches.groupby(["lat", "lon"])
    .size()
    .reset_index(name="accident_count")
    .sort_values("accident_count", ascending=False)
)

# Preview the result
print(accident_counts)

# Save to CSV for review
accident_counts.to_csv("accident_counts_per_intersection.csv", index=False)

##### Matching non-flagged intersections with accident data #####
import pandas as pd

# Load data
accidents = pd.read_csv(config["accident_data_path"])
non_flagged = pd.read_csv(config["non_flagged"])

# filter accident data to be only traffic light accidents
accidents.columns = accidents.columns.str.strip()
accidents["Traffic_Control"] = accidents["Traffic_Control"].astype(str).str.strip()
accidents = accidents[accidents["Traffic_Control"] == "01 - Traffic signal"].copy()

# Round to 7 decimal places
non_flagged["lat_rounded"] = non_flagged["lat"].round(7)
non_flagged["lon_rounded"] = non_flagged["lon"].round(7)

accidents["lat_rounded"] = accidents["Lat"].round(7)
accidents["lon_rounded"] = accidents["Long"].round(7)

# Match based on coordinates
non_flagged_matches = pd.merge(
    accidents,
    non_flagged,
    left_on=["lat_rounded", "lon_rounded"],
    right_on=["lat_rounded", "lon_rounded"],
    how="inner"
)

# Save matched results
print(f"✅ Exact matches at non-flagged intersections: {len(non_flagged_matches)}")
non_flagged_matches.to_csv("accidents_at_non_flagged_intersections.csv", index=False)

# Group and count accidents per intersection
non_flagged_accident_counts = (
    non_flagged_matches.groupby("intersection_cluster")
    .size()
    .reset_index(name="accident_count")
    .sort_values("accident_count", ascending=False)
)

# Save summary
non_flagged_accident_counts.to_csv("accident_counts_non_flagged.csv", index=False)
print("✅ Accident counts per non-flagged intersection saved.")

###### Finding 25m, 50m, and 100m from the impacted intersections less than 200m to see how many accidents there are ######

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Load data
accidents_df = pd.read_csv(config["accident_data_path"])
accidents_df.columns = accidents_df.columns.str.strip()
accidents_df["Traffic_Control"] = accidents_df["Traffic_Control"].astype(str).str.strip()
accidents_df = accidents_df[accidents_df["Traffic_Control"] == "01 - Traffic signal"].copy()

accident_gdf = gpd.GeoDataFrame(
    accidents_df,
    geometry=gpd.points_from_xy(accidents_df["Long"], accidents_df["Lat"]),
    crs="EPSG:4326"
)

intersections_df = pd.read_csv(config["flagged"])

intersections_gdf = gpd.GeoDataFrame(
    intersections_df,
    geometry=gpd.points_from_xy(intersections_df["lon"],intersections_df["lat"]),
    crs="EPSG:4326"
)

# Reproject to meters
accident_gdf = accident_gdf.to_crs(epsg=32617)
intersections_gdf = intersections_gdf.to_crs(epsg=32617)

# Loop over buffer distances
for distance in [25, 50, 100, 200]:
    print(f"\n🔎 Checking accidents within {distance} meters...")

    # Create buffer zones
    intersections_gdf["buffer"] = intersections_gdf.geometry.buffer(distance)

    # Use buffer geometry for join
    buffered = intersections_gdf.set_geometry("buffer")

    # Spatial join: find all accidents within the buffer
    joined = gpd.sjoin(accident_gdf, buffered, predicate="within", how="inner")

    # Add this here
    print(f"\n📏 Distance: {distance} meters")
    print("Total accident records matched:", len(joined))
    print("Unique accident locations matched:", joined[["Lat", "Long"]].drop_duplicates().shape[0])

    # Optional: add actual distance to cluster
    joined["distance_m"] = joined.geometry.distance(joined["geometry_right"])

    # Save full accident-join match data
    joined.to_csv(f"accidents_within_{distance}m_detailed.csv", index=False)

    # Group by intersection to count total accidents per cluster
    accident_counts = joined.groupby("intersection_cluster").size().reset_index(name="accident_count")

    # Merge with flagged intersections to keep all, even those with 0 accidents
    result = intersections_df.merge(accident_counts, on="intersection_cluster", how="left").fillna(0)
    result["accident_count"] = result["accident_count"].astype(int)

    # Save summary
    result.to_csv(f"accidents_within_{distance}m_summary.csv", index=False)
    print(f"✅ Saved: {len(joined)} accident matches and {len(result)} intersection summaries for {distance}m")


###### Capture all intersections which don't have an accident within 25, 50, 100, 200m ######
import pandas as pd

# Load the flagged intersections list
all_intersections = pd.read_csv(config["accident_data_path"])

# Distances to analyze
distances = [25, 50, 100, 200]

for d in distances:
    print(f"\n🔎 Finding intersections with 0 accidents within {d}m...")
    
    # Load corresponding summary
    summary_dir = config["summary_dir"]
    summary_filename = f"accidents_within_{d}m_summary.csv"
    summary_path = os.path.join(summary_dir, summary_filename)

    summary = pd.read_csv(summary_path)
    # Filter intersections with zero accidents
    zero_accidents = summary[summary["accident_count"] == 0]

    print(f"🚫 {len(zero_accidents)} intersections had zero accidents within {d}m")

    # Save
    zero_accidents.to_csv(f"zero_accidents_within_{d}m.csv", index=False)

###### time for statistics - compare intersections with accidents vs without ######

import pandas as pd

# Load all intersection clusters from original project
all_clusters = pd.read_csv(config["cluster_centroids"])

# Load flagged clusters (from <200m traffic light pairs)
flagged = pd.read_csv(config["flagged"])
flagged_ids = flagged["intersection_cluster"].unique()

# Use this if you haven't already
all_clusters = cluster_centroids.copy()

# flagged_ids should be a list or array of IDs from the <200m group
flagged_ids = pd.unique(filtered_route_violation_df[["from_cluster", "to_cluster"]].values.ravel())


# Filter out flagged clusters
non_flagged = all_clusters[~all_clusters["intersection_cluster"].isin(flagged_ids)]

# Save result
non_flagged.to_csv("non_flagged_intersections.csv", index=False)
print(f"✅ Saved {len(non_flagged)} non-flagged intersections.")

print("Total intersection clusters:", cluster_centroids["intersection_cluster"].nunique())
print("Flagged clusters:", len(flagged_ids))

import matplotlib.pyplot as plt

filtered_route_violation_df["road_distance_m"].hist(bins=20)
plt.title("Distance distribution of flagged intersections")
plt.xlabel("Distance (m)")
plt.ylabel("Count")
plt.show()


####### Before statistics, let's do a summary table of the tables we have run for accident data and flagged/non-flagged intersections #####
import pandas as pd

# Load required datasets
flagged = pd.read_csv(config["flagged"])
non_flagged = pd.read_csv(config["non_flagged"])
exact_matches_flagged = pd.read_csv(config["exact"])
exact_matches_non_flagged = pd.read_csv(config["exact_non_flagged"])

### --- Table 1: Flagged vs Non-Flagged Intersections (Exact Match) --- ###

# Count total intersections
total_flagged = flagged["intersection_cluster"].nunique()
total_non_flagged = non_flagged["intersection_cluster"].nunique()

# Flagged accident counts
flagged_exact = exact_matches_flagged.groupby("intersection_cluster").size().reset_index(name="accident_count")
flagged_with_accident = flagged_exact["intersection_cluster"].nunique()
total_flagged_accidents = flagged_exact["accident_count"].sum()

# Non-flagged accident counts
non_flagged_exact = exact_matches_non_flagged.groupby("intersection_cluster").size().reset_index(name="accident_count")
non_flagged_with_accident = non_flagged_exact["intersection_cluster"].nunique()
total_non_flagged_accidents = non_flagged_exact["accident_count"].sum()

# Assemble summary table
table1 = pd.DataFrame({
    "Group": ["Flagged", "Non-Flagged"],
    "# Intersections": [total_flagged, total_non_flagged],
    "# With ≥1 Accident (Exact)": [flagged_with_accident, non_flagged_with_accident],
    "Total Accidents (Exact)": [total_flagged_accidents, total_non_flagged_accidents]
})

print("\n📊 Table 1: Flagged vs Non-Flagged Intersections (Exact Match)\n")
print(table1.to_string(index=False))

### --- Table 2: Accidents Near Flagged Intersections by Distance --- ###

distances = [25, 50, 100, 200]
rows = []
summary_dir = config["summary_dir"]

for d in distances:
    summary_dir = config["summary_dir"]
    summary_filename = f"accidents_within_{d}m_summary.csv"
    summary_path = os.path.join(summary_dir, summary_filename)


    summary = pd.read_csv(summary_path)
    count_with_accident = (summary["accident_count"] > 0).sum()
    total_accidents = summary["accident_count"].sum()
    rows.append({
        "Buffer Radius (m)": d,
        "# Intersections With ≥1 Accident": count_with_accident,
        "Total Accidents": total_accidents
    })

table2 = pd.DataFrame(rows)

print("\n📊 Table 2: Accidents Within Buffer Distance from Flagged Intersections\n")
print(table2.to_string(index=False))

# Export both tables if needed
table1.to_csv("summary_table_flagged_vs_non_flagged.csv", index=False)
table2.to_csv("summary_table_flagged_buffer_distances.csv", index=False)

##### actual time for statistics #####
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

### ---- Load Data ---- ###
flagged = pd.read_csv(config["flagged"])
non_flagged = pd.read_csv(config["non_flagged"])
flagged_counts = pd.read_csv(config["flagged_counts"])
non_flagged_counts = pd.read_csv(config["non_flagged_counts"])

### ---- Prepare for Chi-Square ---- ###
flagged_with_accident = flagged_counts.shape[0]
non_flagged_with_accident = non_flagged_counts.shape[0]

total_flagged = flagged.shape[0]
total_non_flagged = non_flagged.shape[0]

flagged_zero_accidents = total_flagged - flagged_with_accident
non_flagged_zero_accidents = total_non_flagged - non_flagged_with_accident

contingency_table = [
    [flagged_with_accident, flagged_zero_accidents],
    [non_flagged_with_accident, non_flagged_zero_accidents]
]

print("\n📊 Contingency Table (Flagged vs Non-Flagged, Accident vs No Accident):")
print(pd.DataFrame(contingency_table, columns=["≥1 Accident", "0 Accidents"], index=["Flagged", "Non-Flagged"]))

chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)
print(f"\n🔬 Chi-Square Test Results:")
print(f"Chi2 statistic = {chi2:.4f}")
print(f"p-value = {p_chi2:.4f}")
print(f"Degrees of freedom = {dof}")
print("Expected frequencies:\n", expected)

### ---- Mann-Whitney U Test ---- ###
# Merge on (lat, lon) since intersection_cluster is not in accident_counts CSVs

# Load intersection and accident data
flagged = pd.read_csv(config["flagged"])
non_flagged = pd.read_csv(config["non_flagged"])
flagged_counts = pd.read_csv(config["flagged_counts"])
non_flagged_counts = pd.read_csv(config["non_flagged_counts"])

### Merge flagged on lat/lon ###
flagged_full = flagged.merge(
    flagged_counts, on=["lat", "lon"], how="left"
).fillna(0)

### Merge non-flagged on intersection_cluster ###
non_flagged_full = non_flagged.merge(
    non_flagged_counts, on="intersection_cluster", how="left"
).fillna(0)

### Run Mann-Whitney U Test ###
flagged_accidents = flagged_full["accident_count"]
non_flagged_accidents = non_flagged_full["accident_count"]

stat, p_mannwhitney = mannwhitneyu(
    flagged_accidents, non_flagged_accidents, alternative="two-sided"
)

print("\n📈 Mann-Whitney U Test Results:")
print(f"U statistic = {stat:.4f}")
print(f"p-value = {p_mannwhitney:.4f}")

##### tests above are not statistically significant #####
##### let's try stratification by distances of the flagged intersections #####

import pandas as pd

# Load data
violation_df = pd.read_csv(config["violation_df"])

# Define distance bins and labels
bins = [50, 100, 150, 200]
labels = [f"{bins[i]}–{bins[i+1]}m" for i in range(len(bins)-1)]

# Bin the road distances
violation_df["distance_group"] = pd.cut(
    violation_df["road_distance_m"], bins=bins, labels=labels, right=False
).astype(str)

# Combine from and to clusters
from_pairs = violation_df[["from_cluster", "distance_group"]].rename(columns={"from_cluster": "intersection_cluster"})
to_pairs = violation_df[["to_cluster", "distance_group"]].rename(columns={"to_cluster": "intersection_cluster"})
all_pairs = pd.concat([from_pairs, to_pairs])

# Rank groups: smaller distances = higher priority
group_order = {label: i for i, label in enumerate(labels)}  # 50–100m = 0, 100–150m = 1, etc.
all_pairs["group_rank"] = all_pairs["distance_group"].map(group_order)

# Keep only the closest group for each intersection
closest_group = (
    all_pairs.sort_values("group_rank")  # prioritize smaller distances
    .drop_duplicates(subset="intersection_cluster", keep="first")  # keep only best match
    .drop(columns="group_rank")
)

# Sanity check
print("\n✅ Assigned closest distance group per intersection:")
print(closest_group["distance_group"].value_counts())

# Load intersection coordinates and accident counts
intersections = pd.read_csv(config["flagged"])
accidents = pd.read_csv(config["flagged_counts"])

# Merge to assign each intersection its closest distance group
intersections_grouped = intersections.merge(
    closest_group, on="intersection_cluster", how="inner"
)

# Merge with accident count using lat/lon
full = intersections_grouped.merge(
    accidents, on=["lat", "lon"], how="left"
)

# Fill missing accident counts with 0
full["accident_count"] = full["accident_count"].fillna(0).astype(int)

print("✅ Rebuilt flagged intersection dataset with closest distance group.")
print(full.head())

full.to_csv("flagged_intersections_with_closest_distance_group.csv", index=False)
print("✅ Saved 'flagged_intersections_with_closest_distance_group.csv'")

# Group by distance bucket and compute only the three desired metrics
summary = (
    full.groupby("distance_group")["accident_count"]
    .agg([
        ("# Intersections", "count"),
        ("Total Accidents", "sum"),
        ("% With ≥1 Accident", lambda x: (x > 0).mean() * 100)
    ])
    .round({"% With ≥1 Accident": 1})
    .reset_index()
)

print("\n📊 Final Summary Table by Distance Group:\n")
print(summary)

summary.to_csv("accident_summary_by_distance_group.csv", index=False)
print("✅ Saved as 'accident_summary_by_distance_group.csv'")

##### Pairwise Mann-Whitney U Tests - this compares if there is statistical significance between each bucket (ex. more accidents in shorter distance?) #####

from scipy.stats import mannwhitneyu
from itertools import combinations
import pandas as pd

# Ensure the DataFrame is clean
grouped = full[["distance_group", "accident_count"]].dropna()

# Get unique distance group labels
groups = grouped["distance_group"].unique()

# Prepare results list
results = []

# Compare each pair of distance groups
for g1, g2 in combinations(groups, 2):
    x = grouped[grouped["distance_group"] == g1]["accident_count"]
    y = grouped[grouped["distance_group"] == g2]["accident_count"]
    
    stat, p = mannwhitneyu(x, y, alternative="two-sided")
    
    results.append({
        "Group 1": g1,
        "Group 2": g2,
        "U Statistic": stat,
        "p-value": p,
        "Group 1 Median": x.median(),
        "Group 2 Median": y.median(),
        "n1": len(x),
        "n2": len(y)
    })

# Create DataFrame from results
pairwise_results = pd.DataFrame(results)

# Optional: apply Bonferroni correction for multiple comparisons
pairwise_results["Bonferroni-corrected p"] = pairwise_results["p-value"] * len(pairwise_results)
pairwise_results["Significant (p<0.05 after correction)"] = pairwise_results["Bonferroni-corrected p"] < 0.05

# Display
print("\n📈 Pairwise Mann-Whitney U Test Results Between Distance Groups:\n")
print(pairwise_results.sort_values("p-value").round(4))


##### would like to check each bucket with the non-flagged intersections #####

# Load data
non_flagged = pd.read_csv(config["non_flagged"])
non_flagged_counts = pd.read_csv(config["non_flagged_counts"])

# Ensure clean column names
non_flagged.columns = non_flagged.columns.str.strip()
non_flagged_counts.columns = non_flagged_counts.columns.str.strip()

# Merge to include accident counts (fill missing values with 0)
non_flagged_full = non_flagged.merge(
    non_flagged_counts, on="intersection_cluster", how="left"
).fillna(0)

# Make sure accident count is integer
non_flagged_full["accident_count"] = non_flagged_full["accident_count"].astype(int)

print("✅ Non-flagged intersections prepared:", len(non_flagged_full))

full = pd.read_csv(config["full"])

##### Run Mann-Whitney for flagged intersection buckets and non-flagged #####
from scipy.stats import mannwhitneyu

# Get unique distance groups
distance_groups = full["distance_group"].unique()

results = []

for group in distance_groups:
    flagged_group = full[full["distance_group"] == group]["accident_count"]
    non_flagged_group = non_flagged_full["accident_count"]
    
    stat, p = mannwhitneyu(flagged_group, non_flagged_group, alternative="two-sided")
    
    results.append({
        "Flagged Distance Group": group,
        "U Statistic": stat,
        "p-value": p,
        "Flagged Median": flagged_group.median(),
        "Non-Flagged Median": non_flagged_group.median(),
        "Flagged n": len(flagged_group),
        "Non-Flagged n": len(non_flagged_group)
    })

results_df = pd.DataFrame(results).sort_values("p-value")

print("\n📈 Mann-Whitney U Tests: Flagged Distance Groups vs Non-Flagged\n")
print(results_df.round(4))

results_df.to_csv("flagged_intersection_vs_non_flagged_intersection_MannWhitneyU.csv", index=False)
print("✅ Saved 'flagged_intersection_vs_non_flagged_intersection_MannWhitneyU.csv'")


##### The test above was not statistically significant. Let's make a heat map to visual accidents with flagged intersections #####

import pandas as pd

# Load all accident data (with coordinates)
accidents = pd.read_csv(config="accident_data_path")

# Clean up column names
accidents.columns = accidents.columns.str.strip()

# Ensure latitude and longitude columns are named correctly
accidents["lat"] = accidents["Lat"].round(6)
accidents["lon"] = accidents["Long"].round(6)

# Drop rows with missing coordinates (if any)
accidents = accidents.dropna(subset=["lat", "lon"])
print("✅ Accident records:", len(accidents))

# Load the 'full' DataFrame with intersection coords and distance group
flagged = full.copy()  # from earlier step
print("✅ Flagged intersections:", len(flagged))

# Heat map
import folium
from folium.plugins import HeatMap
import pandas as pd

# Load and filter
accidents = pd.read_csv(config["accident_data_path"])
accidents.columns = accidents.columns.str.strip()
accidents["Traffic_Control"] = accidents["Traffic_Control"].astype(str).str.strip()
accidents = accidents[accidents["Traffic_Control"] == "01 - Traffic signal"].copy()

# Prepare lat/lon columns
accidents["lat"] = accidents["Lat"].round(6)
accidents["lon"] = accidents["Long"].round(6)
accidents = accidents.dropna(subset=["lat", "lon"])  # optional safety check

# Base map centered on Ottawa
map_center = [accidents["lat"].mean(), accidents["lon"].mean()]
m = folium.Map(location=map_center, zoom_start=12)

# Add accident heatmap layer
heat_data = accidents[["lat", "lon"]].values.tolist()

HeatMap(
    heat_data,
    radius=12,
    blur=8,
    max_zoom=13,
    min_opacity=0.3
).add_to(m)

# Load flagged intersections (if not already loaded)
# flagged = pd.read_csv("flagged_intersections_with_closest_distance_group.csv")

# Add flagged intersections
color_map = {
    "50–100m": "red",
    "100–150m": "orange",
    "150–200m": "purple"
}

for _, row in flagged.iterrows():
    group = row["distance_group"]
    color = color_map.get(group, "gray")
    
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        popup=f"{group}, Accidents: {row['accident_count']}"
    ).add_to(m)

# Save
m.save("ottawa_accident_heatmap_with_flagged_intersections.html")
print("✅ Saved map as 'ottawa_accident_heatmap_with_flagged_intersections.html'")

##### any impacts with the light conditions #####
accidents = pd.read_csv(config["accident_data_path"])
accidents.columns = accidents.columns.str.strip()
accidents["Traffic_Control"] = accidents["Traffic_Control"].astype(str).str.strip()

flagged = pd.read_csv(config["flagged"])

# Filter only traffic-signal-controlled accidents
accidents = accidents[accidents["Traffic_Control"] == "01 - Traffic signal"].copy()

# Inspect available columns (optional)
print(accidents.columns.tolist())
print(accidents["Light"].value_counts())

# Match flagged intersections to accidents and extract lighting conditions
# Round coordinates for join
accidents["lat_rounded"] = accidents["Lat"].round(7)
accidents["lon_rounded"] = accidents["Long"].round(7)

flagged["lat_rounded"] = flagged["lat"].round(7)
flagged["lon_rounded"] = flagged["lon"].round(7)

# Merge to get only accidents that occurred at flagged intersections
flagged_light_matches = pd.merge(
    accidents,
    flagged,
    on=["lat_rounded", "lon_rounded"],
    how="inner"
)

# Check lighting breakdown
print("\n💡 Lighting condition counts at flagged intersections:")
print(flagged_light_matches["Light"].value_counts())

# Step 3: Compare to non-flagged intersections

non_flagged = pd.read_csv(config["non_flagged"])

non_flagged["lat_rounded"] = non_flagged["lat"].round(7)
non_flagged["lon_rounded"] = non_flagged["lon"].round(7)

non_flagged_light_matches = pd.merge(
    accidents,
    non_flagged,
    on=["lat_rounded", "lon_rounded"],
    how="inner"
)

print("\n💡 Lighting condition counts at non-flagged intersections:")
print(non_flagged_light_matches["Light"].value_counts())

# visualization

import matplotlib.pyplot as plt

# Count lighting condition breakdown
flagged_counts = flagged_light_matches["Light"].value_counts().rename("Flagged")
non_flagged_counts = non_flagged_light_matches["Light"].value_counts().rename("Non-Flagged")

# Combine into one DataFrame
light_df = pd.concat([flagged_counts, non_flagged_counts], axis=1).fillna(0)

# Plot
light_df.plot(kind="bar", figsize=(10, 5))
plt.title("Lighting Condition of Accidents at Intersections")
plt.ylabel("Accident Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Chi square test
from scipy.stats import chi2_contingency

# Reformat for test
chi_data = light_df.astype(int)
chi2, p, dof, expected = chi2_contingency(chi_data)

print("\n📊 Chi-Square Test on Lighting Condition Distribution:")
print(f"Chi2 statistic = {chi2:.4f}")
print(f"p-value = {p:.4f}")
print(f"Degrees of freedom = {dof}")
