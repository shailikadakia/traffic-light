import json
import pandas as pd
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt

from analysis_utils import (
    get_graph,
    get_traffic_lights,
    cluster_intersections,
    snap_to_nearest_node,
    compute_violation_pairs,
    perform_statistical_tests,
    count_accidents_within_buffer,
    extract_flagged_intersections,
    match_and_count_accidents,
    summarize_flagged_vs_nonflagged,
    summarize_by_buffer_radius,
    stratify_by_distance_group,
    compare_flagged_groups_to_nonflagged,
    visualize_accident_heatmap,
    analyze_lighting_conditions
)

# --- Step 1: Get Graph and Traffic Light Data --- #
print("step 1")
place = "Ottawa, Ontario, Canada"
G = get_graph(place=place)
traffic_lights = get_traffic_lights(place=place)

# --- Step 2: Cluster Intersections --- #
print("step 2")
traffic_lights = cluster_intersections(traffic_lights)
traffic_lights.to_file("ottawa_traffic_lights.geojson", driver="GeoJSON")

# --- Step 3: Compute Cluster Centroids --- #
print('step 3')
cluster_centroids = (
    traffic_lights.to_crs(epsg=4326)
    .groupby("intersection_cluster")
    .geometry.apply(lambda g: g.union_all().centroid)
    .reset_index()
)
cluster_centroids["lat"] = cluster_centroids.geometry.y
cluster_centroids["lon"] = cluster_centroids.geometry.x
cluster_centroids.to_csv("cluster_centroids.csv", index=False)

# --- Step 4: Snap to Nearest Node --- #
print("step 4")
cluster_centroids = snap_to_nearest_node(cluster_centroids, G)

# --- Step 5: Compute Violation Pairs --- #
print("step 5")
violation_df, paths_by_pair = compute_violation_pairs(cluster_centroids, G)
violation_df = violation_df[violation_df["road_distance_m"] >= 50].copy()
violation_df.to_csv("filtered_route_violations_50_200m.csv", index=False)

# --- Step 6: Map Rendering --- #
print("strp 6")
traffic_lights_latlon = traffic_lights.to_crs(epsg=4326)
map_center = [cluster_centroids["lat"].mean(), cluster_centroids["lon"].mean()]
m = folium.Map(location=map_center, zoom_start=13)
for _, row in traffic_lights_latlon.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=3,
        color="black",
        fill=True,
        fill_color="blue" if row["intersection_cluster"] in violation_df[["from_cluster", "to_cluster"]].values else "gray",
        fill_opacity=0.7
    ).add_to(m)
for _, row in violation_df.iterrows():
    path = paths_by_pair.get((row["from_cluster"], row["to_cluster"]))
    if path:
        path_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]
        folium.PolyLine(path_coords, color="red", weight=3, opacity=0.7, tooltip=f"{row['road_distance_m']} m").add_to(m)
m.save("ottawa_map.html")

# --- Step 7: Extract Flagged Intersections --- #
print("step 7")
flagged_intersections = extract_flagged_intersections(violation_df, cluster_centroids)
flagged_intersections.to_csv("flagged_intersections.csv", index=False)

# --- Step 8: Accident Matching and Counting --- #
print("step 8")

with open("config.json") as f:
    config = json.load(f)
accidents = pd.read_csv(config["accident_data_path"])
non_flagged = pd.read_csv(config["non_flagged"])

# Match and count
flagged_counts, non_flagged_counts, flagged_matches, non_flagged_matches = match_and_count_accidents(
    accidents, flagged_intersections, non_flagged
)
flagged_matches.to_csv("accidents_exactly_at_intersections.csv", index=False)
non_flagged_matches.to_csv("accidents_at_non_flagged_intersections.csv", index=False)
flagged_counts.to_csv("accident_counts_per_intersection.csv", index=False)
non_flagged_counts.to_csv("accident_counts_non_flagged.csv", index=False)

# --- Step 9: Statistical Testing --- #
print("step 9")

stats = perform_statistical_tests(flagged_counts, non_flagged_counts)
print("\n📊 Statistical Test Results:")
print(stats)

# --- Step 10: Buffer Analysis (25/50/100/200m) --- #
print("step 10")

count_accidents_within_buffer(
    accidents_df=accidents,
    intersections_df=flagged_intersections,
    distances=[25, 50, 100, 200],
    output_dir=""
)

# --- Step 11: Create Summary Tables --- #
print("step 11")

summarize_flagged_vs_nonflagged(config)
summarize_by_buffer_radius(config)

# --- Step 12: Stratified Distance Group Analysis --- #
print("step 12")

stratify_by_distance_group(config)

# --- Step 13: Flagged vs Non-Flagged by Distance Group --- #
print("step 13")

compare_flagged_groups_to_nonflagged(config)

# --- Step 14: Heatmap of Accidents and Flagged Intersections --- #
print("step 14")
visualize_accident_heatmap(config)

# --- Step 15: Lighting Condition Analysis --- #
print("step 15")
analyze_lighting_conditions(config)
