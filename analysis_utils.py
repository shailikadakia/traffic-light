import osmnx as ox 
import geopandas as gpd 
import pandas as pd
import networkx as nx 
from shapely.geometry import Point
from itertools import combinations 
from sklearn.cluster import DBSCAN 
from geopy.distance import geodesic
from scipy.stats import chi2_contingency, mannwhitneyu
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import os
from geopy.geocoders import Nominatim

from geopy.geocoders import Nominatim

def get_place_from_point(lat, lon):
    try:
        geolocator = Nominatim(user_agent="traffic_light_app")
        location = geolocator.reverse((lat, lon), exactly_one=True)
        if location and location.raw and "address" in location.raw:
            address = location.raw["address"]
            city = address.get("city") or address.get("town") or address.get("village") or address.get("hamlet")
            state = address.get("state")
            country = address.get("country")
            if city and state and country:
                return f"{city}, {state}, {country}"
    except Exception as e:
        print(f"Reverse geocoding failed: {e}")
    return None



# Fetch traffic light locations using OSM tags.
def get_graph(place=None, point=None, dist=1500):
    if place:
        G = ox.graph_from_place(place, network_type="drive")
    elif point:
        G = ox.graph_from_point(point, dist=dist, network_type="drive")
    else:
        raise ValueError("You must provide either 'place' or 'point'")
    return G

# Cluster traffic lights into intersection groups (DBSCAN).
def get_traffic_lights(place=None, point=None, dist=1500):
    tags = {"highway": "traffic_signals"}
    if place:
        tl = ox.features_from_place(place, tags=tags)
    elif point:
        tl = ox.features_from_point(point, dist=dist, tags=tags)
    else:
        raise ValueError("You must provide a place")
    return tl.to_crs(epsg=32617)

# Snap centroids to each cluster to nearest graph node
def cluster_intersections(traffic_lights):
  traffic_lights["x"] = traffic_lights.geometry.x
  traffic_lights["y"] = traffic_lights.geometry.y
  coords = traffic_lights[["x", "y"]].values
  clustering = DBSCAN(eps=20, min_samples=1).fit(coords)
  traffic_lights["intersection_cluster"] = clustering.labels_
  return traffic_lights

def snap_to_nearest_node(cluster_centroids, G):
  def safe_nearest_node(row):
      try:
          return ox.distance.nearest_nodes(G, row["lon"], row["lat"])
      except:
          return None
  cluster_centroids["nearest_node"] = cluster_centroids.apply(safe_nearest_node, axis=1)
  return cluster_centroids

# Return all cluster pairs within geodesic and road distance threshold 
def compute_violation_pairs(cluster_centroids, G, threshold=200):
    latlon_lookup = cluster_centroids.set_index("intersection_cluster")[["lat", "lon"]].to_dict("index")
    cluster_ids = cluster_centroids["intersection_cluster"].tolist()
    close_pairs = [
        (c1, c2) for c1, c2 in combinations(cluster_ids, 2)
        if geodesic((latlon_lookup[c1]["lat"], latlon_lookup[c1]["lon"]),
                    (latlon_lookup[c2]["lat"], latlon_lookup[c2]["lon"])).meters < threshold
    ]

    node_lookup = dict(zip(cluster_centroids["intersection_cluster"], cluster_centroids["nearest_node"]))
    violations, paths = [], {}
    for c1, c2 in close_pairs:
        try:
            n1, n2 = node_lookup[c1], node_lookup[c2]
            if n1 == n2: continue
            dist = nx.shortest_path_length(G, n1, n2, weight="length")
            if dist < threshold:
                violations.append({"from_cluster": c1, "to_cluster": c2, "road_distance_m": round(dist, 2)})
                paths[(c1, c2)] = nx.shortest_path(G, n1, n2, weight="length")
        except:
            continue
    return pd.DataFrame(violations), paths

# Run Chi-Square andd Mann-Whitney U Test
def perform_statistical_tests(flagged_counts, non_flagged_counts):
    a = flagged_counts["accident_count"]
    b = non_flagged_counts["accident_count"]
    u_stat, p_u = mannwhitneyu(a, b, alternative="two-sided")

    f_with, f_zero = (a > 0).sum(), (a == 0).sum()
    n_with, n_zero = (b > 0).sum(), (b == 0).sum()

    if f_with == 0 or f_zero == 0 or n_with == 0 or n_zero == 0:
        chi2 = p_chi2 = dof = exp = None
        print("⚠️ Chi-square test skipped: expected frequency is zero in at least one cell.")
    else:
        chi2, p_chi2, dof, exp = chi2_contingency([[f_with, f_zero], [n_with, n_zero]])

    return {
        "mannwhitney": {"U": u_stat, "p": p_u},
        "chi2": {"stat": chi2, "p": p_chi2, "dof": dof, "expected": exp.tolist() if exp is not None else None}
    }

    return {
        "mannwhitney": {"stat": u_stat, "p": p_u},
        "chi2": {"stat": chi2, "p": p_chi2, "expected": exp}
    }


def count_accidents_within_buffer(accidents_df, intersections_df, distances, epsg=32617, output_dir=""):
    accident_gdf = gpd.GeoDataFrame(
        accidents_df,
        geometry=gpd.points_from_xy(accidents_df["Long"], accidents_df["Lat"]),
        crs="EPSG:4326"
    ).to_crs(epsg=epsg)

    intersections_gdf = gpd.GeoDataFrame(
        intersections_df,
        geometry=gpd.points_from_xy(intersections_df["lon"], intersections_df["lat"]),
        crs="EPSG:4326"
    ).to_crs(epsg=epsg)

    for distance in distances:
        print(f"\n🔎 Buffering {distance}m...")

        intersections_gdf["buffer"] = intersections_gdf.geometry.buffer(distance)
        buffered = intersections_gdf.set_geometry("buffer")

        joined = gpd.sjoin(accident_gdf, buffered, predicate="within", how="inner")
        joined["distance_m"] = joined.geometry.distance(joined["geometry_right"])

        # joined.to_csv(f"{output_dir}accidents_within_{distance}m_detailed.csv", index=False)
        joined.to_csv(os.path.join(output_dir, f"accidents_within_{distance}m_detailed.csv"), index=False)

        # Count per intersection
        accident_counts = joined.groupby("intersection_cluster").size().reset_index(name="accident_count")
        summary = intersections_df.merge(accident_counts, on="intersection_cluster", how="left").fillna(0)
        summary["accident_count"] = summary["accident_count"].astype(int)
        # summary.to_csv(f"{output_dir}accidents_within_{distance}m_summary.csv", index=False)
        summary.to_csv(os.path.join(output_dir, f"accidents_within_{distance}m_summary.csv"), index=False)

        print(f"✅ Saved {len(joined)} matches and {len(summary)} summaries for {distance}m buffer.")

def summarize_flagged_vs_nonflagged(config, save_dir):
    flagged = pd.read_csv(config["flagged"])
    non_flagged = pd.read_csv(config["non_flagged"])
    flagged_matches = pd.read_csv(config["exact"])
    non_flagged_matches = pd.read_csv(config["exact_non_flagged"])

    f_count = flagged_matches.groupby("intersection_cluster").size().reset_index(name="accident_count")
    nf_count = non_flagged_matches.groupby("intersection_cluster").size().reset_index(name="accident_count")

    df = pd.DataFrame({
        "Group": ["Flagged", "Non-Flagged"],
        "# Intersections": [flagged.shape[0], non_flagged.shape[0]],
        "# With ≥1 Accident": [(f_count["accident_count"] > 0).sum(), (nf_count["accident_count"] > 0).sum()],
        "Total Accidents": [f_count["accident_count"].sum(), nf_count["accident_count"].sum()]
    })
    df.to_csv(os.path.join(save_dir, "summary_table_flagged_vs_non_flagged.csv"), index=False)
    print("✅ Saved summary_table_flagged_vs_non_flagged.csv")

def summarize_by_buffer_radius(config, save_dir):
    rows = []
    for d in [25, 50, 100, 200]:
        df = pd.read_csv(os.path.join(config["summary_dir"], f"accidents_within_{d}m_summary.csv"))
        rows.append({
            "Buffer Radius (m)": d,
            "# Intersections With ≥1 Accident": (df["accident_count"] > 0).sum(),
            "Total Accidents": df["accident_count"].sum()
        })
    pd.DataFrame(rows).to_csv(os.path.join(save_dir,"summary_table_flagged_buffer_distances.csv"), index=False)
    print("✅ Saved summary_table_flagged_buffer_distances.csv")

def stratify_by_distance_group(config, save_dir):
    df = pd.read_csv(config["violation_df"])
    bins = [50, 100, 150, 200]
    labels = [f"{bins[i]}–{bins[i+1]}m" for i in range(len(bins)-1)]
    df["distance_group"] = pd.cut(df["road_distance_m"], bins=bins, labels=labels, right=False).astype(str)
    from_ = df[["from_cluster", "distance_group"]].rename(columns={"from_cluster": "intersection_cluster"})
    to_ = df[["to_cluster", "distance_group"]].rename(columns={"to_cluster": "intersection_cluster"})
    all_ = pd.concat([from_, to_])
    closest = all_.sort_values("distance_group").drop_duplicates("intersection_cluster")
    inter = pd.read_csv(config["flagged"])
    acc = pd.read_csv(config["flagged_counts"])
    merged = inter.merge(closest, on="intersection_cluster").merge(acc, on=["lat", "lon"], how="left").fillna(0)
    merged["accident_count"] = merged["accident_count"].astype(int)
    merged.to_csv(os.path.join(save_dir, "flagged_intersections_with_closest_distance_group.csv"), index=False)
    summary = (
        merged.groupby("distance_group")["accident_count"]
        .agg([("# Intersections", "count"), ("Total Accidents", "sum"),
              ("% With ≥1 Accident", lambda x: (x > 0).mean()*100)])
        .round(1).reset_index()
    )
    summary.to_csv(os.path.join(save_dir, "accident_summary_by_distance_group.csv"), index=False)
    print("✅ Saved accident_summary_by_distance_group.csv")

def compare_flagged_groups_to_nonflagged(config, save_dir):
    flagged = pd.read_csv(config["full"])
    non_flagged = pd.read_csv(config["non_flagged"])
    non_counts = pd.read_csv(config["non_flagged_counts"])
    non_full = non_flagged.merge(non_counts, on="intersection_cluster", how="left").fillna(0)
    non_full["accident_count"] = non_full["accident_count"].astype(int)

    results = []
    for g in flagged["distance_group"].unique():
        f = flagged[flagged["distance_group"] == g]["accident_count"]
        n = non_full["accident_count"]
        stat, p = mannwhitneyu(f, n, alternative="two-sided")
        results.append({
            "Flagged Distance Group": g,
            "U Statistic": stat,
            "p-value": p,
            "Flagged Median": f.median(),
            "Non-Flagged Median": n.median(),
            "Flagged n": len(f),
            "Non-Flagged n": len(n)
        })
    df = pd.DataFrame(results).sort_values("p-value")
    df.to_csv(os.path.join(save_dir, "flagged_intersection_vs_non_flagged_intersection_MannWhitneyU.csv"), index=False)
    print("✅ Saved flagged_intersection_vs_non_flagged_intersection_MannWhitneyU.csv")

def visualize_accident_heatmap(config, save_dir):
    acc = pd.read_csv(config["accident_data_path"])
    acc = acc[acc["Traffic_Control"].astype(str).str.strip() == "01 - Traffic signal"]
    acc = acc.dropna(subset=["Lat", "Long"])
    acc["lat"] = acc["Lat"].round(6)
    acc["lon"] = acc["Long"].round(6)
    heat_data = acc[["lat", "lon"]].values.tolist()
    m = folium.Map(location=[acc["lat"].mean(), acc["lon"].mean()], zoom_start=12)
    HeatMap(heat_data, radius=12, blur=8, max_zoom=13, min_opacity=0.3).add_to(m)

    flagged = pd.read_csv(config["full"])
    for _, r in flagged.iterrows():
        color = {"50–100m": "red", "100–150m": "orange", "150–200m": "purple"}.get(r["distance_group"], "gray")
        folium.CircleMarker(
            location=[r["lat"], r["lon"]], radius=5, color=color, fill=True,
            fill_color=color, fill_opacity=0.8,
            popup=f"{r['distance_group']}, Accidents: {r['accident_count']}"
        ).add_to(m)
    m.save(os.path.join(save_dir, "ottawa_accident_heatmap_with_flagged_intersections.html"))
    print("✅ Saved heatmap")

def analyze_lighting_conditions(config, save_dir):
    acc = pd.read_csv(config["accident_data_path"])
    acc = acc[acc["Traffic_Control"].astype(str).str.strip() == "01 - Traffic signal"]
    acc["lat_rounded"] = acc["Lat"].round(7)
    acc["lon_rounded"] = acc["Long"].round(7)
    flagged = pd.read_csv(config["flagged"])
    flagged["lat_rounded"] = flagged["lat"].round(7)
    flagged["lon_rounded"] = flagged["lon"].round(7)
    f_matches = pd.merge(acc, flagged, on=["lat_rounded", "lon_rounded"], how="inner")
    non_flagged = pd.read_csv(config["non_flagged"])
    non_flagged["lat_rounded"] = non_flagged["lat"].round(7)
    non_flagged["lon_rounded"] = non_flagged["lon"].round(7)
    nf_matches = pd.merge(acc, non_flagged, on=["lat_rounded", "lon_rounded"], how="inner")

    light_df = pd.concat([
        f_matches["Light"].value_counts().rename("Flagged"),
        nf_matches["Light"].value_counts().rename("Non-Flagged")
    ], axis=1).fillna(0)

    light_df.plot(kind="bar", figsize=(10, 5))
    plt.title("Lighting Condition of Accidents at Intersections")
    plt.ylabel("Accident Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(save_dir, "lighting_condition_bar_chart.png"))

    chi2, p, dof, expected = chi2_contingency(light_df.astype(int))
    print("\n📊 Chi-Square Test on Lighting Condition Distribution:")
    print(f"Chi2 statistic = {chi2:.4f}")
    print(f"p-value = {p:.4f}")
    print(f"Degrees of freedom = {dof}")

def extract_flagged_intersections(violation_df, cluster_centroids):
    flagged_ids = pd.unique(violation_df[["from_cluster", "to_cluster"]].values.ravel())
    return cluster_centroids[cluster_centroids["intersection_cluster"].isin(flagged_ids)].copy()

def match_and_count_accidents(accidents, flagged, non_flagged):
    accidents.columns = accidents.columns.str.strip()
    accidents["Traffic_Control"] = accidents["Traffic_Control"].astype(str).str.strip()
    accidents = accidents[accidents["Traffic_Control"] == "01 - Traffic signal"]

    for df in [flagged, non_flagged]:
        df["lat_rounded"] = df["lat"].round(7)
        df["lon_rounded"] = df["lon"].round(7)

    accidents["lat_rounded"] = accidents["Lat"].round(7)
    accidents["lon_rounded"] = accidents["Long"].round(7)

    flagged_matches = pd.merge(accidents, flagged, on=["lat_rounded", "lon_rounded"], how="inner")
    non_flagged_matches = pd.merge(accidents, non_flagged, on=["lat_rounded", "lon_rounded"], how="inner")

    flagged_counts = flagged_matches.groupby(["lat", "lon"]).size().reset_index(name="accident_count")
    non_flagged_counts = non_flagged_matches.groupby("intersection_cluster").size().reset_index(name="accident_count")

    flagged_counts["accident_count"] = flagged_counts["accident_count"].fillna(0).astype(int)
    non_flagged_counts["accident_count"] = non_flagged_counts["accident_count"].fillna(0).astype(int)

    return flagged_counts, non_flagged_counts, flagged_matches, non_flagged_matches


def run_violation_analysis(city, lat, long, radius, violating_distance):
    print("🛣️ Running violation analysis...")

    # Step 1: Get graph + lights
    if city == "Ottawa, Ontario, Canada":
      G = get_graph(place=city)
      traffic_lights = get_traffic_lights(place=city)
    else:
      G = get_graph(point=(lat, long), dist=radius)
      traffic_lights = get_traffic_lights(point=(lat, long), dist=radius )

    # Step 2: Cluster
    traffic_lights = cluster_intersections(traffic_lights)

    # Step 3: Compute cluster centroids
    cluster_centroids = (
        traffic_lights.to_crs(epsg=4326)
        .groupby("intersection_cluster")
        .geometry.apply(lambda g: g.union_all().centroid)
        .reset_index()
    )
    cluster_centroids["lat"] = cluster_centroids.geometry.y
    cluster_centroids["lon"] = cluster_centroids.geometry.x

    # Step 4: Snap
    cluster_centroids = snap_to_nearest_node(cluster_centroids, G)

    # Step 5: Violation pairs
    violation_df, paths_by_pair = compute_violation_pairs(cluster_centroids, G, threshold=violating_distance)
    violation_df = violation_df[violation_df["road_distance_m"] >= 50].copy()

    # Step 6: Map not saved here — optional preview in GUI
    traffic_lights_latlon = traffic_lights.to_crs(epsg=4326)

    # Step 7: Extract flagged intersections
    flagged = extract_flagged_intersections(violation_df, cluster_centroids)

    print(f"✅ Analysis complete. {len(violation_df)} violating pairs found.")

    return violation_df, paths_by_pair, traffic_lights_latlon, cluster_centroids, flagged, G