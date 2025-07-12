from tkinter import *
from tkinter import ttk
from analysis_utils import run_violation_analysis
import tkintermapview
from tkinter import messagebox
from analysis_utils import get_place_from_point
from ottawa_statistical_analysis import run_statistical_analysis
import zipfile
import tempfile
from tkinter import filedialog

selected_location = None
overlay_elements = []
place = ""
violations_plotted = False

def set_marker_event(coords):
    global selected_location
    print("Add marker:", coords)
    if selected_location:
        selected_location.delete()
    selected_location = map_widget.set_marker(coords[0], coords[1], text="Selected Location")
    
def show_error(message):
    messagebox.showerror("Error", message)
  
def search_results():
    try:
        radius_input = float(radius.get())
        violating_distance_input = float(violating_distance.get())
    except ValueError:
        show_error("Please enter a numeric value for the minimum distance between traffic lights and radius first.")
        return

    if not selected_location:
        show_error("Please place a marker on the map first.")
        return

    latitude_input = map_widget.get_position()[0]
    longitude_input = map_widget.get_position()[1]

    # Reverse geocode lat/lon to place name
    global place 
    place = get_place_from_point(latitude_input, longitude_input)
    if not place:
        show_error("Could not determine the place from selected location.")
        return

    try:
        # Run violation analysis using place name
        route_violation_df, paths_by_pair, traffic_lights_latlon, cluster_centroids, flagged, G = run_violation_analysis(
            lat=latitude_input,
            long=longitude_input,
            city = place,
            radius=radius_input,
            violating_distance=violating_distance_input,
        )

        for item in overlay_elements:
            item.delete()
        overlay_elements.clear()

        if route_violation_df.empty:
            messagebox.showinfo("No Violations", "There are no violations under the given parameters.")
            return

        violation_clusters = set(route_violation_df["from_cluster"]).union(route_violation_df["to_cluster"])
        for _, row in traffic_lights_latlon.iterrows():
            if row["intersection_cluster"] in violation_clusters:
                marker = map_widget.set_marker(row.geometry.y, row.geometry.x, text="🚦")
                overlay_elements.append(marker)

        for _, row in route_violation_df.iterrows():
            c1, c2 = row["from_cluster"], row["to_cluster"]
            path_nodes = paths_by_pair.get((c1, c2))
            if path_nodes:
                coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path_nodes]
                path = map_widget.set_path(coords)
                path.tooltip_text = f"{row['road_distance_m']}m"
                overlay_elements.append(path)

        messagebox.showinfo("Violations Found", f"{len(route_violation_df)} violations found and plotted.")
        global violations_plotted
        violations_plotted = True

    except Exception as e:
        print("Error occurred:", e)
        show_error(f"Something went wrong:\n{str(e)}")

def reset_map():
    global selected_location, overlay_elements, violations_plotted
    # Remove overlay paths and markers
    for item in overlay_elements:
        item.delete()
    overlay_elements.clear()

    # Remove user marker
    if selected_location:
        selected_location.delete()
        selected_location = None

    # Reset map position and zoom
    map_widget.set_position(43.7315, -79.7624)
    map_widget.set_zoom(10)
    violations_plotted = False

def check_location():
    if not violations_plotted:
        show_error("Please search a place first")
    elif place != "Ottawa, Ontario, Canada":
        show_error("Accident data is only available for Ottawa, Ontario, Canada")
    else: 
        run_statistical_analysis()
        

root = Tk()
root.title("Traffic Light Violations App")
root.geometry("1024x768")

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

mainframe = Frame(root, padx=20, pady=20)
mainframe.grid(column=0, row=0, sticky="nsew")

mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(2, weight=1)  # Make map area expandable

# Instructions
label = Label(mainframe,
              text="Instructions: Click on the map to drop a pin. Then enter distance between traffic lights below.",
              font=("Helvetica", 14), anchor="w", justify="left")
label.grid(column=0, row=0, sticky="ew", pady=(0, 10))

# Top Inputs
topframe = Frame(mainframe)
topframe.grid(row=1, column=0, sticky="ew", pady=(0, 10))
topframe.columnconfigure(1, weight=1)

Label(topframe, text="Distance Between Lights:").grid(row=0, column=0, sticky="e", padx=(10, 0))
violating_distance = StringVar()
ttk.Entry(topframe, textvariable=violating_distance, width=10).grid(row=0, column=1)

Label(topframe, text="Radius").grid(row=0, column=2, sticky="e", padx=(10, 0))
radius = StringVar()
ttk.Entry(topframe, textvariable=radius, width=10).grid(row=0, column=3)

ttk.Button(topframe, text="Search", command=search_results).grid(row=0, column=4, padx=(10, 0))
ttk.Button(topframe, text="Redo", command=reset_map).grid(row=0, column=5, padx=(10, 0))
ttk.Button(topframe, text="Run stats", command=check_location).grid(row=0, column=6, padx=(10, 0))



# Map
mapframe = LabelFrame(mainframe)
mapframe.grid(row=2, column=0, sticky="nsew")

map_widget = tkintermapview.TkinterMapView(mapframe, width=800, height=600, corner_radius=0)
map_widget.set_tile_server("https://a.tile.openstreetmap.org/{z}/{x}/{y}.png")
map_widget.set_position(43.7315, -79.7624)
map_widget.set_zoom(10)
map_widget.pack(fill=BOTH, expand=True)

map_widget.add_right_click_menu_command(label="Add Marker Here",
                                        command=set_marker_event,
                                        pass_coords=True)
root.mainloop()