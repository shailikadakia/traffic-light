from tkinter import *
from tkinter import ttk
from main import run_violation_analysis
import tkintermapview
from tkinter import messagebox

selected_location = None

def set_marker_event(coords):
    global selected_location
    print("Add marker:", coords)
    if selected_location:
        selected_location.delete()
    selected_location = map_widget.set_marker(coords[0], coords[1], text="Selected Location")
    
def show_error(message):
    messagebox.showerror("Error", message)
  
def search_results():
    city_input = 'Test'
    try:
      radius_input = float(radius.get())
      violating_distance_input = float(violating_distance.get())
    except ValueError:
        show_error("Please entry numeric values for radius and distance first")
    
    if not selected_location:
        show_error("Please place a marker on the map first")

    if map_widget.get_position():
        latitude_input = map_widget.get_position()[0]
        longitude_input = map_widget.get_position()[1]
        try:
          violation_count = run_violation_analysis(city_input, radius_input, latitude_input, longitude_input, violating_distance_input)

          if violation_count == 0:
              messagebox.showinfo("No violationns", "There are no violations under the given parameters")
          else: 
            messagebox.showinfo("Violations Found", f"{violation_count} violations found and mapped.")

        except Exception as e:
            print("Error occured")
            show_error(str(e))

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
              text="Instructions: Click on the map to drop a pin. Then enter radius and violation distance below.",
              font=("Helvetica", 14), anchor="w", justify="left")
label.grid(column=0, row=0, sticky="ew", pady=(0, 10))

# Top Inputs
topframe = Frame(mainframe)
topframe.grid(row=1, column=0, sticky="ew", pady=(0, 10))
topframe.columnconfigure(1, weight=1)

Label(topframe, text="Radius:").grid(row=0, column=0, sticky="e")
radius = StringVar()
ttk.Entry(topframe, textvariable=radius, width=10).grid(row=0, column=1)

Label(topframe, text="Distance Between Lights:").grid(row=0, column=2, sticky="e", padx=(10, 0))
violating_distance = StringVar()
ttk.Entry(topframe, textvariable=violating_distance, width=10).grid(row=0, column=3)

ttk.Button(topframe, text="Search", command=search_results).grid(row=0, column=4, padx=(10, 0))

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
