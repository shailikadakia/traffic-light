from tkinter import *
from tkinter import ttk
import re
from main import run_violation_analysis
from tkinterhtml import HtmlFrame 
import webbrowser
import json
import os


def search_results():
    radius_input = float(radius.get())
    violating_distance_input = float(violating_distance.get())

    run_violation_analysis(radius_input, violating_distance_input)

def validate_city():
  print('validating city')


def validate_latitude():
  print('validating latitude')


def validate_longtitude():
  print('validating longitude')

def validate_radius():
  print('validating radius')

def validate_distance():
  print('validating distance')

root = Tk()
root.title("Traffic Light Violations App")
root.geometry("1024x768")  # Start big enough to fit everything

# Allow root window to expand
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Main frame
mainframe = Frame(root, padx=20, pady=20)
mainframe.grid(column=0, row=0, sticky="nsew")

# Allow columns 1 and 3 to stretch
mainframe.columnconfigure(1, weight=1)
mainframe.columnconfigure(3, weight=1)
mainframe.columnconfigure(5, weight=1)

# Instructions
label = Label(mainframe,
              text="Instructions: This is an app to view traffic light violations in a city.",
              justify="left", anchor="w", font=("Helvetica", 14))
label.grid(column=0, row=0, columnspan=6, sticky="ew", pady=(0, 20))

# Wrapping
def on_resize(event):
    label.config(wraplength=event.width - 40)

label.bind("<Configure>", on_resize)

def show_click_map():
    if os.path.exists("selected_coords.json"):
        os.remove("selected_coords.json")
    webbrowser.open("click_map.html")  # opens in browser

def check_for_coords_and_run():
    if os.path.exists("selected_coordinates.json"):
        with open("selected_coordinates.json.json") as f:
            data = json.load(f)
        lat = float(data["lat"])
        lng = float(data["lng"])
        run_violation_analysis(city="Selected", radius=float(radius.get()),
                               latitude=lat, longtitude=lng,
                               violating_distance=float(violating_distance.get()))
        webbrowser.open("Selected.html")  # opens generated map
    else:
        print("No coordinates selected yet.")


# Row 2
Label(mainframe, text="Radius:").grid(column=0, row=2, sticky="e", padx=(0, 10))
radius = StringVar()
ttk.Entry(mainframe, textvariable=radius).grid(column=1, row=2, sticky="ew")

Label(mainframe, text="Distance to check between traffic lights:").grid(column=2, row=2, columnspan=2, sticky="e", padx=(10, 10))
violating_distance = StringVar()
ttk.Entry(mainframe, textvariable=violating_distance).grid(column=4, row=2, columnspan=2, sticky="ew")

ttk.Button(mainframe, text="Select Location on Map", command=show_click_map).grid(column=1, row=3, columnspan=2, pady=10)
ttk.Button(mainframe, text="Generate Violation Map", command=check_for_coords_and_run).grid(column=3, row=3, columnspan=2, pady=10)

root.mainloop()


