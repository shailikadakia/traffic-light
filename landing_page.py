from tkinter import *
from tkinter import ttk

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

# Row 1
Label(mainframe, text="City:").grid(column=0, row=1, sticky="e", padx=(0, 10))
city = StringVar()
ttk.Entry(mainframe, textvariable=city).grid(column=1, row=1, sticky="ew")

Label(mainframe, text="Latitude:").grid(column=2, row=1, sticky="e", padx=(10, 10))
latitude = StringVar()
ttk.Entry(mainframe, textvariable=latitude).grid(column=3, row=1, sticky="ew")

Label(mainframe, text="Longitude:").grid(column=4, row=1, sticky="e", padx=(10, 10))
longitude = StringVar()
ttk.Entry(mainframe, textvariable=longitude).grid(column=5, row=1, sticky="ew")

# Row 2
Label(mainframe, text="Radius:").grid(column=0, row=2, sticky="e", padx=(0, 10))
radius = StringVar()
ttk.Entry(mainframe, textvariable=radius).grid(column=1, row=2, sticky="ew")

Label(mainframe, text="Distance to check between traffic lights:").grid(column=2, row=2, columnspan=2, sticky="e", padx=(10, 10))
violating_distance = StringVar()
ttk.Entry(mainframe, textvariable=violating_distance).grid(column=4, row=2, columnspan=2, sticky="ew")

root.mainloop()


