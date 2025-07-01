from tkinter import * 
import tkintermapview

root = Tk()
root.title("Map Example")
root.geometry("900x700")

my_label = LabelFrame(root)
my_label.pack(pady=20)

map_widget = tkintermapview.TkinterMapView(my_label, width=800, height=600, corner_radius=0)
map_widget.set_position(36.1699, -115.1396)
map_widget.set_zoom(10)
map_widget.pack()

root.mainloop()