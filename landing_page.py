from tkinter import *
from tkinter import ttk

def open_new_window():
    new_window = Tk() # Create a new window
    new_window.title("New Window")
    new_window.geometry("250x150")  

    Label(new_window, text="This is a new window").grid(column=0, row=0, sticky="w")
    root.destroy()
  
root = Tk()
root.title("Traffic Light Violations App")
root.geometry("500x500")  

# Set up the main frame
mainframe = Frame(root, padx=10, pady=10)
mainframe.grid(column=0, row=0, sticky="nsew")

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

label = Label(mainframe,
              text="This is an app to view traffic light violations in a city. ",
              justify="left", anchor="w")
label.grid(column=0, row=0, sticky="nsew")
button = Button(master=mainframe, text="Open New Window", command=open_new_window)
button.grid(column=0, row=0, sticky="w")



def on_resize(event):
    label.config(wraplength=label.winfo_width())

label.bind("<Configure>", on_resize)

root.mainloop()
