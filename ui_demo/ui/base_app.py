import tkinter as tk
import tkinter.ttk as ttk

from ..lib.asynctk import AsyncTk


class MainApp:
    def __init__(self, master=None):
        # build ui
        self.main_toplevel = AsyncTk()
        self.top_frame = ttk.Frame(self.main_toplevel)
        self.input_label = tk.Label(self.top_frame)
        self.input_label.configure(
            bitmap="gray12", height=300, relief="ridge", width=300
        )
        self.input_label.pack(expand="true", fill="both", padx=3, pady=3, side="left")
        self.switch_button = ttk.Button(self.top_frame)
        self.switch_button.configure(width=3)
        self.switch_button.pack(side="left")
        self.switch_button.configure(command=self.switch)
        self.output_label = tk.Label(self.top_frame)
        self.output_label.configure(
            bitmap="gray12", height=300, relief="ridge", width=300
        )
        self.output_label.pack(expand="true", fill="both", padx=3, pady=3, side="left")
        self.top_frame.configure(height=200, width=200)
        self.top_frame.pack(expand="true", fill="both", side="top")
        self.open_button = ttk.Button(self.main_toplevel)
        self.open_button.configure(text="open")
        self.open_button.pack(anchor="w", padx=9, pady=3, side="top")
        self.open_button.configure(command=self.open_camera)
        self.close_button = ttk.Button(self.main_toplevel)
        self.close_button.configure(state="disabled", text="close")
        self.close_button.pack(anchor="w", padx=9, pady=3, side="top")
        self.close_button.configure(command=self.close_camera)
        self.main_menu = tk.Menu(self.main_toplevel)
        self.file_submenu = tk.Menu(
            self.main_menu, relief="flat", takefocus=False, tearoff="false"
        )
        self.main_menu.add(tk.CASCADE, menu=self.file_submenu, label="file")
        self.mi_open_command = 0
        self.file_submenu.add("command", label="open")
        self.file_submenu.entryconfigure(
            self.mi_open_command, command=self.open_from_file
        )
        self.main_menu.configure(tearoff="false")
        self.main_toplevel.configure(menu=self.main_menu)
        self.main_toplevel.configure(height=200, width=200)

        # Main widget
        self.mainwindow = self.main_toplevel

    def run(self):
        self.mainwindow.mainloop()

    def switch(self):
        pass

    def open_camera(self):
        pass

    def close_camera(self):
        pass

    def open_from_file(self):
        pass


if __name__ == "__main__":
    app = MainApp()
    app.run()

