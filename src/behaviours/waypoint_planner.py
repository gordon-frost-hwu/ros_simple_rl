#!/usr/bin/python

from Tkinter import *
import csv

# CANVAS_SIZE / SCALING_FACTOR is the max North & East position in the canvas
CANVAS_SIZE = 600   # in pixels
SCALING_FACTOR = 30
CANVAS_SIZE_METRES = float(CANVAS_SIZE) / float(SCALING_FACTOR)

points = []
points_to_save = []
points_to_save_6DOF = []

spline = 0

tag1 = "theline"

class ChildWindow(object):
    def __init__(self, main_app, click_event):
        """
        :param main_app: root
        :param click_event: class is used to enter more params for a wp on a mouse click
        :return: returns 6DOF waypoint back to the main app using the ok_button callback
        """
        self.child_window = Toplevel()
        self.main_app = main_app
        self.click_event = click_event

        # Define the GUI widgets and their variables/methods
        self.setWidgets()

    def on_okCallback(self, event=False):
        # Get all the data ready to pass back to main application (root)
        if self.depth_entry.get() != "" and self.pitch_entry.get() != "" and self.yaw_entry.get() != "":
            waypoint = [float(self.north_entry.get()), float(self.east_entry.get()), float(self.depth_entry.get()),
                                    float(self.pitch_entry.get()), float(self.yaw_entry.get())]
        else:
            waypoint = [float(self.north_entry.get()), float(self.east_entry.get()), 0, 0, 0]
        self.main_app._save_waypoint(waypoint)
        self.child_window.destroy()

    def setWidgets(self):
        """ Set all widgets for the GUI """

        self.req_type = StringVar(self.child_window)
        self.req_type.set("World") # default value

        if self.main_app.round_coordinates.get() == False:
            north_s = "{0:.2f}".format(float(CANVAS_SIZE_METRES - (self.click_event.y)/SCALING_FACTOR))
            east_s = "{0:.2f}".format(float(self.click_event.x)/SCALING_FACTOR)
        else:
            north_s = "{0:.0f}".format(CANVAS_SIZE_METRES - (self.click_event.y/SCALING_FACTOR))
            east_s = "{0:.0f}".format(self.click_event.x/SCALING_FACTOR)

        # North Entry
        north_label = Label(self.child_window, text='North: ')
        north_label.grid(row=4, column=1)
        self.north_entry = Entry(self.child_window)
        self.north_entry.insert(0, north_s)
        self.north_entry.grid(row=4, column=2)

        # East Entry
        east_label = Label(self.child_window, text='East: ')
        east_label.grid(row=5, column=1)
        self.east_entry = Entry(self.child_window)
        self.east_entry.insert(0, east_s)
        self.east_entry.grid(row=5, column=2)

        # Depth Entry
        depth_label = Label(self.child_window, text='Depth: ')
        depth_label.grid(row=6, column=1)
        self.depth_entry = Entry(self.child_window)
        self.depth_entry.grid(row=6, column=2)
        self.depth_entry.focus()

        # Pitch Entry
        pitch_label = Label(self.child_window, text='Pitch: ')
        pitch_label.grid(row=7, column=1)
        self.pitch_entry = Entry(self.child_window)
        self.pitch_entry.grid(row=7, column=2)

        # Yaw Entry
        yaw_label = Label(self.child_window, text='Yaw: ')
        yaw_label.grid(row=8, column=1)
        self.yaw_entry = Entry(self.child_window)
        self.yaw_entry.grid(row=8, column=2)

        self.ok_button = Button(self.child_window, text="Ok", command=self.on_okCallback)
        self.ok_button.grid(row=9, column=2)
        self.child_window.bind('<Return>', self.on_okCallback)

class MainApplication(Tk):
    def __init__(self, root):
        self.setWidgets(root)

    def _save_waypoint(self, point_6dof):
        print "Adding waypoint to list: %s" % point_6dof
        points_to_save_6DOF.append(point_6dof)

    def setWidgets(self, root):
        self.c = Canvas(root, borderwidth=5, bg="white", width=CANVAS_SIZE, height=CANVAS_SIZE)
        self.c.configure(cursor="crosshair")
        self.c.pack(padx=10,pady=10)
        self.c.bind("<Button-1>", self.pointButtonCallback)
        self.c.bind("<Button-3>", self.graphButtonCallback)
        self.c.bind("<Button-2>", self.toggleButtonCallback)
        self.c.bind('<Motion>', self.updateCanvasMousePosition)

        self.b = Button(root, text='Save Waypoints', command=self.saveWaypoints)
        self.b.pack()
        self.b_del_points = Button(root, text='Delete Waypoints', command=self.deleteWaypoints)
        self.b_del_points.pack()

        self.round_coordinates = IntVar()
        self.round_coordinates.set(1)    # Set it to default as 1/True
        self.checkbox_round = Checkbutton(root, text="Round Coordinates", variable=self.round_coordinates)
        self.checkbox_round.pack()

        self.curr_var = StringVar(value='-:-')
        Label(root, text='Current point: ').pack(side=LEFT)
        self.curr = Label(root, textvariable=self.curr_var)
        self.curr.pack(side=LEFT)

    def pointButtonCallback(self, event):
        #print "X: %s, Y: %s" % (event.x, event.y)
        self.c.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, fill="black")
        points.append(event.x)
        points.append(event.y)
        points_to_save.append([CANVAS_SIZE_METRES - (float(event.y)/SCALING_FACTOR), float(event.x)/SCALING_FACTOR])

        sub_figure = ChildWindow(self, event)
        return points

    def graphButtonCallback(self, event):
        global theline
        self.c.create_line(points, tags="theline")

    def toggleButtonCallback(self, event):
        global spline
        if spline == 0:
            self.c.itemconfigure(tag1, smooth=1)
            spline = 1
        elif spline == 1:
            self.c.itemconfigure(tag1, smooth=0)
            spline = 0
        return spline

    def updateCanvasMousePosition(self, event):
        # Limit the calculated North, East to 2 decimal places
        if self.round_coordinates.get() == False:
            s = "North: " + "{0:.2f}".format(CANVAS_SIZE_METRES - (float(event.y)/SCALING_FACTOR)) + "; East: " + "{0:.2f}".format(float(event.x)/SCALING_FACTOR)
        else:
            s = "North: " + "{0:.0f}".format(CANVAS_SIZE_METRES - (event.y/SCALING_FACTOR)) + "; East: " + "{0:.0f}".format(event.x/SCALING_FACTOR)
        self.curr_var.set(s)

    def saveWaypoints(self, event=False):
        # Save the 2D waypoints to one file
        self.save('/home/gordon/ros_workspace/auv_learning/behaviours/data/waypoints_2d.csv', points_to_save)

        # TODO: Fix the non-working round function for 6DOF saving of waypoints
        # and save the full 6DOF waypoints to another file
        self.save('/home/gordon/ros_workspace/auv_learning/behaviours/data/waypoints_6d.csv', points_to_save_6DOF)

    def save(self, filepath, data):
        print "data: ", data
        with open(filepath, 'wb') as outcsv:
            writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            for line in data:
                if self.round_coordinates.get() == False:
                    north = float(line[0])/SCALING_FACTOR; east = float(line[1])/SCALING_FACTOR
                    to_write = [float(a) for a in line]
                else:
                    north = round(line[0]/SCALING_FACTOR); east = round(line[1]/SCALING_FACTOR)
                    to_write = [round(a) for a in line]
                print "Writing: " + str(to_write)
                # writer.writerow([north, east])
                writer.writerow(to_write)
    def deleteWaypoints(self, event=False):
        # TODO: doesn't work atm. Need to look up canvas.create_oval method to see how it handles the drawn points
        points = []
        points_to_save = []
        print points


if __name__ == '__main__':
    root = Tk()
    root.title("Simple Waypoint Planner")
    root.resizable(0,0)

    app = MainApplication(root)
    root.mainloop()
