import plotly.graph_objs as go
import cv2
from matplotlib import pyplot as plt
import utils
from ipywidgets import interact, IntSlider, Play, VBox, jslink
from IPython.display import display
import numpy as np

def plot_detections_on_image(img, robot_positions, cube_positions, donut_positions):
    plt.figure(figsize=(12, 8))
    plt.imshow(img)

    # Plot robot arms (2 points)
    plt.plot(robot_positions[0][0], robot_positions[0][1], 'co', markersize=10, label='Robot Arm 1')
    plt.plot(robot_positions[1][0], robot_positions[1][1], 'c^', markersize=10, label='Robot Arm 2')

    # Plot cubes (3 points)
    cube_colors = ['blue', 'green', 'red']
    for i, cube in enumerate(cube_positions):
        plt.plot(cube[0], cube[1], 'o', color=cube_colors[i], markersize=10, label=f'{cube_colors[i].capitalize()} Cube')

    # Plot donuts (3 points)
    donut_colors = ['blue', 'green', 'red']
    for i, donut in enumerate(donut_positions):
        plt.plot(donut[0], donut[1], 'x', color=donut_colors[i], markersize=10, label=f'{donut_colors[i].capitalize()} Donut')

    plt.title(f"Detections on Image")
    plt.axis('off')

    # Remove duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc='upper right')

    plt.show()


def localize_show(image_path, new_size):
    """
    arg: new_size : Calibration Image size
        image_path_path : Image path to localize

    returns: Localizations of robot centre, arm, cubes and target

    This function also helps to visualize the points on the image 
    """
    test_im = cv2.imread(image_path)
    #new_size = (1096,730 )  # (width, height)

    # Resize the image
    resized_img = cv2.resize(test_im, new_size, interpolation=cv2.INTER_LINEAR)

    #print(resized_img.shape)
    POSITIONS = utils.locate_robot(resized_img )
    cubes = utils.locate_all_cubes(resized_img)
    donuts = utils.locate_donuts(resized_img )
    rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    plot_detections_on_image(rgb, POSITIONS, cubes, donuts)
    return POSITIONS, cubes, donuts

def show_sys3d (robot3d,robot_arm,b_cube3d,r_cube3d,g_cube3d, r_tar3d, g_tar3d,b_tar3d):

    points = [
        (robot3d, 'robot', 'black'),
        (robot_arm, 'arm', 'gray'),
        (b_cube3d, 'b_cube', 'blue'),
        (g_cube3d, 'g_cube', 'green'),
        (r_cube3d, 'r_cube', 'red'),
        (b_tar3d, 'b_target', 'cyan'),
        (g_tar3d, 'g_target', 'lime'),
        (r_tar3d, 'r_target', 'magenta'),
    ]

    # Create traces
    traces = []
    for pt, label_, color in points:
        trace = go.Scatter3d(
            x=[pt[0]], y=[pt[1]], z=[pt[2]],
            mode='markers+text',
            marker=dict(size=6, color=color),
            text=[label_],
            textposition='top center',
            name=label_
        )
        traces.append(trace)

    # Layout
    layout = go.Layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title='Interactive 3D Point Plot'
    )

    # Plot
    fig = go.Figure(data=traces, layout=layout)
    fig.show()

def show_sys_plane(robot3d,robot_arm,b_cube3d,r_cube3d,g_cube3d, r_tar3d, g_tar3d,b_tar3d):
    robot2d = robot3d[:2]
    robot_arm2d = robot_arm[:2]
    b_cube2d = b_cube3d[:2]
    g_cube2d = g_cube3d[:2]
    r_cube2d = r_cube3d[:2]
    b_tar2d = b_tar3d[:2]
    g_tar2d = g_tar3d[:2]
    r_tar2d = r_tar3d[:2]

    # Collect all 2D points
    points_2d = [
        (robot2d, 'robot', 'black'),
        (robot_arm2d, 'arm (adjusted)', 'gray'),
        (b_cube2d, 'b_cube', 'blue'),
        (g_cube2d, 'g_cube', 'green'),
        (r_cube2d, 'r_cube', 'red'),
        (b_tar2d, 'b_target', 'cyan'),
        (g_tar2d, 'g_target', 'lime'),
        (r_tar2d, 'r_target', 'magenta'),
    ]

    # Plot using Plotly 2D scatter
    traces = []
    for pt, label_, color in points_2d:
        trace = go.Scatter(
            x=[pt[0]], y=[pt[1]],
            mode='markers+text',
            marker=dict(size=10, color=color),
            text=[label_],
            textposition='top center',
            name=label_
        )
        traces.append(trace)

    # Layout
    layout = go.Layout(
        xaxis_title='X', yaxis_title='Y',
        title='Flattened 2D View (Z Removed, Y of Arm = Y of Robot)',
        margin=dict(l=20, r=20, t=40, b=40),
        showlegend=False
    )

    # Show figure
    fig = go.Figure(data=traces, layout=layout)
    fig.show()
    
def vis_command(cmds,robot3d,robot_arm,b_cube3d,r_cube3d,g_cube3d, r_tar3d, g_tar3d,b_tar3d):
    # Define all 2D points (using x,y from original 3D points)
    robot = robot3d[:2]          # Base robot position
    robot_arm  = robot_arm[:2]       # Arm attachment point
    b_cube = b_cube3d[:2]           # Blue cube
    r_cube = r_cube3d[:2]           # Red cube
    g_cube = g_cube3d[:2]           # Green cube
    b_target = b_tar3d[:2]      # Blue target
    r_target = r_tar3d[:2]         # Red target
    g_target = g_tar3d[:2]         # Green target

    # Initial heading (robot to arm)
    arm_vec = robot_arm - robot
    heading = np.rad2deg(np.arctan2(arm_vec[1], arm_vec[0]))



    # Animation storage
    frames = []
    current_pos = robot.copy()  # Tracks robot position
    held_cube = None            # Tracks held cube (None if none)

    def save_frame(pos, heading, cube):
        """Store current state for animation."""
        frames.append({
            "robot_pos": pos.copy(),
            "heading": heading,
            "cube_pos": cube.copy() if cube is not None else None
        })

    save_frame(current_pos, heading, None)  # Initial frame

    # Generate animation frames
    for cmd in cmds:
        if cmd["action"] == "turn":
            delta = cmd["value"]
            steps = 20
            for i in range(1, steps + 1):
                new_heading = heading + delta * i / steps
                save_frame(current_pos, new_heading, held_cube)
            heading += delta

        elif cmd["action"] == "go":
            dist = cmd["value"]
            steps = 30
            dx = np.cos(np.deg2rad(heading)) * dist / steps
            dy = np.sin(np.deg2rad(heading)) * dist / steps
            for _ in range(steps):
                current_pos += np.array([dx, dy])
                if held_cube is not None:
                    held_cube = current_pos.copy()
                save_frame(current_pos, heading, held_cube)

        elif cmd["action"] == "grab":
            held_cube = b_cube.copy()  # Pick blue cube by default
            save_frame(current_pos, heading, held_cube)

        elif cmd["action"] == "let_go":
            held_cube = None
            save_frame(current_pos, heading, None)

    # 2D Plot function
    def plot_frame(i):
        frame = frames[i]
        plt.figure(figsize=(10, 8))
        plt.xlim(-20, 50)
        plt.ylim(-20, 50)
        plt.grid(True)
        plt.title(f"Frame {i} | Heading: {frame['heading']:.2f}°")
        plt.xlabel("X")
        plt.ylabel("Y")

        # Plot robot (red dot)
        plt.scatter(*frame["robot_pos"], color='magenta', s=100, label='Robot')
        
        # Plot heading arrow
        dx = np.cos(np.deg2rad(frame["heading"])) * 5
        dy = np.sin(np.deg2rad(frame["heading"])) * 5
        plt.arrow(
            frame["robot_pos"][0], frame["robot_pos"][1],
            dx, dy, head_width=1, color='black'
        )

        # Plot held cube (if any)
        if frame["cube_pos"] is not None:
            plt.scatter(*frame["cube_pos"], color='blue', s=50, label='Held Cube')

        # Plot all static objects (semi-transparent)
        for name, pos, color in [
            ("Arm", robot_arm, "orange"),
            ("Blue Cube", b_cube, "blue"),
            ("Red Cube", r_cube, "red"),
            ("Green Cube", g_cube, "green"),
            ("Blue Target", b_target, "cyan"),
            ("Red Target", r_target, "pink"),
            ("Green Target", g_target, "lime")
        ]:
            plt.scatter(*pos, color=color, s=50, alpha=0.3, label=name)

        plt.legend()
        plt.show()

    # Interactive widgets
    slider = IntSlider(value=0, min=0, max=len(frames)-1, step=1, description='Frame')
    play = Play(value=0, min=0, max=len(frames)-1, step=1, interval=100, description="Play")
    jslink((play, 'value'), (slider, 'value'))

    display(VBox([play, slider]))
    interact(plot_frame, i=slider)

if __name__ == "__main__":
    pass