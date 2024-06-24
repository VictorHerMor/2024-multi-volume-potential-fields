#!/usr/bin/python3

'''
Copyright (C) 2024 Victor Hernandez Moreno

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Animation3D(object):
    """
    Animation class
    """

    def __init__(self, specs, save_path, scenery_objs, ee_meshes, rep_positions, goal_obj=None, dem_positions=None, ee_velos=None, thetas=None, wrenches=None):
        """
        Initialise class

        Args:
            specs: dict | dictionary of the animation parameters
            save_path: String | path to the corresponding test_dir
            sqObj_obs_list: list of all the obstacles as a sqObj | Used to plot the obstacles as a mesh
            dem_positions: np ndarray | original trajectory positions (timestamp, num_dim)
            ee_mesh_points: np ndarray | position of the EE mesh overtime. Shape is (steps, num_dim, mesh_x, mesh_y, mesh_z)
            ee_position: np ndarray | xyz position of the EE over time (steps, num_dim(ie xyz positions))
            theta_data: np ndarray | theta values overtime (steps theta values)
            wrench_data: np ndarray | wrench values overtime (steps, wrench values)
            ee_vel_data: np ndarray | ee lin and ang vel overtime (steps, lin/ang vel)
        """
        self.save_path = save_path
        self.specs = specs
        # self.frames = ee_mesh_points.shape[0]
        # self.frames = len(ee_mesh)
        # self.interval = specs["interval"]
        self.frames = len(rep_positions)
        self.interval = 10 * specs["sample_interval"] / specs["time_scaler"]

        print('interval: ', self.interval)
        # Save variables as class attributes
        self.scenery_objs = scenery_objs
        self.dem_positions = dem_positions
        self.goal_obj   = goal_obj
        self.ee_meshes = ee_meshes
        self.ee_position = rep_positions
        self.theta = thetas

        # # Obtain norm values which will be animated
        # lin_vel_norms = np.linalg.norm(ee_velos[:, 0:3], axis=1)
        # ang_vel_norms = np.linalg.norm(ee_velos[:, 3:], axis=1).reshape(-1, 1)
        # self.vel_and_norm = np.insert(ee_velos, 3, lin_vel_norms, axis=1)
        # self.vel_and_norm = np.hstack((self.vel_and_norm, ang_vel_norms))

        # force_norms = np.linalg.norm(wrenches[:, 0:3], axis=1)
        # rot_norms = np.linalg.norm(wrenches[:, 3:], axis=1).reshape(-1, 1)
        # self.wrench_and_norm = np.insert(wrenches, 3, force_norms, axis=1)
        # self.wrench_and_norm = np.hstack((self.wrench_and_norm, rot_norms))

        # print(f"Creating animations using {self.frames} frames")


    def init_3D_scenery(self):
        """
        init_func for the 3D animation. Draws the static bg, ie axes labels/limits and the obstacles
        """
        self.ax_anim_sq.clear()
        self.ax_anim_sq.axes.set_xlabel('X')
        self.ax_anim_sq.axes.set_ylabel('Y')
        self.ax_anim_sq.axes.set_zlabel('Z')
        self.ax_anim_sq.axes.set_xlim3d(left   = self.specs["xlim3d"][0], right = self.specs["xlim3d"][1])
        self.ax_anim_sq.axes.set_ylim3d(bottom = self.specs["ylim3d"][0], top   = self.specs["ylim3d"][1])
        self.ax_anim_sq.axes.set_zlim3d(bottom = self.specs["zlim3d"][0], top   = self.specs["zlim3d"][1])

        if self.dem_positions is not None:
            self.ax_anim_sq.plot(self.dem_positions[:, 0], self.dem_positions[:, 1], self.dem_positions[:, 2])

        # Plot the obstacles on the axes
        for k in self.scenery_objs: 
            if k.startswith("obstacle_"):
                self.scenery_objs[k].plot_sq(self.ax_anim_sq, "red", "3D")
            elif k.startswith("workspace_"):
                self.scenery_objs[k].plot_sq(self.ax_anim_sq, "yellow", "3D", alpha=0.1)
            elif k.startswith('voxel_'):
                self.scenery_objs[k].plot_sq(self.ax_anim_sq, "yellow", "3D", alpha=0.1)

        # Plot the goal object
        if self.goal_obj is not None:
            self.goal_obj.plot_sq(self.ax_anim_sq, "blue", "3D", alpha=0.1)


    def update_Endeffector(self, i):
        """
        Updates frame for 3D animation. Used in FuncAnimation
        i: int | number of frames. Changed internally by FuncAnimation
        """
        step_start = time.time()

        # Remove plots from previous frames
        if i > 0:
            self.sf.remove()  # Remove the surface plot from previous frame
            self.closest_line.remove()  # Remove line from previous frame

        # Obtain current ee position
        ee_x = self.ee_position[i, 0]
        ee_y = self.ee_position[i, 1]
        ee_z = self.ee_position[i, 2]

        # Plot the ee mesh
        self.sf = self.ax_anim_sq.plot_surface(self.ee_meshes[i, 0], self.ee_meshes[i, 1], self.ee_meshes[i, 2],
                                               color="green", alpha=.2)

        # Plot the current ee position
        self.ax_anim_sq.plot(ee_x, ee_y, ee_z, color='green', marker='_', linewidth=1, linestyle='solid')

        # Plot the distance between the ee and closest object
        min_dist = sys.float_info.max
        closest_object_pos = 0
        for k in self.scenery_objs:
            if k.startswith("workspace_"):
                continue
            obj_pos = self.scenery_objs[k].get_pose()[0]
            dist = np.linalg.norm(self.ee_position[i] - obj_pos)
            if dist < min_dist:
                closest_object_pos = obj_pos
                min_dist = dist

        self.closest_line, = self.ax_anim_sq.plot([ee_x, closest_object_pos[0]], [ee_y, closest_object_pos[1]],
                                                  [ee_z, closest_object_pos[2]], color='red', marker='_',
                                                  linewidth=1, linestyle='solid')

        step_end = time.time()

        print('time: ', i*self.interval/1000, '[s] \t (step time: ', round(step_end-step_start,3), 's)')


    def animate_3d(self):
        """
        Function to create a 3d animation of the ee moving in the workspace
        """
        # Initialise figure and axes for the sq animation
        self.fig_anim_sq = plt.figure("3D animation")
        self.fig_anim_sq.set_size_inches(10.8, 7.2)
        self.ax_anim_sq = self.fig_anim_sq.add_subplot(111, projection='3d')

        _animation = FuncAnimation(fig       = self.fig_anim_sq, 
                                   func      = self.update_Endeffector,
                                   frames    = self.frames, 
                                   init_func = self.init_3D_scenery,
                                   interval  = self.interval,
                                   blit      = False)

        self.ax_anim_sq.view_init(elev=self.specs["view_angles"][0], azim=self.specs["view_angles"][1], roll=self.specs["view_angles"][2])

        if self.specs["save"] is not None:
            gif_path = self.save_path + "/3d_animation.gif"
            print(f"Saved sq animation in {self.save_path}")
            _animation.save(gif_path, dpi=100)

        if self.specs["plot"] is not None:
            plt.show()







    # def update_params(self, frame, param_type):
    #     """
    #     https://stackoverflow.com/questions/29832055/animated-subplots-using-matplotlib
    #     Animates the parameters
    #     Args:
    #           frame: int | changed internally by the funcAimation function
    #           psram_type: string | either velocity or wrench
    #     """
    #     x_data = np.linspace(0, self.frames, num=self.frames)

    #     # '''Set values for the velocity or wrench lines'''
    #     # if param_type == "velocity":
    #     #     for i in range(len(self.velocity_lines)):
    #     #         x = x_data[:frame]
    #     #         y = self.vel_and_norm[:frame, i]
    #     #         self.velocity_lines[i].set_data(x, y)

    #     # else:
    #     for i in range(len(self.wrench_lines)):
    #         x = x_data[:frame]
    #         y = self.wrench_and_norm[:frame, i]
    #         self.wrench_lines[i].set_data(x, y)


    # def animate_params(self):
    #     """
    #     Function to animate theta, wrenches, velocities and their norms
    #     """
    #     ''' Setup velocity figure '''
    #     # vel_fig, vel_ax = plt.subplots(nrows=2, ncols=4, constrained_layout=True)
    #     # vel_fig.suptitle('Velocities', fontsize=16)
    #     # vel_fig.set_size_inches(10.8, 7.2)
    #     # y_label = 4 * ["m/s"] + 4 * ["rad/s"]
    #     # x_label = "Step"
    #     # subplot_titles = ["Vx", "Vy", "Vz", "Vnorm", "ωx", "ωy", "ωz", "ωnorm"]
    #     # X_MAX = int(self.frames*1.1)  # no. frames corresponds to the number of steps

    #     # self.velocity_lines = []
    #     # for idx, a in enumerate(vel_ax.reshape(-1)):
    #     #     # Add empty lines, axes labels, titles and limits
    #     #     a.set_ylabel(y_label[idx])
    #     #     a.set_xlabel(x_label)
    #     #     a.title.set_text(subplot_titles[idx])
    #     #     a.set_xlim(-0.5, X_MAX)
    #     #     a.set_ylim((np.min(self.vel_and_norm[:, idx])*1.1)-0.1, (np.max(self.vel_and_norm[:, idx])*1.1)+0.1)
    #     #     temp_line, = a.plot([], [], lw=1, color='r')
    #     #     self.velocity_lines.append(temp_line)

    #     ''' Setup wrench figure '''
    #     wrench_fig, wrench_ax = plt.subplots(nrows=2, ncols=4, constrained_layout=True)
    #     wrench_fig.suptitle('Wrench', fontsize=16)
    #     wrench_fig.set_size_inches(10.8, 7.2)
    #     y_label = 4 * ["N"] + 4 * ["Nm"]
    #     x_label = "Step"
    #     subplot_titles = ["Fx", "Fy", "Fz", "Fnorm", "Tx", "Ty", "Tz", "Tnorm"]
    #     X_MAX = int(self.frames*1.1)  # no. frames corresponds to the number of steps

    #     self.wrench_lines = []
    #     for idx, a in enumerate(wrench_ax.reshape(-1)):
    #         # Add axes labels, titles and limits
    #         a.set_ylabel(y_label[idx])
    #         a.set_xlabel(x_label)
    #         a.title.set_text(subplot_titles[idx])
    #         a.set_xlim(-0.5, X_MAX)
    #         a.set_ylim((np.min(self.wrench_and_norm[:, idx])*1.1)-0.1, (np.max(self.wrench_and_norm[:, idx])*1.1)+0.1)
    #         temp_line, = a.plot([], [], lw=1, color='r')
    #         self.wrench_lines.append(temp_line)

    #     ''' Run the animation '''
    #     # vel_anim = FuncAnimation(vel_fig, self.update_params, fargs=("velocity",),
    #     #                          frames=self.frames, interval=self.interval, blit=False)
    #     wrench_anim = FuncAnimation(wrench_fig, self.update_params, fargs=("wrench",),
    #                                 frames=self.frames, interval=self.interval, blit=False)

    #     if self.specs["save"] is not None:
    #         vel_path = self.save_path + "/vel_animation.gif"
    #         wrench_path = self.save_path + "/wrench_animation.gif"
    #         print(f"Saved parameter animation in {self.save_path}")
    #         # vel_anim.save(vel_path, dpi=100)
    #         wrench_anim.save(wrench_path, dpi=100)

    #     if self.specs["plot"] is not None:
    #         plt.show()





# class Plot2D(object):
#     """
#     Class for plotting 2D graphs
#     """

#     def __init__(self, specs, ):














    # def open_pickle(pickle_path):
    #     """
    #     Utility function for deserialising a pickle file
    #     Args:
    #         pickle_path: String | Path to pickle file

    #     Returns:
    #         pickle_file: pickle file that was deserialised
    #     """
    #     with open(pickle_path, 'rb') as f:
    #         pickle_file = pickle.load(f)

    #     return pickle_file


    # def save_pickle(pickle_name, data):
    #     """
    #     Save the data as a pickle file

    #     Args:
    #         pickle_name: String | Filename including the path
    #         data: data to be saved as a pickle

    #     Returns:
    #     """
    #     if not pickle_name.endswith('.pickle'):
    #         pickle_name = pickle_name + '.pickle'

    #     with open(pickle_name, 'wb') as f:
    #         pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
