#!/usr/bin/python3

import sys
import yaml
import math
import os.path
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from matplotlib.colors import Normalize

from src.cdmps import CartesianDMPs
from src.superquadrics import SuperquadricObject, VolumetricDistance
from src.visuals import Animation3D


if len(sys.argv) != 2:
    raise "Please specify a test directory"
TEST_DIR = sys.argv[1]
PATH = f"demos/{TEST_DIR}"
if not os.path.exists(PATH):
    print(f"Path is {PATH}. Are you running from the scripts director?")
    raise "Bitte überprüfen Sie Ihren Dateipfad"

with open(f"{PATH}/test_values.yaml", "r") as f:
    test_values = yaml.safe_load(f)



if __name__ == "__main__":

    ''' --------------- DEMONSTRATION --------------- '''
    
    dem_path = f"{PATH}/demo_trajectory.txt"
    # dem_path = "demos/exp1_/demo_trajectory.txt"
    data = np.loadtxt(dem_path, delimiter=',', skiprows=1)
    
    with open(f"{PATH}/test_values.yaml", "r") as f:
        test_values = yaml.safe_load(f)

    with open(dem_path, 'r') as file:
        column_names = file.readline().rstrip().split(',')

    dem_time   = data[:, column_names.index("Time")]
    dem_pos    = data[:, [column_names.index("PosX"), column_names.index("PosY"), column_names.index("PosZ")]]
    dem_quat   = data[:, [column_names.index("QuatW"), column_names.index("QuatX"), column_names.index("QuatY"), column_names.index("QuatZ")]]


    ''' --------------- SETUP --------------- '''

    # # # ENDEFFECTOR
    
    a_dyn_val = test_values["endeffector"]["a_val"]
    b_dyn_val = test_values["endeffector"]["b_val"]
    c_dyn_val = test_values["endeffector"]["c_val"]
    eps1_dyn_val = test_values["endeffector"]["eps1_val"]
    eps2_dyn_val = test_values["endeffector"]["eps2_val"]
    sqObj_dyn = SuperquadricObject(a=a_dyn_val, b=b_dyn_val, c=c_dyn_val, 
                                  eps1=eps1_dyn_val, eps2=eps2_dyn_val)


    # # # ATTRACTIVE GOAL 

    if 'goal' in test_values:
        a_goal_val = test_values['goal']['a_val']
        b_goal_val = test_values['goal']['b_val']
        c_goal_val = test_values['goal']['c_val']
        eps1_goal_val = test_values['goal']['eps1_val']
        eps2_goal_val = test_values['goal']['eps2_val']

        if 'x_val' in test_values['goal']:
            x_goal_val = test_values['goal']['x_val']
            y_goal_val = test_values['goal']['y_val']
            z_goal_val = test_values['goal']['z_val']
            q_goal_val = Quaternion(test_values['goal']['qw_val'], test_values['goal']['qx_val'],
                                    test_values['goal']['qy_val'], test_values['goal']['qz_val'])
        else:
            x_goal_val = dem_pos[-1, 0]
            y_goal_val = dem_pos[-1, 1]
            z_goal_val = dem_pos[-1, 2]
            q_goal_val = Quaternion(dem_quat[-1, :])  # Quaternion(w, x, y, z)
        
        sqObj_goal = SuperquadricObject(a=a_goal_val, b=b_goal_val, c=c_goal_val, 
                                    eps1=eps1_goal_val, eps2=eps2_goal_val,
                                    pose=(np.array([x_goal_val, y_goal_val, z_goal_val]), q_goal_val))

        attractive_distance = VolumetricDistance(sqObj_dyn, sqObj_goal, distance_type='attractive')


    # # # SCENERY (OBSTACLES AND WORKSPACES)

    scenery_keys = [key for key in test_values.keys() if key.startswith('obstacle_') or key.startswith('workspace_')]

    scenery_objects = {key: None for key in scenery_keys}
    scenery_distances = {key: None for key in scenery_keys}

    for key in scenery_keys:

        a_stat_val    = test_values[key]["a_val"]
        b_stat_val    = test_values[key]["b_val"]
        c_stat_val    = test_values[key]["c_val"]
        eps1_stat_val = test_values[key]["eps1_val"]
        eps2_stat_val = test_values[key]["eps2_val"]

        x_stat_val = test_values[key]["x_val"]
        y_stat_val = test_values[key]["y_val"]
        z_stat_val = test_values[key]["z_val"]
        q_stat_val = Quaternion(test_values[key]["qw_val"], test_values[key]["qx_val"],
                               test_values[key]["qy_val"], test_values[key]["qz_val"])  # Quaternion(w, x, y, z)
        
        scenery_objects[key] = SuperquadricObject(a=a_stat_val, b=b_stat_val, c=c_stat_val,
                                                eps1=eps1_stat_val, eps2=eps2_stat_val,
                                                pose=(np.array([x_stat_val, y_stat_val, z_stat_val]), q_stat_val))
        
        if key.startswith('obstacle_'):
            scenery_distances[key] = VolumetricDistance(sqObj_dyn, scenery_objects[key], distance_type="outside")
        elif key.startswith('workspace_'):
            scenery_distances[key] = VolumetricDistance(sqObj_dyn, scenery_objects[key], distance_type="inside")        
        
    # print(scenery_keys)


    # # # VISUALS

    errors = np.empty([1, 1])
    wrenches = np.zeros([1,6])
    attractive_wrenches = np.zeros([1,6])
    distances = np.empty([1,len(scenery_keys)])
    thetas_lin = np.empty([1,len(scenery_keys)])
    thetas_ang = np.empty([1,len(scenery_keys)])
    velocities = np.empty([1,6])


    ''' --------------- DYNAMIC MOVEMENT PRIMITIVES --------------- '''

    alpha_s    = test_values['cdmp']['alpha_s']
    k_gain     = test_values['cdmp']['k_gain']
    rbfs_pSec  = test_values['cdmp']['rbfs_pSec']
    tau_scaler = test_values['cdmp']['tau_scaler']
    max_steps  = test_values['cdmp']['max_steps']
    tolerance  = test_values['cdmp']['tolerance']

    cdmp = CartesianDMPs()

    cdmp.load_demo(filename   = TEST_DIR,
                   dem_time   = dem_time,
                   dem_pos    = dem_pos,
                   dem_quat   = dem_quat)

    cdmp.learn_cdmp(alpha_s   = alpha_s,
                    k_gain    = k_gain,
                    rbfs_pSec = rbfs_pSec)

    cdmp.init_reproduction(tau_scaler = tau_scaler)

    curr_error = np.linalg.norm(cdmp.rep_Pgoal - cdmp.rep_Pstart)
    curr_s = cdmp.rep_cs.s_step

    while ((curr_error > tolerance) and (cdmp.s_stepnum <= max_steps)):

        if cdmp.s_stepnum % 100 == 0:
            print(f"Step: {cdmp.s_stepnum}")

        curr_s      = cdmp.rep_cs.phase_step(curr_s)
        curr_pos    = cdmp.rep_pos[-1, :]
        curr_linVel = cdmp.rep_linVel[-1, :]
        curr_edq    = cdmp.rep_edq[-1, :]
        curr_vel    = np.append(curr_linVel, curr_edq).reshape((6,1))

        curr_wrenches = np.zeros((6, 1))

        curr_quat   = Quaternion(cdmp.rep_quat[-1, :])
        curr_repulsive_force = np.zeros((3, 1))
        curr_repulsive_torque = np.zeros((3, 1))
        curr_distances = np.zeros((len(scenery_keys)))
        curr_thetas = np.zeros((len(scenery_keys)))


        ''' --------------- ATTRACTIVE WRENCH --------------- '''

        if 'goal' in test_values:

            attractive_distance.update_scene(x_dyn_abs=curr_pos,
                                            q_dyn_abs=curr_quat,
                                            x_stat_abs=sqObj_goal.get_pose()[0],
                                            q_stat_abs=sqObj_goal.get_pose()[1])
            
            Kappa_FT = [test_values['goal']['kappa_FT'][0]] * 3 + [test_values['goal']['kappa_FT'][1]] * 3
            Kappa_FT = np.array(Kappa_FT).reshape((6, 1))

            if attractive_distance.get_dist_centres() < test_values['goal']['c0']:
                attractive_wrench = - 2 * Kappa_FT * attractive_distance.get_distance() * attractive_distance.get_nabla_distance()  
                
            else:
                attractive_wrench = np.zeros((6, 1))
        else:
            attractive_wrench = np.zeros((6, 1))

        attractive_wrenches = np.append(attractive_wrenches, attractive_wrench.reshape(1, 6), axis=0)


        ''' --------------- REPULSIVE WRENCHES --------------- '''

        for key_idx, key in enumerate(scenery_keys):
            
            beta_val    = test_values[key]["beta_val"]
            eta_val     = test_values[key]["eta_val"]
            Lambda_FT  = [test_values[key]['lambda_FT'][0]] * 3 + [test_values[key]['lambda_FT'][1]] * 3 
            Lambda_FT  = np.array(Lambda_FT).reshape((6, 1))
            scenery_distances[key].update_scene(x_dyn_abs=curr_pos, q_dyn_abs=curr_quat,
                                                x_stat_abs=scenery_objects[key].get_pose()[0],
                                                q_stat_abs=scenery_objects[key].get_pose()[1])
            
            curr_distances[key_idx] = scenery_distances[key].get_distance()
            
            curr_nabla_distance = np.asarray(scenery_distances[key].get_nabla_distance()).reshape((6,1))

            cos_theta = np.dot(np.squeeze(curr_nabla_distance), curr_vel) / (np.linalg.norm(curr_nabla_distance) * np.linalg.norm(curr_vel) + sys.float_info.min)
            theta = math.acos(cos_theta)

            repulsive_wrench = np.zeros((6, 1))

            if math.pi / 2 < theta <= math.pi:
                repulsive_wrench = ((-cos_theta) ** (beta_val) * Lambda_FT * eta_val * np.linalg.norm(curr_vel) /
                                scenery_distances[key].get_distance() ** (eta_val + 1) * curr_nabla_distance)
                        
            curr_wrenches += repulsive_wrench

        next_pos, next_quat = cdmp.rollout_step(curr_s_step=curr_s,
                                                ext_force = curr_wrenches[0:3],
                                                ext_torque = curr_wrenches[3:6],
                                                ext_linVel = attractive_wrench[0:3],
                                                ext_angVel = attractive_wrench[3:6])

        curr_vel    = np.append(curr_linVel, curr_edq)

        curr_error = np.linalg.norm(cdmp.rep_Pgoal - next_pos)

        velocities = np.append(velocities, curr_vel.reshape(1, 6), axis=0)
        wrenches = np.append(wrenches, curr_wrenches.reshape(1, 6), axis=0)
        errors   = np.append(errors, curr_error.reshape((1,1)), axis=0)
        distances = np.append(distances, curr_distances.reshape((1,len(scenery_keys))), axis=0)

    print(f"Final Step count: {cdmp.s_stepnum}")




    # # # # # # # # # VISUALISATION # # # # # # # # # 


    ''' --------------- EXPERIMENT 1 PLOTS --------------- '''

    if 'exp1_' in TEST_DIR:

        # # SETUP
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(cdmp.dem_pos[:,0], cdmp.dem_pos[:,1], cdmp.dem_pos[:,2], 'r')
        for key in scenery_objects:
            X, Y, Z = scenery_objects[key].get_mesh()
            # if key includes 'obstacle' --> red, else --> green
            if key.startswith('obstacle_'):
                color = 'red'
            else:
                color = 'yellow'
            ax.plot_surface(X, Y, Z, color=color, alpha=0.1)
            ax.plot(scenery_objects[key].get_pose()[0][0], scenery_objects[key].get_pose()[0][1], scenery_objects[key].get_pose()[0][2], color=color, marker='o')
        if 'goal' in test_values:
            X, Y, Z = sqObj_goal.get_mesh()
            ax.plot_surface(X, Y, Z, color='g', alpha=0.1)
            ax.plot(sqObj_goal.get_pose()[0][0], sqObj_goal.get_pose()[0][1], sqObj_goal.get_pose()[0][2], 'go')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.z lim    
        ax.set_zlim(-0.5, 0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.view_init(elev=22, azim=-109, roll=0)
        plt.show()
        
        fig = plt.figure()


        plt.plot(cdmp.dem_pos[:,0], cdmp.dem_pos[:,2], 'r')
        plt.plot(cdmp.rep_pos[:,0], cdmp.rep_pos[:,2], 'b')


        for key in scenery_objects:
            X, Y, Z = scenery_objects[key].get_mesh()
            plt.plot(X[:,1], Z[:,1], 'r')
            plt.plot(X[:,X.shape[1] // 2], Z[:,X.shape[1] // 2], 'r')
            plt.plot(scenery_objects[key].get_pose()[0][0], scenery_objects[key].get_pose()[0][2], 'ro')

        # # Frames equally distributed in time 
        # frames = np.arange(0, len(cdmp.rep_pos), 50)
        # print(frames)

        # Frames equally distributed in x
        x_dist = 1.0 / 15.0
        target_x_values = np.arange(0, 1.0 + x_dist, x_dist)
        frames = np.argmin(np.abs(cdmp.rep_pos[:, 0][:, None] - target_x_values), axis=0)
        print(frames)

        # Colormap
        z = np.linspace(0, 1, len(frames))  # Continuous variable for color
        cmap = plt.cm.viridis

        # Normalize the continuous variable to the range [0, 1]
        norm = Normalize(vmin=min(z), vmax=max(z))
        
        for count, frame in enumerate(frames):
            X, Y, Z = sqObj_dyn.get_mesh(cdmp.rep_pos[frame, :], cdmp.rep_quat[frame, :])
            # plt.plot(X, Z, 'b', alpha=0.1)
            plt.plot(X[:,1], Z[:,1], color=cmap(norm(z[len(frames)-1-count])))
            plt.plot(X[:,X.shape[1] // 2], Z[:,X.shape[1] // 2], color=cmap(norm(z[len(frames)-1-count])))
            plt.plot(cdmp.rep_pos[frame, 0], cdmp.rep_pos[frame, 2], 'o', color=cmap(norm(z[len(frames)-1-count])))

        # set window of plot to equal aspect ratio
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xlim(-0.1, 1.1)
        plt.show()


    ''' --------------- EXPERIMENT 2 PLOTS --------------- '''

    if 'exp2_' in TEST_DIR:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(cdmp.dem_pos[:,0], cdmp.dem_pos[:,1], cdmp.dem_pos[:,2], 'r')
        for key in scenery_objects:
            X, Y, Z = scenery_objects[key].get_mesh()
            # if key includes 'obstacle' --> red, else --> green
            if key.startswith('obstacle_'):
                color = 'red'
            else:
                color = 'yellow'
            ax.plot_surface(X, Y, Z, color=color, alpha=0.1)
            ax.plot(scenery_objects[key].get_pose()[0][0], scenery_objects[key].get_pose()[0][1], scenery_objects[key].get_pose()[0][2], color=color, marker='o')
        if 'goal' in test_values:
            X, Y, Z = sqObj_goal.get_mesh()
            ax.plot_surface(X, Y, Z, color='g', alpha=0.1)
            ax.plot(sqObj_goal.get_pose()[0][0], sqObj_goal.get_pose()[0][1], sqObj_goal.get_pose()[0][2], 'go')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal', adjustable='box')
        ax.view_init(elev=22, azim=-109, roll=0)
        plt.show()


        frames = [677, 997, 1246, 2508]
        print(frames)

        fig, axs = plt.subplots(1, 2)
        axs[0].plot(wrenches[:,0], label='Fx')
        axs[0].plot(wrenches[:,1], label='Fy')
        axs[0].plot(wrenches[:,2], label='Fz')
        for frame in frames:
            axs[0].axvline(x=frame, color='k', linestyle='--')
        axs[0].set_title('Repulsive Force')
        axs[0].legend()
        axs[0].set_xlim(0, 2508)
        axs[1].plot(wrenches[:,3], label='Tx')
        axs[1].plot(wrenches[:,4], label='Ty')
        axs[1].plot(wrenches[:,5], label='Tz')
        for frame in frames:
            axs[1].axvline(x=frame, color='k', linestyle='--')
        axs[1].set_title('Repulsive Torque')
        axs[1].legend()
        axs[1].set_xlim(0, 2508)
        fig.set_size_inches(15, 5)
        plt.show()


        # same for attractive_wrenches 
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(attractive_wrenches[:,0], label='Fx')
        axs[0].plot(attractive_wrenches[:,1], label='Fy')
        axs[0].plot(attractive_wrenches[:,2], label='Fz')
        for frame in frames:
            axs[0].axvline(x=frame, color='k', linestyle='--')
        axs[0].set_title('Attractive Force')
        axs[0].legend()
        axs[0].set_xlim(0, 2508)
        axs[1].plot(attractive_wrenches[:,3], label='Tx')
        axs[1].plot(attractive_wrenches[:,4], label='Ty')
        axs[1].plot(attractive_wrenches[:,5], label='Tz')
        for frame in frames:
            axs[1].axvline(x=frame, color='k', linestyle='--')
        axs[1].set_title('Attractive Torque')
        axs[1].legend()
        axs[1].set_xlim(0, 2508)

        fig.set_size_inches(15, 5)

        plt.show()

        fig = plt.figure()

        num_3d_subplots = len(frames)

        for count, frame in enumerate(frames):

            subplot_index = count + 1

            ax = fig.add_subplot(2, num_3d_subplots, subplot_index, projection='3d')

            for key in scenery_objects:
                X, Y, Z = scenery_objects[key].get_mesh()
                # if key includes 'obstacle' --> red, else --> green
                if key.startswith('obstacle_'):
                    color = 'red'
                else:
                    color = 'yellow'
                ax.plot_surface(X, Y, Z, color=color, alpha=0.1)
                ax.plot(scenery_objects[key].get_pose()[0][0], scenery_objects[key].get_pose()[0][1], scenery_objects[key].get_pose()[0][2], color=color, marker='o')

            if 'goal' in test_values:
                X, Y, Z = sqObj_goal.get_mesh()
                ax.plot_surface(X, Y, Z, color='g', alpha=0.1)
                ax.plot(sqObj_goal.get_pose()[0][0], sqObj_goal.get_pose()[0][1], sqObj_goal.get_pose()[0][2], 'go')

            X, Y, Z = sqObj_dyn.get_mesh(cdmp.rep_pos[frame, :], cdmp.rep_quat[frame, :])
            ax.plot_surface(X, Y, Z, color='b', alpha=0.1)
            ax.plot(cdmp.rep_pos[:frame,0], cdmp.rep_pos[:frame,1], cdmp.rep_pos[:frame,2], 'b')
            ax.plot(cdmp.rep_pos[frame, 0], cdmp.rep_pos[frame, 1], cdmp.rep_pos[frame, 2], 'bo')

            ax.plot(cdmp.dem_pos[:,0], cdmp.dem_pos[:,1], cdmp.dem_pos[:,2], 'r')
    
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # set window of plot to equal aspect ratio
            ax.set_aspect('equal', adjustable='box')

            # set axis limits
            ax.set_xlim(test_values['animation']['xlim3d'])
            ax.set_ylim(test_values['animation']['ylim3d'])
            ax.set_zlim(test_values['animation']['zlim3d'])

            ax.view_init(elev=0, azim=-90, roll=0)

            # SECOND ROW !!! 

            ax = fig.add_subplot(2, num_3d_subplots, subplot_index+len(frames), projection='3d')

            for key in scenery_objects:
                X, Y, Z = scenery_objects[key].get_mesh()
                # if key includes 'obstacle' --> red, else --> green
                if key.startswith('obstacle_'):
                    color = 'red'
                else:
                    color = 'yellow'
                ax.plot_surface(X, Y, Z, color=color, alpha=0.1)
                ax.plot(scenery_objects[key].get_pose()[0][0], scenery_objects[key].get_pose()[0][1], scenery_objects[key].get_pose()[0][2], color=color, marker='o')

            if 'goal' in test_values:
                X, Y, Z = sqObj_goal.get_mesh()
                ax.plot_surface(X, Y, Z, color='g', alpha=0.1)
                ax.plot(sqObj_goal.get_pose()[0][0], sqObj_goal.get_pose()[0][1], sqObj_goal.get_pose()[0][2], 'go')


            X, Y, Z = sqObj_dyn.get_mesh(cdmp.rep_pos[frame, :], cdmp.rep_quat[frame, :])
            ax.plot_surface(X, Y, Z, color='b', alpha=0.1)
            ax.plot(cdmp.rep_pos[:frame,0], cdmp.rep_pos[:frame,1], cdmp.rep_pos[:frame,2], 'b')
            ax.plot(cdmp.rep_pos[frame, 0], cdmp.rep_pos[frame, 1], cdmp.rep_pos[frame, 2], 'bo')

            ax.plot(cdmp.dem_pos[:,0], cdmp.dem_pos[:,1], cdmp.dem_pos[:,2], 'r')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # set window of plot to equal aspect ratio
            ax.set_aspect('equal', adjustable='box')

            # set axis limits
            ax.set_xlim(test_values['animation']['xlim3d'])
            ax.set_ylim(test_values['animation']['ylim3d'])
            ax.set_zlim(test_values['animation']['zlim3d'])

            ax.view_init(elev=90, azim=-90, roll=0)

        plt.show()


    ''' --------------- 3D ANIMATION --------------- '''

    ee_meshes_sampled = []

    sample_interval = test_values['animation']["sample_interval"]
    
    print(f"total number of poses: {len(cdmp.rep_pos)}. These positions will now be sampled...")

    anim_frames = np.arange(0, len(cdmp.rep_pos), sample_interval)

    rep_positions_sampled = cdmp.rep_pos[anim_frames, :]

    # print(anim_frames)

    for idx_count, idx in enumerate(anim_frames):
        ee_meshes_sampled.append(sqObj_dyn.get_mesh(cdmp.rep_pos[idx, :], cdmp.rep_quat[idx, :]))

    ee_meshes_sampled = np.array(ee_meshes_sampled)

    if 'goal' not in test_values:
        sqObj_goal = None

    _ani = Animation3D(specs          = test_values["animation"],
                        save_path     = PATH,
                        scenery_objs  = scenery_objects,
                        goal_obj      = sqObj_goal,
                        ee_meshes     = ee_meshes_sampled,
                        dem_positions = cdmp.dem_pos,
                        rep_positions = rep_positions_sampled
                        )
    
    _ani.animate_3d()