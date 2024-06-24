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

import math
import numpy as np

class SuperquadricObject():
    """
    Class for defining the properties of a superquadric object
    """
    
    def __init__(self, a=None, b=None, c=None, eps1=1.0, eps2=1.0, pose=None): #, obstacle_type=None):
        """
        The isopotential arising from an object expressed as a superquadric model depends on the relative position of the point in question.
        Generally, the isopotential is equal 1 at the contour of the object and increases as the point moves away from the object. Within the object, the isopotenial is less than 1.
        While superquadrics can be used to express any symmetric profile, the simplification of objects to surrounding objects is sufficient in most cases. 
        The approximation of a recangular object with known width, height and depth as an ellipsoid is embedded to the class and further shapes can be added as needed.

                                     /  /  xB_1  \ (2/eps2)    /  xB_2  \ (2/eps2) \ (eps2/eps1)     /  xB_3  \ (2/eps1)
            F(a,b,c,eps1,eps2,xB) =  |  |--------|          +  |--------|          |             +   |--------|
                                     \  \    a   /             \    b   /          /                 \    c   /

        Args:
            a        : float  | ellipsoid radius of the object corresponding to x-axis
            b        : float  | ellipsoid radius of the object corresponding to y-axis
            c        : float  | ellipsoid radius of the object corresponding to z-axis
            eps1,eps2: fload  | shape parameters of the superquadric
            pose     : tuple  | absolute pose of the object (xyz position as np.array, qxyz quaternion as a Quaternion)
            obstacle_type  : String | type of obstacle, either in or out. Specify "ee" if object is ee
        """

        self.a = a
        self.b = b
        self.c = c
        self.eps1 = eps1
        self.eps2 = eps2

        if not (pose is None):
            self.x_abs = pose[0]
            self.q_abs = pose[1]

        
    def update_scene(self, x_abs, q_abs, p_abs):
        """
        Setter for updating the object's pose and the point in question

        Args:
            x_abs: np array   | The vector x_abs is the position vector of the object's center with respect to the world frame        
            q_abs: Quaternion | The quaternion q_abs is the orientation of the object in question with respect to the world frame
            p_abs: np array   | The vector p_abs is the position vector of the point in question with respect to the world frame
        """
        self.x_abs = x_abs
        self.q_abs = q_abs
        self.p_abs = p_abs

        qw = q_abs.w
        qx = q_abs.x
        qy = q_abs.y
        qz = q_abs.z

        self.A = np.array([
            [
                qx**2 - qy**2 - qz**2 + qw**2,
                2 * (qx * qy + qz * qw),
                2 * (qx * qz - qy * qw)
            ],
            [
                2 * (qx * qy - qz * qw),
                -qx**2 + qy**2 - qz**2 + qw**2,
                2 * (qy * qz + qx * qw)
            ],
            [
                2 * (qx * qz + qy * qw),
                2 * (qy * qz - qx * qw),
                -qx**2 - qy**2 + qz**2 + qw**2
            ]
        ])

        x_rel = self.x_abs - self.p_abs

        self.H_1 = np.matmul(self.A[0], x_rel.reshape(-1, 1) / self.a) 
        self.H_2 = np.matmul(self.A[1], x_rel.reshape(-1, 1) / self.b) 
        self.H_3 = np.matmul(self.A[2], x_rel.reshape(-1, 1) / self.c) 

        # derivative of H_i with respect to the object's own coordinates
        self.nabla_H_1_obj = np.array([
            [ (qw**2 + qx**2 - qy**2 - qz**2) / self.a],
            [  2 * (qw * qz + qx * qy) / self.a],
            [- 2 * (qw * qy - qx * qz) / self.a],
            [- (  2 * qx * (p_abs[0] - x_abs[0]) + 2 * qy * (p_abs[1] - x_abs[1]) + 2 * qz * (p_abs[2] - x_abs[2]) ) / self.a],
            [  (  2 * qy * (p_abs[0] - x_abs[0]) - 2 * qx * (p_abs[1] - x_abs[1]) + 2 * qw * (p_abs[2] - x_abs[2]) ) / self.a],
            [- (- 2 * qz * (p_abs[0] - x_abs[0]) + 2 * qw * (p_abs[1] - x_abs[1]) + 2 * qx * (p_abs[2] - x_abs[2]) ) / self.a]
        ]) 

        self.nabla_H_2_obj = np.array([
            [- 2 * (qw * qz - qx * qy) / self.b],
            [ (qw**2 - qx**2 + qy**2 - qz**2) / self.b],
            [  2 * (qw * qx + qy * qz) / self.b],
            [- (  2 * qy * (p_abs[0] - x_abs[0]) - 2 * qx * (p_abs[1] - x_abs[1]) + 2 * qw * (p_abs[2] - x_abs[2]) ) / self.b],
            [- (  2 * qx * (p_abs[0] - x_abs[0]) + 2 * qy * (p_abs[1] - x_abs[1]) + 2 * qz * (p_abs[2] - x_abs[2]) ) / self.b],
            [  (  2 * qw * (p_abs[0] - x_abs[0]) + 2 * qz * (p_abs[1] - x_abs[1]) - 2 * qy * (p_abs[2] - x_abs[2]) ) / self.b]
        ]) 

        self.nabla_H_3_obj = np.array([
            [  2 * (qw * qy + qx * qz) / self.c],
            [- 2 * (qw * qx - qy * qz) / self.c],
            [ (qw**2 - qx**2 - qy**2 + qz**2) / self.c],
            [  (- 2 * qz * (p_abs[0] - x_abs[0]) + 2 * qw * (p_abs[1] - x_abs[1]) + 2 * qx * (p_abs[2] - x_abs[2]) ) / self.c],
            [- (  2 * qw * (p_abs[0] - x_abs[0]) + 2 * qz * (p_abs[1] - x_abs[1]) - 2 * qy * (p_abs[2] - x_abs[2]) ) / self.c],
            [- (  2 * qx * (p_abs[0] - x_abs[0]) + 2 * qy * (p_abs[1] - x_abs[1]) + 2 * qz * (p_abs[2] - x_abs[2]) ) / self.c]
        ]) 


        # derivative of H_i with respect to the point of interest's own coordinates
        self.nabla_H_1_pt = np.array([
            [- (qw**2 + qx**2 - qy**2 - qz**2) / self.a],
            [-  2 * (qw * qz + qx * qy) / self.a],
            [   2 * (qw * qy - qx * qz) / self.a],
            [0],
            [0],
            [0]
        ]) 

        self.nabla_H_2_pt = np.array([
            [   2 * (qw * qz - qx * qy) / self.b],
            [- (qw**2 - qx**2 + qy**2 - qz**2) / self.b],
            [-  2 * (qw * qx + qy * qz) / self.b],
            [0],
            [0],
            [0]
        ]) 

        self.nabla_H_3_pt = np.array([
            [-  2 * (qw * qy + qx * qz) / self.c],
            [   2 * (qw * qx - qy * qz) / self.c],
            [- (qw**2 - qx**2 - qy**2 + qz**2) / self.c],
            [0],
            [0],
            [0]
        ]) 

        # compute the inside-outside function and its derivatives
        self.__compute_F()
        self.__compute_nabla_F_obj()
        self.__compute_nabla_F_pt()


    def __compute_F(self):
        """
        Calculate the isopotential value of the superquadric object in canoncial form (xB)

                                     /  /  xB_1  \ (2/eps2)    /  xB_2  \ (2/eps2) \ (eps2/eps1)     /  xB_3  \ (2/eps1)
            F(a,b,c,eps1,eps2,xB) =  |  |--------|          +  |--------|          |             +   |--------|
                                     \  \    a   /             \    b   /          /                 \    c   /

        Returns:
            FaxB: float | isopotential value at the given relative position vector (xB)
        """
        self.F = ( (self.H_1)**(2/self.eps2) + (self.H_2)**(2/self.eps2) )**(self.eps2/self.eps1) + (self.H_3)**(2/self.eps1) 


    def __compute_nabla_F_obj(self):
        """
        Partial derivative of the inside-outside function at the current object's pose and point in question with respect to the object's own world coordinates 
        Returns: np array | gradient of the inside-outside function with respect to the object's own world coordinates
        """

        subterm = (self.H_1**(2/self.eps2) + self.H_2**(2/self.eps2))**(self.eps2/self.eps1 - 1)

        self.nabla_F_obj = np.array([
            (2*self.H_3**(2/self.eps1 - 1)*self.nabla_H_3_obj[0])/self.eps1 + (self.eps2*subterm*((2*self.H_2**(2/self.eps2 - 1)*self.nabla_H_2_obj[0])/self.eps2 + (2*self.H_1**(2/self.eps2 - 1)*self.nabla_H_1_obj[0])/self.eps2))/self.eps1,
            (2*self.H_3**(2/self.eps1 - 1)*self.nabla_H_3_obj[1])/self.eps1 + (self.eps2*subterm*((2*self.H_2**(2/self.eps2 - 1)*self.nabla_H_2_obj[1])/self.eps2 + (2*self.H_1**(2/self.eps2 - 1)*self.nabla_H_1_obj[1])/self.eps2))/self.eps1,
            (2*self.H_3**(2/self.eps1 - 1)*self.nabla_H_3_obj[2])/self.eps1 + (self.eps2*subterm*((2*self.H_2**(2/self.eps2 - 1)*self.nabla_H_2_obj[2])/self.eps2 + (2*self.H_1**(2/self.eps2 - 1)*self.nabla_H_1_obj[2])/self.eps2))/self.eps1,
            (2*self.H_3**(2/self.eps1 - 1)*self.nabla_H_3_obj[3])/self.eps1 + (self.eps2*subterm*((2*self.H_2**(2/self.eps2 - 1)*self.nabla_H_2_obj[3])/self.eps2 + (2*self.H_1**(2/self.eps2 - 1)*self.nabla_H_1_obj[3])/self.eps2))/self.eps1,
            (2*self.H_3**(2/self.eps1 - 1)*self.nabla_H_3_obj[4])/self.eps1 + (self.eps2*subterm*((2*self.H_2**(2/self.eps2 - 1)*self.nabla_H_2_obj[4])/self.eps2 + (2*self.H_1**(2/self.eps2 - 1)*self.nabla_H_1_obj[4])/self.eps2))/self.eps1,
            (2*self.H_3**(2/self.eps1 - 1)*self.nabla_H_3_obj[5])/self.eps1 + (self.eps2*subterm*((2*self.H_2**(2/self.eps2 - 1)*self.nabla_H_2_obj[5])/self.eps2 + (2*self.H_1**(2/self.eps2 - 1)*self.nabla_H_1_obj[5])/self.eps2))/self.eps1,
        ]) 


    def __compute_nabla_F_pt(self):
        """
        Partial derivative of the isopotential with respect to the relative position vector xB at the given point xB
        Returns: np array | gradient of the isopotential at the given point
        """
        
        subterm = (self.H_1**(2/self.eps2) + self.H_2**(2/self.eps2))**(self.eps2/self.eps1 - 1)

        self.nabla_F_pt = np.array([
            (2*self.H_3**(2/self.eps1 - 1)*self.nabla_H_3_pt[0])/self.eps1 + (self.eps2*subterm*((2*self.H_2**(2/self.eps2 - 1)*self.nabla_H_2_pt[0])/self.eps2 + (2*self.H_1**(2/self.eps2 - 1)*self.nabla_H_1_pt[0])/self.eps2))/self.eps1,
            (2*self.H_3**(2/self.eps1 - 1)*self.nabla_H_3_pt[1])/self.eps1 + (self.eps2*subterm*((2*self.H_2**(2/self.eps2 - 1)*self.nabla_H_2_pt[1])/self.eps2 + (2*self.H_1**(2/self.eps2 - 1)*self.nabla_H_1_pt[1])/self.eps2))/self.eps1,
            (2*self.H_3**(2/self.eps1 - 1)*self.nabla_H_3_pt[2])/self.eps1 + (self.eps2*subterm*((2*self.H_2**(2/self.eps2 - 1)*self.nabla_H_2_pt[2])/self.eps2 + (2*self.H_1**(2/self.eps2 - 1)*self.nabla_H_1_pt[2])/self.eps2))/self.eps1,
            [0],
            [0],
            [0],
        ]) 
        

    def get_eps1(self):
        return self.eps1
    

    def get_F(self):
        """
        Getter for the value of the inside-outside function at the given point computed with function "__compute_F"
        Returns: float | value of the inside-outside function at the given point 
        """
        return self.F
    

    def get_nabla_F_obj(self):
        """
        Getter for the isopotential's partial derivatives at the given point xB computed with function "__compute_nabla_F"
        Returns: np array | gradient of the isopotential at the given position vector (xB)
        """
        return self.nabla_F_obj
    

    def get_nabla_F_pt(self):
        """
        Getter for the isopotential's partial derivatives at the given point xB computed with function "__compute_nabla_F"
        Returns: np array | gradient of the isopotential at the given position vector (xB)
        """
        return self.nabla_F_pt

    def get_pose(self):
        """
        Getter for the object's pose
        Returns: tuple(np.array, Quaternion) | (xyz position, wxyz quaternion)
        """
        return (self.x_abs, self.q_abs)
    

    def plot_sq(self, ax, colour, plot_type='3D', alpha=0.2):
        """
        Plot
        """

        X, Y, Z = self.get_mesh()

        if plot_type == '2D':
            return ax.plot(X, Z, alpha=alpha, linewidth=.3, color=colour)
        elif plot_type == '3D':
            return ax.plot_surface(X, Y, Z, alpha=alpha, color=colour)
        elif plot_type == 'animate':
            return X, Y, Z
        else:
            raise 'Undefined plot type. Should be either animeate, 2D or 3D'


    def get_mesh(self, pos=None, quat=None):
        """
        Get expansion dimensions of object based on given pose
        """
        if pos is None:
            pos_x = self.x_abs[0]
            pos_y = self.x_abs[1]
            pos_z = self.x_abs[2]

            qw = self.q_abs.w
            qx = self.q_abs.x
            qy = self.q_abs.y
            qz = self.q_abs.z
        
        else:
            pos_x = pos[0]
            pos_y = pos[1]
            pos_z = pos[2]
            
            qw = quat[0]
            qx = quat[1]
            qy = quat[2]
            qz = quat[3]

        scos = lambda theta, eps: math.copysign(abs(math.cos(theta)) ** eps, math.cos(theta))
        ssin = lambda theta, eps: math.copysign(abs(math.sin(theta)) ** eps, math.sin(theta))

        WN = 41 # 40
        NN = 81 # 80

        w_array = np.linspace(-math.pi, math.pi, WN)
        n_array = np.linspace(-math.pi / 2, math.pi / 2, NN)
        w_mesh, n_mesh = np.meshgrid(w_array, n_array)

        X = np.zeros(w_mesh.shape)
        Y = np.zeros(w_mesh.shape)
        Z = np.zeros(w_mesh.shape)

        for (i, j), __ in np.ndenumerate(w_mesh):
            w = w_mesh[i, j]
            n = n_mesh[i, j]

            X[i, j] = self.a * scos(n, self.eps1) * scos(w, self.eps2)
            Y[i, j] = self.b * scos(n, self.eps1) * ssin(w, self.eps2)
            Z[i, j] = self.c * ssin(n, self.eps1)
        R = np.zeros((3, 3))

        R[0, 0] = 1 - 2*qy*qy - 2*qz*qz
        R[0, 1] = 2*qx*qy - 2*qz*qw
        R[0, 2] = 2*qx*qz + 2*qy*qw

        R[1, 0] = 2*qx*qy + 2*qz*qw
        R[1, 1] = 1 - 2*qx*qx - 2*qz*qz
        R[1, 2] = 2*qy*qz - 2*qx*qw

        R[2, 0] = 2*qx*qz - 2*qy*qw
        R[2, 1] = 2*qy*qz + 2*qx*qw
        R[2, 2] = 1 - 2*qx*qx - 2*qy*qy

        T = np.zeros((4, 4))
        T[0:3, 0:3] = R
        T[0:3, 3] = np.array([pos_x, pos_y, pos_z]).transpose()
        T[3, :] = np.array([[0, 0, 0, 1]])

        for (i, j), __ in np.ndenumerate(X):
            xp = np.array([[X[i, j], Y[i, j], Z[i, j], 1]]).transpose()
            xp = T @ xp
            X[i, j] = xp[0]
            Y[i, j] = xp[1]
            Z[i, j] = xp[2]

        return X, Y, Z 





class VolumetricDistance():
    """
    Estimates the distance between two superquadric objects' contours.
    Based on two initially given instances of superquadric objects, the class includes functions 
    to compute all attributes concerning the distance between the two objects.
    The class assumes one stationary object and one moving object (e.g., obstacle and end-effector) and
    can be extended to two moving objects in the future.
    """
    
    def __init__(self, SuperquadricObject_Dynamic, SuperquadricObject_Static, distance_type):
        """
        For two volumetric objects, the distance is approximated using the approach for rigid body radial euclidean distance defined by [Badawy2007] 
        
        The rbr Euclidean distance takes into consideration the possible difference between the obstacle and 
        manoeuvring object shapes, sizes and orientations as the inside-outside function, F, is calculated for each object.
        
        Args:
            SuperquadricObject_Instances: SuperquadricObject | Objects expressed as superquadric models, Instance1 is the moving object and Instance2 is the stationary object
        """

        self.sqObj_dyn  = SuperquadricObject_Dynamic  # dynamic object, referred to as end-effector
        self.sqObj_stat = SuperquadricObject_Static  # static object, referred to as obstacle
        self.distance_type = distance_type

    def update_scene(self, x_dyn_abs, q_dyn_abs, x_stat_abs, q_stat_abs):
        self.r_dyn_stat = np.linalg.norm(x_dyn_abs - x_stat_abs)

        self.nabla_r_dyn_stat = np.array([
            [ (x_dyn_abs[0] - x_stat_abs[0]) / self.r_dyn_stat ],
            [ (x_dyn_abs[1] - x_stat_abs[1]) / self.r_dyn_stat ],
            [ (x_dyn_abs[2] - x_stat_abs[2]) / self.r_dyn_stat ],
            [0],
            [0],
            [0],
        ])

        self.sqObj_dyn.update_scene(x_dyn_abs, q_dyn_abs, x_stat_abs)

        self.sqObj_stat.update_scene(x_stat_abs, q_stat_abs, x_dyn_abs)

        if self.distance_type == "outside":
            self.__compute_outside_distance()
            self.__compute_nabla_outside_distance()
        elif self.distance_type == "inside":
            self.__compute_inside_distance()
            self.__compute_nabla_inside_distance()
        elif self.distance_type == "attractive":
            self.__compute_attractive_distance()
            self.__compute_nabla_attractive_distance()



    def __compute_outside_distance(self):
        """
        The rigid body radial euclidean distance for two objects outside of each other is calculated as follows:

        d_12 = r_12 - r_1 - r_2 (2 represents the obstacle object)

        ==>

        d_12 = | r_12 | * (1 - F_1 ^ (-eps1_1/2) - F_2 ^ (-eps1_2/2))
        
        with r_12 being the translational distance between the i-th and j-th objects' centres 
        and F_i being the inside-outside function of the i-th object with the j-th object's centre used as point of interest.
        """
        self.outside_distance = self.r_dyn_stat * (1 - self.sqObj_dyn.get_F()**(-self.sqObj_dyn.get_eps1()/2) - self.sqObj_stat.get_F()**(-self.sqObj_stat.get_eps1()/2)) 
    

    def __compute_nabla_outside_distance(self):
        """
        Compute partial derivative of rigid body radial euclidean outside distance
        """
        self.nabla_outside_distance = (
                ( self.nabla_r_dyn_stat * (1 - self.sqObj_dyn.get_F()**(-self.sqObj_dyn.get_eps1()/2) - self.sqObj_stat.get_F()**(-self.sqObj_stat.get_eps1()/2)))
                + self.r_dyn_stat * ( (self.sqObj_dyn.get_eps1()/2) * self.sqObj_dyn.get_F()**(-self.sqObj_dyn.get_eps1()/2 - 1) * self.sqObj_dyn.get_nabla_F_obj() 
                                  + (self.sqObj_stat.get_eps1()/2) * self.sqObj_stat.get_F()**(-self.sqObj_stat.get_eps1()/2 - 1) * self.sqObj_stat.get_nabla_F_pt()) 
        ) 


    def __compute_inside_distance(self):
        """
        The rigid body radial euclidean distance between two objects, while one object (1) is inside the other's object (2) is calculated as follows:

        d_12 = r_2 - r_1 - r_12 (2 represents the workspace object)

        ==>

        d_12 = | r_12 | * (F_2 ^ (-eps1_2/2) - F_1 ^ (-eps1_1/2) - 1)
        
        with r_12 being the translational distance between the i-th and j-th objects' centres 
        and F_i being the inside-outside function of the i-th object with the j-th object's centre used as point of interest.
        """
        self.inside_distance = self.r_dyn_stat * ( self.sqObj_stat.get_F()**(-self.sqObj_stat.get_eps1()/2) - self.sqObj_dyn.get_F()**(-self.sqObj_dyn.get_eps1()/2) - 1 ) 
    

    def __compute_nabla_inside_distance(self):
        """
        Compute partial derivative of rigid body radial euclidean inside distance
        """
        self.nabla_inside_distance = (
                ( self.nabla_r_dyn_stat * (self.sqObj_stat.get_F()**(-self.sqObj_stat.get_eps1()/2) - self.sqObj_dyn.get_F()**(-self.sqObj_dyn.get_eps1()/2) - 1) )
                + self.r_dyn_stat * ( ( - self.sqObj_stat.get_eps1()/2) * self.sqObj_stat.get_F()**(-self.sqObj_stat.get_eps1()/2 - 1) * self.sqObj_stat.get_nabla_F_pt()
                                  + (self.sqObj_dyn.get_eps1()/2) * self.sqObj_dyn.get_F()**(-self.sqObj_dyn.get_eps1()/2 - 1) * self.sqObj_dyn.get_nabla_F_obj() )
        ) 


    def __compute_attractive_distance(self):
        """
        The rigid body radial euclidean distance for attraction of object 1 to object 2 is expressed as the distance to the object 2 opposite side and is calculated as follows :

        d_12 = r_12 - r_1 + r_2 (2 represents the attractive goal object)

        ==>

        d_12 = | r_12 | * (1 - F_1 ^ (-eps1_1/2) + F_2 ^ (-eps1_2/2))
        
        with r_12 being the translational distance between the i-th and j-th objects' centres 
        and F_i being the inside-outside function of the i-th object with the j-th object's centre used as point of interest.
        """
        self.attractive_distance = self.r_dyn_stat * (1 - self.sqObj_dyn.get_F()**(-self.sqObj_dyn.get_eps1()/2) + self.sqObj_stat.get_F()**(-self.sqObj_stat.get_eps1()/2)) 
    

    def __compute_nabla_attractive_distance(self):
        """
        Compute partial derivative of rigid body radial euclidean attractive distance
        """
        self.nabla_attractive_distance = (
                ( self.nabla_r_dyn_stat * (1 - self.sqObj_dyn.get_F()**(-self.sqObj_dyn.get_eps1()/2) + self.sqObj_stat.get_F()**(-self.sqObj_stat.get_eps1()/2)))
                + self.r_dyn_stat * ( (self.sqObj_dyn.get_eps1()/2) * self.sqObj_dyn.get_F()**(-self.sqObj_dyn.get_eps1()/2 - 1) * self.sqObj_dyn.get_nabla_F_obj() 
                                  - (self.sqObj_stat.get_eps1()/2) * self.sqObj_stat.get_F()**(-self.sqObj_stat.get_eps1()/2 - 1) * self.sqObj_stat.get_nabla_F_pt()) 
        )




    def _get_outside_distance(self):
        """
        Getter for the rbr-Euclidean Distance value between the contour of the given objects 
        at the given absolute position vectors computed with function "__compute_rbrEdistance"
        Returns: float | rbr-Euclidean Distance value between the contour of the given objects 
                         at the given absolute position vectors (pos_dyn, pos_stat)
        """
        return self.outside_distance


    def _get_nabla_outside_distance(self):
        """
        Getter for the rbr-Euclidean Distance's partial derivatives at the given absolute position
        vectors computed with function "__compute_nabla_rbrEdistance"
        Returns: np array | gradient of the rbr-Euclidean Distance at the given absolute position vectors (pos_dyn, pos_stat)
        """
        return self.nabla_outside_distance
    

    def _get_inside_distance(self):
        """
        Getter for the rbr-Euclidean Distance value between the contour of the given objects 
        at the given absolute position vectors computed with function "__compute_rbrEdistance"
        Returns: float | rbr-Euclidean Distance value between the contour of the given objects 
                         at the given absolute position vectors (pos_dyn, pos_stat)
        """
        return self.inside_distance


    def _get_nabla_inside_distance(self):
        """
        Getter for the rbr-Euclidean Distance's partial derivatives at the given absolute position
        vectors computed with function "__compute_nabla_rbrEdistance"
        Returns: np array | gradient of the rbr-Euclidean Distance at the given absolute position vectors (pos_dyn, pos_stat)
        """
        return self.nabla_inside_distance
    

    def _get_attractive_distance(self):
        """
        Getter for the rbr-Euclidean Distance value between the contour of the given objects 
        at the given absolute position vectors computed with function "__compute_rbrEdistance"
        Returns: float | rbr-Euclidean Distance value between the contour of the given objects 
                         at the given absolute position vectors (pos_dyn, pos_stat)
        """
        return self.attractive_distance
    

    def _get_nabla_attractive_distance(self):
        """
        Getter for the rbr-Euclidean Distance's partial derivatives at the given absolute position
        vectors computed with function "__compute_nabla_rbrEdistance"
        Returns: np array | gradient of the rbr-Euclidean Distance at the given absolute position vectors (pos_dyn, pos_stat)
        """
        return self.nabla_attractive_distance


    def get_distance(self):
        """
        Getter for distance.

        Returns: float | inside distance if workspace boundary, else obtains outside distance for normal obstacles
        """
        if self.distance_type == "outside":
            return self._get_outside_distance()
        elif self.distance_type == "inside":
            return self._get_inside_distance()
        elif self.distance_type == "attractive":
            return self._get_attractive_distance()
        else:
            print(self.distance_type)
            raise "sq obstacle type is incorrect"


    def get_nabla_distance(self):
        """
        Getter for distance.

        Returns: float | inside distance if workspace boundary, else obtains outside distance for normal obstacles
        """
        if self.distance_type == "outside":
            return self._get_nabla_outside_distance()
        elif self.distance_type == "inside":
            return self._get_nabla_inside_distance()
        elif self.distance_type == "attractive":
            return self._get_nabla_attractive_distance()
        else:
            print(self.distance_type)
            raise "sq distance type is incorrect"
        
    def get_dist_centres(self):
        return self.r_dyn_stat
    
    def get_nabla_dist_centres(self):
        return self.nabla_r_dyn_stat