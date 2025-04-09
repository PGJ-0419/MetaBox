"""
=========================================================
File Name: uav.py
Author: Sijie Ma (GitHub: Feather_red)
Date: 2025-03-07
=========================================================

Description:
This script implements UAV path planning based on the paper:
"Benchmarking Global Optimization Techniques for Unmanned Aerial Vehicle Path Planning."
It provides a Python migration of a MATLAB implementation.

Features:
- Implements UAV path planning in a real-world scenario.
- Uses torch for numerical computations.
- Translates MATLAB code to Python.

Dependencies:
- numpy
- torch
- pickle

Usage:
Provide a brief explanation of how to run or use the script.

References:
- Shehadeh, M. A., & Kudela, J. (2025). Benchmarking global optimization techniques for unmanned aerial vehicle path planning.
  arXiv. https://arxiv.org/abs/2501.14503

Version:
- Python implementation using Torch

=========================================================
"""
from problem.basic_problem import Basic_Problem
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import time
from scipy.interpolate import RegularGridInterpolator

class UAV_Basic_Problem(Basic_Problem):
    def __init__(self):
        self.terrain_model = None
        self.FES = 0
        self.optimum = None
        self.problem_id = None
        self.dim = None
        self.lb = None
        self.ub = None

    def __str__(self):
        return f"Terrain {self.problem_id}"

    def __boundaries__(self):
        model = self.terrain_model

        nVar = model['n']

        # Initialize the boundaries dictionaries
        VarMin = {'x': model['xmin'], 'y': model['ymin'], 'z': model['zmin'],
                  'r': 0, 'psi': -np.pi / 4, 'phi': None}
        VarMax = {'x': model['xmax'], 'y': model['ymax'], 'z': model['zmax'],
                  'r': None, 'psi': np.pi / 4, 'phi': None}

        # Calculate the radial distance range based on the model's start and end points
        distance = np.linalg.norm(np.array(model['start']) - np.array(model['end']))
        VarMax['r'] = 2 * distance / nVar

        # Inclination (elevation) limits (angle range is pi/4)
        AngleRange = np.pi / 4
        VarMin['psi'] = -AngleRange
        VarMax['psi'] = AngleRange

        # Azimuth (phi)
        dirVector = np.array(model['end']) - np.array(model['start'])
        phi0 = np.arctan2(dirVector[1], dirVector[0])
        VarMin['phi'] = (phi0 - AngleRange).item()
        VarMax['phi'] = (phi0 + AngleRange).item()

        # Lower and upper Bounds of velocity
        alpha = 0.5
        VelMax = {'r': alpha * (VarMax['r'] - VarMin['r']),
                  'psi': alpha * (VarMax['psi'] - VarMin['psi']),
                  'phi': alpha * (VarMax['phi'] - VarMin['phi'])}
        VelMin = {'r': -VelMax['r'],
                  'psi': -VelMax['psi'],
                  'phi': -VelMax['phi']}

        # Create bounds by stacking both position and velocity limits
        bounds = np.array([
            [VarMin['r'], VarMax['r']],
            [VarMin['psi'], VarMax['psi']],
            [VarMin['phi'], VarMax['phi']],
        ])
        # Since we are interested in r, phi, psi for nVar points in each item of the population
        bounds = np.tile(bounds, (int(nVar), 1))

        # Assign the 0th column to self.lb and the 1st column to self.ub
        self.lb = bounds[:, 0]
        self.ub = bounds[:, 1]

    def spherical_to_cart_vec(self, solve):
        # solve : 2D [NP, 3 * n]
        # Extracting r, phi, and psi from the solution row
        r = solve[:, 0::3]  # r values are in indices 0, 3, 6, ...
        psi = solve[:, 1::3]  # psi values are in indices 1, 4, 7, ...
        phi = solve[:, 2::3]  # phi values are in indices 2, 5, 8, ...
        # [NP, n]
        model = self.terrain_model
        xs, ys, zs = model['start'] # 1D

        # [NP, n]
        x = torch.zeros_like(r)
        y = torch.zeros_like(r)
        z = torch.zeros_like(r)

        xs = torch.tensor(xs, dtype = torch.float64)
        ys = torch.tensor(ys, dtype = torch.float64)
        zs = torch.tensor(zs, dtype = torch.float64)

        x[:, 0] = xs + r[:, 0] * torch.cos(psi[:, 0]) * torch.sin(phi[:, 0])
        y[:, 0] = ys + r[:, 0] * torch.cos(psi[:, 0]) * torch.cos(phi[:, 0])
        z[:, 0] = zs + r[:, 0] * torch.sin(psi[:, 0])

        x[:, 0] = torch.clip(x[:, 0], model['xmin'], model['xmax'])
        y[:, 0] = torch.clip(y[:, 0], model['ymin'], model['ymax'])
        z[:, 0] = torch.clip(z[:, 0], model['zmin'], model['zmax'])

        # Next Cartesian coordinates
        for i in range(1, x.shape[1]):
            x[:, i] = x[:, i - 1] + r[:, i] * torch.cos(psi[:, i]) * torch.sin(phi[:, i])
            x[:, i] = torch.clip(x[:, i], model['xmin'], model['xmax'])

            y[:, i] = y[:, i - 1] + r[:, i] * torch.cos(psi[:, i]) * torch.cos(phi[:, i])
            y[:, i] = torch.clip(y[:, i], model['ymin'], model['ymax'])

            z[:, i] = z[:, i - 1] + r[:, i] * torch.sin(psi[:, i])
            z[:, i] = torch.clip(z[:, i], model['zmin'], model['zmax'])

        return x, y, z

    def DistP2S(self, xs, a, b):
        """
        Compute the shortest distance from a point xs to a line segment defined by points a and b.

        Args:
            xs: Tensor of shape [2], representing the point.
            a: Tensor of shape [2, NP], representing the start points of the segments.
            b: Tensor of shape [2, NP], representing the end points of the segments.

        Returns:
            dist: Tensor of shape [NP], representing the shortest distances.
        """

        # Convert xs to shape [2, 1] to match broadcasting
        x = xs[:, None]  # Shape: [2, 1]

        # Compute distances
        d_ab = torch.norm(a - b, dim = 0)  # [NP]
        d_ax = torch.norm(a - x, dim = 0)  # [NP]
        d_bx = torch.norm(b - x, dim = 0)  # [NP]

        # Initialize distance: If d_ab == 0, use d_ax; otherwise, use min(d_ax, d_bx)
        dist = torch.where(d_ab == 0, d_ax, torch.minimum(d_ax, d_bx))

        # Compute dot products
        dot_product_ab_ax = torch.sum((b - a) * (x - a), dim = 0)  # [NP]
        dot_product_ba_bx = torch.sum((a - b) * (x - b), dim = 0)  # [NP]

        # Compute perpendicular distance
        cross_product = torch.abs((b - a)[0] * (x - a)[1] - (b - a)[1] * (x - a)[0])  # 2D cross product
        perpendicular_dist = cross_product / d_ab

        # Update distance if within valid range
        mask = (d_ab != 0) & (dot_product_ab_ax >= 0) & (dot_product_ba_bx >= 0)
        dist = torch.where(mask, perpendicular_dist, dist)

        return dist

    def eval(self, x):
        """
                A general version of func() with adaptation to evaluate both individual and population.
                """
        start = time.perf_counter()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if x.dtype != torch.float64:
            x = x.type(torch.float64)
        if x.ndim == 1:  # x is a single individual
            y = self.func(x.reshape(1, -1))[0]
            end = time.perf_counter()
            self.T1 += (end - start) * 1000
            return y
        elif x.ndim == 2:  # x is a whole population
            y = self.func(x)
            end = time.perf_counter()
            self.T1 += (end - start) * 1000
            return y
        else:
            y = self.func(x.reshape(-1, x.shape[-1]))
            end = time.perf_counter()
            self.T1 += (end - start) * 1000
            return y

    def func(self, x):
        raise NotImplementedError

class Terrain(UAV_Basic_Problem):
    def __init__(self, terrain_model, problem_id):
        super(Terrain, self).__init__()
        self.terrain_model = terrain_model
        self.dim = 3 * terrain_model['n']
        self.problem_id = problem_id
        self.optimum = None
        self.__boundaries__()
        # self.SphCost = eng.CreateSphCost(self.terrain_model)


    def func(self, solve):
        # solve : 2D [NP, 3 * nv]
        model = self.terrain_model

        J_pen = torch.tensor(model['J_pen'], dtype = torch.float64)
        H = torch.tensor(model['H'], dtype = torch.float64)

        x, y, z = self.spherical_to_cart_vec(solve) # 2D : [NP, nv]

        NP, n = x.shape
        # Start location
        xs, ys, zs = model['start'] # 1D

        # Final location
        xf, yf, zf = model['end'] # 1D

        xs = torch.tensor(xs, dtype = torch.float64)
        ys = torch.tensor(ys, dtype = torch.float64)
        zs = torch.tensor(zs, dtype = torch.float64)

        xf = torch.tensor(xf, dtype = torch.float64)
        yf = torch.tensor(yf, dtype = torch.float64)
        zf = torch.tensor(zf, dtype = torch.float64)

        # REPEAT
        xs = torch.tile(xs, (NP, 1))
        ys = torch.tile(ys, (NP, 1))
        zs = torch.tile(zs, (NP, 1))

        xf = torch.tile(xf, (NP, 1))
        yf = torch.tile(yf, (NP, 1))
        zf = torch.tile(zf, (NP, 1))

        # Concatenate
        x_all = torch.cat((xs, x, xf), dim = 1)
        y_all = torch.cat((ys, y, yf), dim = 1)
        z_all = torch.cat((zs, z, zf), dim = 1)

        N = x_all.shape[1] # Full path length

        # Altitude wrt sea level = z_relative + ground_level
        z_abs = torch.zeros((NP, N), dtype = torch.float64)
        for i in range(N):
            x_index = np.round(x_all[:, i].cpu().numpy()).astype(int) - 1
            y_index = np.round(y_all[:, i].cpu().numpy()).astype(int) - 1
            z_abs[:, i] = z_all[:, i] + H[y_index, x_index]

        # ---------- J1 Cost for path length ----------
        diff = torch.stack((x_all[:, 1:] - x_all[:, :-1],
                            y_all[:, 1:] - y_all[:, :-1],
                            z_abs[:, 1:] - z_abs[:, :-1]), dim = -1)  # [NP, N-1, 3]
        J1 = torch.sum(torch.norm(diff, dim = 2), dim = 1)  # Shape: [NP]

        # ---------- J2 - threats / obstacles Cost ----------
        threats = model['threats']
        threat_num = threats.shape[0]

        # Checking if UAV passes through a threat
        J2 = torch.zeros(NP, dtype = torch.float64)
        for i in range(threat_num):
            threat = threats[i, :]
            threat_x = threat[0]
            threat_y = threat[1]
            threat_radius = threat[3]

            for j in range(N - 1):
                dist = self.DistP2S(torch.tensor([threat_x, threat_y], dtype = torch.float64),
                                    torch.cat((x_all[:, j][None, :], y_all[:, j][None, :]), dim = 0),
                                    torch.cat((x_all[:, j + 1][None, :], y_all[:, j + 1][None, :]), dim = 0))
                # dist 1D NP
                threat_cost = threat_radius + model['drone_size'] + model['danger_dist'] - dist
                # threat_cost = torch.full([NP], threat_radius + model['drone_size'] + model['danger_dist'] - dist, dtype = torch.float64) # Dangerous Zone
                threat_cost = torch.where(dist > threat_radius + model['drone_size'] + model['danger_dist'], torch.tensor(0.0, dtype = torch.float64), threat_cost)  # No Collision
                threat_cost = torch.where(dist < threat_radius + model['drone_size'], J_pen, threat_cost)  # Collision

                J2 += threat_cost

        # ---------- J3 - Altitude cost ----------
        z_max = torch.tensor(model['zmax'], dtype = torch.float64)
        z_min = torch.tensor(model['zmin'], dtype = torch.float64)
        J3 = torch.sum(torch.where(z < 0, J_pen, torch.abs(z - (z_max + z_min) / 2)), dim = 1)

        # ---------- J4 - Smooth cost ----------
        J4 = torch.zeros(NP, dtype = torch.float64)
        turning_max = 45
        climb_max = 45

        # Calculate the projections of the segments in the xy-plane (x, y, 0) for all paths at once
        diff_1 = torch.stack((x_all[:, 1:] - x_all[:, :-1], y_all[:, 1:] - y_all[:, :-1], torch.zeros((NP, N - 1))),  dim = -1)  # [NP, N-1, 3]
        diff_2 = torch.stack((x_all[:, 2:] - x_all[:, 1:-1], y_all[:, 2:] - y_all[:, 1:-1], torch.zeros((NP, N - 2))), dim = -1)  # [NP, N-2, 3]

        for i in range(0, N - 2):
            segment1_proj = diff_1[:, i, :] # [NP, 3]
            segment2_proj = diff_2[:, i, :] # [NP, 3]

            # Find rows where all values in segment1_proj are zero (no movement in this segment)
            zero_segment1 = torch.all(segment1_proj == 0, dim = 1)  # [NP,] - boolean array, True where all are zeros

            # Find rows where all values in segment2_proj are zero (no movement in this segment)
            zero_segment2 = torch.all(segment2_proj == 0, dim = 1)  # [NP,] - boolean array, True where all are zeros

            # Handle zero segments: if a segment is all zeros, use the previous or next valid segment
            # For segment1_proj, if it's zero, we will use the previous segment (diff_1[i-1])
            i1 = i - 1
            i2 = i + 1
            while i1 >= 0 and torch.any(zero_segment1):
                segment1_proj[zero_segment1] = diff_1[zero_segment1, i1, :]
                zero_segment1 = torch.all(segment1_proj == 0, dim = 1)
                i1 -= 1

            while i2 < N - 2 and torch.any(zero_segment2):
                segment2_proj[zero_segment2] = diff_2[zero_segment2, i2, :]
                zero_segment2 = torch.all(segment2_proj == 0, dim = 1)
                i2 += 1

            # segment1_proj and segment2_proj [NP, 3]
            # Calculate the climb angles
            climb_angle1 = torch.atan2(z_abs[:, i + 1] - z_abs[:, i], torch.norm(segment1_proj, dim = 1)) * (180.0 / np.pi)
            climb_angle2 = torch.atan2(z_abs[:, i + 2] - z_abs[:, i + 1], torch.norm(segment2_proj, dim = 1)) * (180.0 / np.pi)

            # Calculate the turning angle
            turning_angle = torch.atan2(torch.norm(torch.cross(segment1_proj, segment2_proj, dim = 1), dim = 1),
                                        torch.sum(segment1_proj * segment2_proj, dim = 1)) * (180.0 / np.pi)

            addition_J_1 = torch.where(torch.abs(turning_angle) > turning_max, torch.abs(turning_angle), torch.tensor(0.0, dtype = torch.float64))
            addition_J_2 = torch.where(torch.abs(climb_angle2 - climb_angle1) > climb_max, torch.abs(climb_angle2 - climb_angle1), torch.tensor(0.0, dtype = torch.float64))

            J4 += addition_J_1 + addition_J_2

            # ---------- J5 - terrain cost ----------
            J5 = torch.full([NP], J_pen, dtype = torch.float64)
            paths_above = self.are_paths_clear(x_all, y_all, z_abs, H)
            J5[paths_above] = 0
            b1 = 5
            b2 = 1
            b3 = 10
            b4 = 1
            b5 = 1
            return torch.column_stack((b1 * J1, b2 * J2, b3 * J3, b4 * J4, b5 * J5))

    def are_paths_clear(self, x_all, y_all, z_abs, H, num_samples = 12):
        """
        Check if all the line segments connecting adjacent points of NP paths are completely above the terrain H.
        :param x_all: (NP, N) shaped array, x coordinates of N points for each of NP paths
        :param y_all: (NP, N) shaped array, y coordinates of N points for each of NP paths
        :param z_abs: (NP, N) shaped array, absolute heights of N points for each of NP paths
        :param H: (H_rows, H_cols) shaped terrain height matrix
        :param num_samples: The number of sample points along each line segment (excluding the endpoints)
        :return: A boolean array of shape (NP,) where each value indicates whether all segments of a path are above the terrain
        """
        # Generate the interpolation function for the terrain

        x_all_np = x_all.detach().cpu().numpy()
        y_all_np = y_all.detach().cpu().numpy()
        z_abs_np = z_abs.detach().cpu().numpy()
        H_np = H.detach().cpu().numpy()


        H_rows, H_cols = H_np.shape
        x_indices = np.arange(H_cols)  # X-direction indices
        y_indices = np.arange(H_rows)  # Y-direction indices
        interp_func = RegularGridInterpolator((y_indices, x_indices), H)

        # Calculate the sample points for all line segments
        x_interp = np.linspace(x_all_np[:, :-1], x_all_np[:, 1:], steps = num_samples, axis = 2)
        y_interp = np.linspace(y_all_np[:, :-1], y_all_np[:, 1:], steps = num_samples, axis = 2)
        z_interp = np.linspace(z_abs_np[:, :-1], z_abs_np[:, 1:], steps = num_samples, axis = 2)

        # Compute the terrain heights at all interpolated points
        terrain_heights = interp_func(np.stack([y_interp, x_interp], axis = -1))  # (NP, N-1, num_samples)

        # Check if all sampled points are above the terrain
        return np.all(z_interp > terrain_heights, axis = (1, 2))

