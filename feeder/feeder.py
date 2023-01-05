# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import networkx as nx
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from net.utils.graph import Graph

# visualization
import time

# operation
from . import tools

class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    AUG_MODS = ['azimuthal', 'so3', 'limbs_scale', 'no_aug']
    FEAT_MODS = ['xyz', 'so3_chains']

    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True,
                 training=True,
                 aug_mod='no_aug',
                 feat_mode='xyz',
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
                 ):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.aug_mod = aug_mod
        self.feat_mode = feat_mode
        self.is_training = training

        # load graph
        self.graph = Graph(**graph_args)
        self.tree_edges = self.graph.get_tree_edges()
        self.kinematic_tree = nx.Graph()
        self.kinematic_tree.add_edges_from(self.tree_edges)

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
            
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        # Generates random angles to apply to the examples, if augmentation is required
        # Note that when testing the seed is feed to guarantee the same data in each test.
        if not self.is_training:
            np.random.seed(1256)
        self.random_angles = 2 * np.pi * np.random.rand(len(self.label), 3)
        self.random_scales = (0.7 * np.random.rand() + 0.5) * np.random.rand(len(self.label), 1)
        np.random.seed()    # Restart, the seed to get real random numbers

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def _augment_example(self, data, data_idx):

        data_aux = data.reshape(3, -1)
        valid_elements = ~(data_aux.reshape(3, -1) == 0.0).all(axis=0)
        scene_center = data_aux[:, valid_elements].mean(axis=-1, keepdims=True)

        # Translate the data to the estimated scene center
        data_aux[:, valid_elements] -= scene_center
        # Apply augmentation if any
        if self.aug_mod == 'azimuthal':  # In the NTU dataset the y-axis points up
            # Rotates all the points in the kinematic tree around the y-axis with a random rotation
            data_aux = tools.get_y_rot(theta=self.random_angles[data_idx, 0]) @ data_aux
        elif self.aug_mod == 'so3':
            # Rotates all the points in the kinematic tree with a random so3 rotation
            rot_x = tools.get_x_rot(theta=self.random_angles[data_idx, 0])
            rot_y = tools.get_y_rot(theta=self.random_angles[data_idx, 1])
            rot_z = tools.get_z_rot(theta=self.random_angles[data_idx, 2])
            data_aux = rot_x @ rot_y @ rot_z @ data_aux
        elif self.aug_mod == 'limbs_scale':
            data_aux *= self.random_scales[data_idx, 0]
        elif self.aug_mod == 'no_aug':
            pass
        else:
            raise ValueError(f'Augmentation mode "{self.aug_mod}" is not valid. Try a value in {self.AUG_MODS}.')

        # Translate the data back to the original coordinate system
        data_aux[:, valid_elements] += scene_center

        # Back to the original data shape
        data = data_aux.reshape(data.shape)

        return data

    def _compute_so3_chain(self, data):

        C, T, V, M = data.shape

        joint_coords = data.transpose(3, 1, 2, 0).reshape(-1, 1, V, C)
        mask = ~(joint_coords == 0.0).all(axis=-1).any(axis=-1)[:, 0]

        masked_joint_coords = joint_coords[mask]

        # Compute the rotation matrices for each segment in the tree
        R_mod = tools.kinematic_tree_3d(masked_joint_coords, self.tree_edges)

        # Computes the pose energy along each pair of nodes in the tree

        # find out the path between the root node '0' and <n_idx> node, along with all the rotation
        # matrices in the path
        selected_joint_pairs = np.stack([np.zeros(V), np.arange(V)]).T

        path_rot_masked = tools.get_pose_path(self.kinematic_tree, R_mod, edges=self.tree_edges, pairs=selected_joint_pairs)

        # Reshape the features
        path_rot = np.zeros((M * T, 1, V, 3, 3))
        path_rot[mask] = path_rot_masked
        path_rot = path_rot.reshape(M, T, V, 3, 3).transpose(3, 4, 1, 2, 0)

        return path_rot

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        
        # processing
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # Applies augmentations to the xyz coordinates, if any
        data_numpy = self._augment_example(data_numpy, index)

        # Computes the feature mode
        if self.feat_mode == 'so3_chains':
            data_numpy = self._compute_so3_chain(data_numpy)
            C1, C2, T, V, M = data_numpy.shape

            # Only the first two columns of the rotation matrix
            data_numpy = data_numpy[:, :2].reshape(-1, T, V, M)
        elif self.feat_mode == 'xyz':
            pass
        else:
            raise ValueError(f'Feature mode "{self.feat_mode}" is not valid. Try a value in {self.FEAT_MODS}.')

        return data_numpy, label
