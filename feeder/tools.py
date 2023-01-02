import numpy as np
import random
import networkx as nx


def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1)**2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
            t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy


def top_k_by_category(label, score, top_k):
    instance_num, class_num = score.shape
    rank = score.argsort()
    hit_top_k = [[] for i in range(class_num)]
    for i in range(instance_num):
        l = label[i]
        hit_top_k[l].append(l in rank[i, -top_k:])

    accuracy_list = []
    for hit_per_category in hit_top_k:
        if hit_per_category:
            accuracy_list.append(sum(hit_per_category) * 1.0 / len(hit_per_category))
        else:
            accuracy_list.append(0.0)
    return accuracy_list


def calculate_recall_precision(label, score):
    instance_num, class_num = score.shape
    rank = score.argsort()
    confusion_matrix = np.zeros([class_num, class_num])

    for i in range(instance_num):
        true_l = label[i]
        pred_l = rank[i, -1]
        confusion_matrix[true_l][pred_l] += 1

    precision = []
    recall = []

    for i in range(class_num):
        true_p = confusion_matrix[i][i]
        false_n = sum(confusion_matrix[i, :]) - true_p
        false_p = sum(confusion_matrix[:, i]) - true_p
        precision.append(true_p * 1.0 / (true_p + false_p))
        recall.append(true_p * 1.0 / (true_p + false_n))

    return precision, recall


def get_x_rot(theta=None):
    theta = 2 * np.pi * np.random.rand() if theta is None else theta
    return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])


def get_y_rot(theta=None):
    theta = 2 * np.pi * np.random.rand() if theta is None else theta
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])


def get_z_rot(theta=None):
    theta = 2 * np.pi * np.random.rand() if theta is None else theta
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])


def compute_so3_chains(joints, edges, selected_joint_pairs, kinematic_tree=None):

    if len(joints.shape) < 4:
        joints = joints[None]

    if kinematic_tree is None:
        kinematic_tree = nx.Graph()
        kinematic_tree.add_edges_from(edges)

    # TODO: fix this for when we use 3D keypoints
    #kp_mod = np.concatenate([seq_mod[..., :2], np.zeros_like(seq_mod[..., :2][..., :1])], axis=-1)
    #kp_par = np.concatenate([seq_par[..., :2], np.zeros_like(seq_par[..., :2][..., :1])], axis=-1)

    # Compute the rotation matrices for each segment in the tree
    R_mod = kinematic_tree_3d(joints, edges)

    # Computes the pose energy along each pair of nodes in the tree

    # find out the path between the root node '0' and <n_idx> node, along with all the rotation
    # matrices in the path
    #distances_mod = get_pose_energy(all_paths_from_root, kinematic_tree, R_mod, edges=edges, mode='pairs')  # [:, :, 1:] # Ignores the root node
    distances = get_pose_energy(kinematic_tree, R_mod, edges=edges, pairs=selected_joint_pairs)

    return distances


def kinematic_tree_3d(kp, edges):
    eps = 1e-10

    bs, num_frames, n_kp, _ = kp.shape
    #n_segs = edges.shape[0]
    #segments = kp[:, :, edges[:, 1]] - kp[:, :, edges[:, 0]]

    #base_segment = np.tile(np.array([1.0, 0.0, 0.0])[None, None, None], reps=[bs, num_frames, 1, 1])
    #segments = np.concatenate([base_segment, segments], axis=-2)
    #segments /= np.linalg.norm(segments, axis=-1, keepdims=True)# + eps    # normalize the segments

    #left_seg = np.tile(aa[None, None, None], reps=[1, 50, 13, 1])
    #right_seg = np.tile(aa[None, None, None], reps=[1, 50, 13, 1])

    aux = edges[:, None, 0] == edges[None, :, 1]
    parent_edges = np.stack([edges[aux.argmax(axis=-1), 0], edges[:, 0]]).T # Assumes a tree (only one parent)

    valid_segs = np.any(aux, axis=-1)
    #valid_segments = segments[:, :, valid_segs, :]
    filtered_edges = edges[valid_segs]
    filtered_parent_edges = parent_edges[valid_segs]

    left_seg = kp[:, :, filtered_parent_edges[:, 1]] - kp[:, :, filtered_parent_edges[:, 0]]

    left_seg_mag = np.linalg.norm(left_seg, axis=-1, keepdims=True)
    assert (left_seg_mag == 0.0).sum() == 0, "The length of the Kinematic tree's edges cannot be zero"

    # Fixes the singularities when the left and right segments are co-linear
    left_seg += np.random.normal(loc=0.0, scale=eps, size=left_seg.shape)   # to avoid co-linear segments
    left_seg_mag = np.linalg.norm(left_seg, axis=-1, keepdims=True)
    left_seg /= left_seg_mag

    right_seg = kp[:, :, filtered_edges[:, 1]] - kp[:, :, filtered_edges[:, 0]]
    right_seg_mag = np.linalg.norm(right_seg, axis=-1, keepdims=True)
    assert (right_seg_mag == 0.0).sum() == 0, "The length of the Kinematic tree's edges cannot be zero"
    right_seg /= right_seg_mag

    # compute the rotation matrix bt. segments
    c = (left_seg * right_seg).sum(axis=-1, keepdims=True)[..., None]

    v = np.cross(left_seg, right_seg)
    s = np.linalg.norm(v, axis=-1, keepdims=True)[..., None]

    I = np.tile(np.eye(3)[None, None, None, ...], reps=[bs, num_frames, filtered_edges.shape[0], 1, 1])
    S = get_skew_matrix(v)

    # Computes the rotation matrix of each segment relative ot its parent
    R = I + S + S @ S * (1 - c) / (s**2)

    R_all_edges = np.tile(np.eye(3)[None, None, None, ...], reps=[bs, num_frames, edges.shape[0], 1, 1])
    #nx.set_edge_attributes(kg, {(i, j): R[:, :, n, ...] for n, (i, j) in enumerate(edges.T)})

    # Assign the appropriate rotation metric to each edge. For all edges with out a parent, the rotation matrix remains
    # the identity matrix. NOTE: This is a general approach to this results, but it should be only one edge without α
    # parent in the kinematic tree
    R_all_edges[:, :, valid_segs, ...] = R

    if np.isnan(R_all_edges).sum() > 0:
        import pdb
        pdb.set_trace()

        from actim.utils.visualization import show_kinematic_tree_2d
        a = 5

    return R_all_edges


def get_skew_matrix(v):

    M = np.zeros(v.shape + (3,))
    M[..., 0, 1] = -v[..., 2]
    M[..., 0, 2] =  v[..., 1]
    M[..., 1, 0] =  v[..., 2]
    M[..., 1, 2] = -v[..., 0]
    M[..., 2, 0] = -v[..., 1]
    M[..., 2, 1] =  v[..., 0]
    return M


def get_v_from_skew_matrix(M):

    v = np.zeros(M.shape[:-1])
    v[..., 0] = M[..., 2, 1]
    v[..., 1] = M[..., 0, 2]
    v[..., 2] = M[..., 1, 0]

    return v


def get_pose_path(kinematic_tree, R, edges, pairs):

    #import pdb
    #pdb.set_trace()

    bs, num_frames, _, _, _ = R.shape
    n_pairs = pairs.shape[0]

    composed_R_left = np.tile(np.eye(3)[None, None, None, ...], reps=[bs, num_frames, n_pairs, 1, 1])
    composed_R_right = np.tile(np.eye(3)[None, None, None, ...], reps=[bs, num_frames, n_pairs, 1, 1])

    for n_idx, pair in enumerate(pairs):

        path_0 = nx.shortest_path(kinematic_tree, 0, pair[0])
        path_1 = nx.shortest_path(kinematic_tree, 0, pair[1])

        path_edges_0 = np.stack([path_0[:-1], path_0[1:]])
        edges_idx_0 = np.where(np.all((edges.T[:, None, :] == path_edges_0[:, :, None]), axis=0))[1]
        n_segs_0 = edges_idx_0.shape[0]
        path_R_0 = R[:, :, edges_idx_0, ...]

        # Chain-multiply all the rotations matrices in the path
        for i in range(n_segs_0):
            composed_R_left[:, :, n_idx] = path_R_0[:, :, i, ...] @ composed_R_left[:, :, n_idx]

        path_edges_1 = np.stack([path_1[:-1], path_1[1:]])
        edges_idx_1 = np.where(np.all((edges.T[:, None, :] == path_edges_1[:, :, None]), axis=0))[1]
        n_segs_1 = edges_idx_1.shape[0]
        path_R_1 = R[:, :, edges_idx_1, ...]

        # Chain-multiply all the rotations matrices in the path
        for i in range(n_segs_1):
            composed_R_right[:, :, n_idx] = path_R_1[:, :, i, ...] @ composed_R_right[:, :, n_idx]

    pair_rot = composed_R_left.transpose(0, 1, 2, 4, 3) @ composed_R_right
    #pose_energy = log_map(composed_R_left.transpose(0, 1, 2, 4, 3) @ composed_R_right)

    return pair_rot
