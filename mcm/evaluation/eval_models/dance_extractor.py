import numpy as np
import os
import sys

sys.path.append(os.curdir)
from evaluation.eval_models.feature_utils import calc_average_velocity, calc_average_velocity_horizontal, \
    calc_average_velocity_vertical, calc_average_acceleration, distance_between_points, \
    velocity_direction_above_threshold, velocity_direction_above_threshold_normal, distance_from_plane, \
    distance_from_plane_normal, angle_within_range, velocity_above_threshold

SMPL_JOINT_NAMES = [
    "root",
    "lhip", "rhip", "belly",
    "lknee", "rknee", "spine",
    "lankle", "rankle", "chest",
    "ltoes", "rtoes", "neck",
    "linshoulder", "rinshoulder",
    "head", "lshoulder", "rshoulder",
    "lelbow", "relbow",
    "lwrist", "rwrist",
    "lhand", "rhand"
]


class DanceExtractor(object):
    def extract_kinetic_features_batch(self, positions):
        """
        :param positions: b t j c
        :param m_lens: b
        :return: b 66
        """
        return np.stack([self.extract_kinetic_features(p) for p in positions])

    def extract_manual_features_batch(self, positions):
        """
        :param positions: b t j c
        :return: b 32
        """
        return np.stack([self.extract_manual_features(p) for p in positions])

    def extract_kinetic_features(self, positions):
        """ cal average horizontal, vertical velocity and acceleration as feature of motion
        :param positions: t j c
        :return: j*3
        """
        assert len(positions.shape) == 3  # (seq_len, n_joints, 3)
        features = KineticFeatures(positions, frame_time=1 / 20.)
        kinetic_feature_vector = []
        # feature of every joint
        for i in range(positions.shape[1]):
            feature_vector = np.hstack(
                [
                    features.average_kinetic_energy_horizontal(i),
                    features.average_kinetic_energy_vertical(i),
                    features.average_energy_expenditure(i),
                ]
            )
            kinetic_feature_vector.extend(feature_vector)
        kinetic_feature_vector = np.asarray(kinetic_feature_vector, dtype=np.float32)

        return kinetic_feature_vector

    def extract_manual_features(self, positions):
        assert len(positions.shape) == 3  # (seq_len, n_joints, 3)
        features = []
        f = ManualFeatures(positions)
        for _ in range(1, positions.shape[0]):
            pose_features = []
            pose_features.append(
                f.f_nmove("neck", "rhip", "lhip", "rwrist", 1.8 * f.hl)
            )
            pose_features.append(
                f.f_nmove("neck", "lhip", "rhip", "lwrist", 1.8 * f.hl)
            )
            pose_features.append(
                f.f_nplane("chest", "neck", "neck", "rwrist", 0.2 * f.hl)
            )
            pose_features.append(
                f.f_nplane("chest", "neck", "neck", "lwrist", 0.2 * f.hl)
            )
            pose_features.append(
                f.f_move("belly", "chest", "chest", "rwrist", 1.8 * f.hl)
            )
            pose_features.append(
                f.f_move("belly", "chest", "chest", "lwrist", 1.8 * f.hl)
            )
            pose_features.append(
                f.f_angle("relbow", "rshoulder", "relbow", "rwrist", [0, 110])
            )
            pose_features.append(
                f.f_angle("lelbow", "lshoulder", "lelbow", "lwrist", [0, 110])
            )
            pose_features.append(
                f.f_nplane(
                    "lshoulder", "rshoulder", "lwrist", "rwrist", 2.5 * f.sw
                )
            )
            pose_features.append(
                f.f_move("lwrist", "rwrist", "rwrist", "lwrist", 1.4 * f.hl)
            )
            pose_features.append(
                f.f_move("rwrist", "root", "lwrist", "root", 1.4 * f.hl)
            )
            pose_features.append(
                f.f_move("lwrist", "root", "rwrist", "root", 1.4 * f.hl)
            )
            pose_features.append(f.f_fast("rwrist", 2.5 * f.hl))
            pose_features.append(f.f_fast("lwrist", 2.5 * f.hl))
            pose_features.append(
                f.f_plane("root", "lhip", "ltoes", "rankle", 0.38 * f.hl)
            )
            pose_features.append(
                f.f_plane("root", "rhip", "rtoes", "lankle", 0.38 * f.hl)
            )
            pose_features.append(
                f.f_nplane("zero", "y_unit", "y_min", "rankle", 1.2 * f.hl)
            )
            pose_features.append(
                f.f_nplane("zero", "y_unit", "y_min", "lankle", 1.2 * f.hl)
            )
            pose_features.append(
                f.f_nplane("lhip", "rhip", "lankle", "rankle", 2.1 * f.hw)
            )
            pose_features.append(
                f.f_angle("rknee", "rhip", "rknee", "rankle", [0, 110])
            )
            pose_features.append(
                f.f_angle("lknee", "lhip", "lknee", "lankle", [0, 110])
            )
            pose_features.append(f.f_fast("rankle", 2.5 * f.hl))
            pose_features.append(f.f_fast("lankle", 2.5 * f.hl))
            pose_features.append(
                f.f_angle("neck", "root", "rshoulder", "relbow", [25, 180])
            )
            pose_features.append(
                f.f_angle("neck", "root", "lshoulder", "lelbow", [25, 180])
            )
            pose_features.append(
                f.f_angle("neck", "root", "rhip", "rknee", [50, 180])
            )
            pose_features.append(
                f.f_angle("neck", "root", "lhip", "lknee", [50, 180])
            )
            pose_features.append(
                f.f_plane("rankle", "neck", "lankle", "root", 0.5 * f.hl)
            )
            pose_features.append(
                f.f_angle("neck", "root", "zero", "y_unit", [70, 110])
            )
            # if distance between rwrist and plane(zero,-y,y_min) > -1.2*hl
            # vertical dist between rwrist and floor > 1.2 upper arm

            pose_features.append(
                f.f_nplane("zero", "minus_y_unit", "y_min", "rwrist", -1.2 * f.hl)
            )
            pose_features.append(
                f.f_nplane("zero", "minus_y_unit", "y_min", "lwrist", -1.2 * f.hl)
            )
            pose_features.append(f.f_fast("root", 2.3 * f.hl))
            features.append(pose_features)
            f.next_frame()
        features = np.array(features, dtype=np.float32).mean(axis=0)
        return features


class KineticFeatures:
    def __init__(
            self, positions, frame_time=1 / 60., up_vec="y", sliding_window=2
    ):
        self.positions = positions
        self.frame_time = frame_time
        self.up_vec = up_vec
        self.sliding_window = sliding_window

    def average_kinetic_energy(self, joint):
        average_kinetic_energy = 0
        for i in range(1, len(self.positions)):
            average_velocity = calc_average_velocity(
                self.positions, i, joint, self.sliding_window, self.frame_time
            )
            average_kinetic_energy += average_velocity ** 2
        average_kinetic_energy = average_kinetic_energy / (
                len(self.positions) - 1.0
        )
        return average_kinetic_energy

    def average_kinetic_energy_horizontal(self, joint):
        """
        :param joint: i
        :return:
        """
        val = 0
        for i in range(1, len(self.positions)):
            average_velocity = calc_average_velocity_horizontal(
                self.positions,
                i,  # frame_idx
                joint,
                self.sliding_window,
                self.frame_time,
                self.up_vec
            )
            val += average_velocity ** 2
        val = val / (len(self.positions) - 1.0)
        return val

    def average_kinetic_energy_vertical(self, joint):
        val = 0
        for i in range(1, len(self.positions)):
            average_velocity = calc_average_velocity_vertical(
                self.positions,
                i,
                joint,
                self.sliding_window,
                self.frame_time,
                self.up_vec,
            )
            val += average_velocity ** 2
        val = val / (len(self.positions) - 1.0)
        return val

    def average_energy_expenditure(self, joint):
        val = 0.0
        for i in range(1, len(self.positions)):
            val += calc_average_acceleration(
                self.positions, i, joint, self.sliding_window, self.frame_time
            )
        val = val / (len(self.positions) - 1.0)
        return val


class ManualFeatures:
    def __init__(self, positions, joint_names=SMPL_JOINT_NAMES, frame_time=1. / 120.):
        self.positions = positions
        self.joint_names = joint_names
        self.frame_num = 1
        self.frame_time = frame_time
        self.hl = distance_between_points(
            [1.99113488e-01, 2.36807942e-01, -1.80702247e-02],  # "lshoulder",
            [4.54445392e-01, 2.21158922e-01, -4.10167128e-02],  # "lelbow"
        )
        # shoulder width
        self.sw = distance_between_points(
            [1.99113488e-01, 2.36807942e-01, -1.80702247e-02],  # "lshoulder"
            [-1.91692337e-01, 2.36928746e-01, -1.23055102e-02, ],  # "rshoulder"
        )
        # hip width
        self.hw = distance_between_points(
            [5.64076714e-02, -3.23069185e-01, 1.09197125e-02],  # "lhip"
            [-6.24834076e-02, -3.31302464e-01, 1.50412619e-02],  # "rhip"
        )
        # print(self.hl,self.sw,self.hw)

    def next_frame(self):
        self.frame_num += 1

    def transform_and_fetch_position(self, j):
        if j == "y_unit":
            return [0, 1, 0]
        elif j == "minus_y_unit":
            return [0, -1, 0]
        elif j == "zero":
            return [0, 0, 0]
        elif j == "y_min":
            return [
                0,
                min(
                    [y for (_, y, _) in self.positions[self.frame_num]]
                ),
                0,
            ]
        return self.positions[self.frame_num][
            self.joint_names.index(j)
        ]

    def transform_and_fetch_prev_position(self, j):
        return self.positions[self.frame_num - 1][
            self.joint_names.index(j)
        ]

    def f_move(self, j1, j2, j3, j4, range):
        j1_prev, j2_prev, j3_prev, j4_prev = [
            self.transform_and_fetch_prev_position(j) for j in [j1, j2, j3, j4]
        ]
        j1, j2, j3, j4 = [
            self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]
        ]
        return velocity_direction_above_threshold(
            j1, j1_prev, j2, j2_prev, j3, j3_prev, range, self.frame_time
        )

    def f_nmove(self, j1, j2, j3, j4, range):
        j1_prev, j2_prev, j3_prev, j4_prev = [
            self.transform_and_fetch_prev_position(j) for j in [j1, j2, j3, j4]
        ]
        j1, j2, j3, j4 = [
            self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]
        ]
        return velocity_direction_above_threshold_normal(
            j1, j1_prev, j2, j3, j4, j4_prev, range, self.frame_time
        )

    def f_plane(self, j1, j2, j3, j4, threshold):
        j1, j2, j3, j4 = [
            self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]
        ]
        return distance_from_plane(j1, j2, j3, j4, threshold)

    #
    def f_nplane(self, j1, j2, j3, j4, threshold):
        j1, j2, j3, j4 = [
            self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]
        ]
        return distance_from_plane_normal(j1, j2, j3, j4, threshold)

    # relative
    def f_angle(self, j1, j2, j3, j4, range):
        j1, j2, j3, j4 = [
            self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]
        ]
        return angle_within_range(j1, j2, j3, j4, range)

    # non-relative
    def f_fast(self, j1, threshold):
        j1_prev = self.transform_and_fetch_prev_position(j1)
        j1 = self.transform_and_fetch_position(j1)
        return velocity_above_threshold(j1, j1_prev, threshold)


if __name__ == '__main__':
    extractor = DanceExtractor()
    joints = np.random.random([60, 22, 3])
    print(extractor.extract_kinetic_features(joints))
    print(extractor.extract_manual_features(joints))
