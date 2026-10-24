

import numpy as np
from estimator.config import POSE_CONNECTIONS, ANGLE_TRIPLETS,UPPER_TRIPLETS
# from estimator.upper_detector import compute_angles_from_three_points





import numpy as np
import math


def compute_angles_from_three_points(P1, P2, P3, ref=np.array([0.0,0.0,1.0])):
    """
    P1: endpoint of first link (3-array)
    P2: joint (center) between links (3-array)
    P3: endpoint of second link (3-array)
    ref: reference vector for defining roll (default global Z). If ref ~ parallel to v1, function
         will automatically switch to global X internally.

    Returns (theta1_deg, theta2_deg, theta3_deg, theta4_deg)
      theta1 = yaw  (rotation about global Z)  -- direction of v1 in XY
      theta2 = pitch (tilt of v1)
      theta3 = roll  (twist around v1, determined from v2)
      theta4 = angle between v1 and v2 (joint bend)
    """

    P1 = np.asarray(P1, dtype=float)
    P2 = np.asarray(P2, dtype=float)
    P3 = np.asarray(P3, dtype=float)
    v1 = P2 - P1
    v2 = P3 - P2
    
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 < 1e-12 or norm_v2 < 1e-12:
        raise ValueError("One of the link vectors has (near) zero length.")

    # unit axis of first link
    e1 = v1 / norm_v1

    # ensure reference is not (nearly) parallel to e1
    ref = np.asarray(ref, dtype=float)
    if abs(np.dot(ref, e1)) > 0.999:
        # ref parallel to e1 -> switch to X
        ref = np.array([1.0, 0.0, 0.0])

    # r_ref: projection of ref onto plane perpendicular to e1, normalized
    r_ref = ref - np.dot(ref, e1) * e1
    r_ref /= np.linalg.norm(r_ref)

    # e2: projection of v2 onto plane perpendicular to e1, normalized
    e2 = v2 - np.dot(v2, e1) * e1
    norm_e2 = np.linalg.norm(e2)
    if norm_e2 < 1e-10:
        # v2 parallel to v1 -> roll undefined / arbitrary
        theta3_deg = 0.0
        e2 = None
    else:
        e2 = e2 / norm_e2
        # signed roll angle from r_ref to e2 about axis e1:
        num = np.dot(e1, np.cross(r_ref, e2))
        den = np.dot(r_ref, e2)
        theta3_deg = math.degrees(math.atan2(num, den))

    # yaw and pitch from e1
    ex, ey, ez = e1
    theta1_deg = math.degrees(math.atan2(ey, ex))                      # yaw
    theta2_deg = math.degrees(math.atan2(ez, math.sqrt(ex*ex + ey*ey)))# pitch

    # angle between v1 and v2 (theta4)
    dot = np.dot(v1, v2) / (norm_v1 * norm_v2)
    dot = max(-1.0, min(1.0, dot))  # clamp numerical error
    theta4_deg = math.degrees(math.acos(dot))
    error = False
    if ( theta3_deg-90 > 90 or theta3_deg-90 < -90 or theta4_deg+90 < 0 or theta4_deg > 170 or theta1_deg < -75 or theta1_deg > 90 or theta2_deg < -10 or theta2_deg > +90 ):
        # print("error!!!")
        error = True
    
    return theta1_deg, theta2_deg, theta3_deg, theta4_deg,error

class SkeletonKinematicModel:
    
    def __init__(self):
        """
        connection_pairs: list of (jointA, jointB) tuples
        """
        self.connection_pairs = POSE_CONNECTIONS
        self.angle_triplets = ANGLE_TRIPLETS
        self.upper_tripltes = UPPER_TRIPLETS
        self.bone_lengths = {}  # Dict: (jointA, jointB): length

    @staticmethod
    def bone_length(joint_a_pos, joint_b_pos):
        """
        Euclidean distance between two joints (x, y, z)
        """
        return np.linalg.norm(np.array(joint_b_pos) - np.array(joint_a_pos))

    def compute_average_bone_lengths(self, frames_landmarks, N_init=3):
        """
        frames_landmarks: list of dicts {joint_name: (x, y, z)}, per frame
        Only uses the first N_init frames (if available).
        """
        bone_samples = {pair: [] for pair in self.connection_pairs}
        frames_used = 0
        for joints in frames_landmarks:
            if joints is None:
                continue
            valid = True
            for a, b in self.connection_pairs:
                if (a not in joints) or (b not in joints):
                    valid = False  # skip frames missing joints
                    break
            if not valid:
                continue
            for a, b in self.connection_pairs:
                length = self.bone_length(joints[a], joints[b])
                bone_samples[(a, b)].append(length)
            frames_used += 1
            if frames_used >= N_init:
                break

        self.bone_lengths = {}
        for pair, samples in bone_samples.items():
            if samples:
                self.bone_lengths[pair] = np.mean(samples)
            else:
                self.bone_lengths[pair] = None  

    def get_bone_length(self, jointA, jointB):
        return self.bone_lengths.get((jointA, jointB), None)
    
    @staticmethod
    def joint_angle(posA, posB, posC, eps=1e-6):
        """
        Returns angle at jointB (in degrees)
        """
        v1 = np.array(posA) - np.array(posB)
        v2 = np.array(posC) - np.array(posB)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < eps or norm2 < eps:
            return np.nan  # can't define angle
        dot = np.dot(v1, v2)
        arg = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
        angle_rad = np.arccos(arg)
        return np.degrees(angle_rad)

    def compute_joint_angles_sequence(self, frames_landmarks):
        """
        For each frame, compute angles at all specified joints.
        Returns: list of dicts { joint_name: angle_in_deg } per frame
        """
        angles_per_frame = []
        for joints in frames_landmarks:
            if joints is None:
                angles_per_frame.append(None)
                continue
            frame_angles = {}
            for (a, b, c) in self.angle_triplets:
                
                if all(j in joints for j in (a, b, c)):
                    # angle = self.joint_angle(joints[a], joints[b], joints[c])
                    
                    angle, error = compute_angles_from_three_points(joints[a], joints[b], joints[c], joints[a])[3:5]
                    frame_angles[b] = angle
                    frame_angles[b+'_error']= error
            angles_per_frame.append(frame_angles)
        return angles_per_frame





