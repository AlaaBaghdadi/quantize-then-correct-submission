

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
    if ( theta3_deg-90 > 90 or theta3_deg-90 < -90 or theta4_deg+90 < 0 or theta4_deg > 170 or theta1_deg < -75 or theta1_deg > 90 or theta2_deg < -10 or theta2_deg > +90 ):
        print("error!!!")
    return theta1_deg, theta2_deg, theta3_deg, theta4_deg
