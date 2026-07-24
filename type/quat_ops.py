import numpy as np
import math

# ==============================================================================
# JPL Quaternion Utility Functions
# Based on OpenVINS 'quat_ops.h'
# ==============================================================================

def rot_2_quat(rot):
    """
    Returns a JPL quaternion from a rotation matrix.
    Based on equation 74 in Trawny05b.
    
    :param rot: 3x3 rotation matrix (numpy array)
    :return: 4x1 quaternion [x, y, z, w] (numpy array)
    """
    q = np.zeros(4)
    T = np.trace(rot)
    
    if (rot[0, 0] >= T) and (rot[0, 0] >= rot[1, 1]) and (rot[0, 0] >= rot[2, 2]):
        q[0] = math.sqrt((1 + (2 * rot[0, 0]) - T) / 4)
        q[1] = (1 / (4 * q[0])) * (rot[0, 1] + rot[1, 0])
        q[2] = (1 / (4 * q[0])) * (rot[0, 2] + rot[2, 0])
        q[3] = (1 / (4 * q[0])) * (rot[1, 2] - rot[2, 1])

    elif (rot[1, 1] >= T) and (rot[1, 1] >= rot[0, 0]) and (rot[1, 1] >= rot[2, 2]):
        q[1] = math.sqrt((1 + (2 * rot[1, 1]) - T) / 4)
        q[0] = (1 / (4 * q[1])) * (rot[0, 1] + rot[1, 0])
        q[2] = (1 / (4 * q[1])) * (rot[1, 2] + rot[2, 1])
        q[3] = (1 / (4 * q[1])) * (rot[2, 0] - rot[0, 2])
        
    elif (rot[2, 2] >= T) and (rot[2, 2] >= rot[0, 0]) and (rot[2, 2] >= rot[1, 1]):
        q[2] = math.sqrt((1 + (2 * rot[2, 2]) - T) / 4)
        q[0] = (1 / (4 * q[2])) * (rot[0, 2] + rot[2, 0])
        q[1] = (1 / (4 * q[2])) * (rot[1, 2] + rot[2, 1])
        q[3] = (1 / (4 * q[2])) * (rot[0, 1] - rot[1, 0])
        
    else:
        q[3] = math.sqrt((1 + T) / 4)
        q[0] = (1 / (4 * q[3])) * (rot[1, 2] - rot[2, 1])
        q[1] = (1 / (4 * q[3])) * (rot[2, 0] - rot[0, 2])
        q[2] = (1 / (4 * q[3])) * (rot[0, 1] - rot[1, 0])

    # Enforce positive scalar component (unique representation)
    if q[3] < 0:
        q = -q
        
    # Normalize
    return q / np.linalg.norm(q)

def skew_x(w):
    """
    Skew-symmetric matrix from a given 3x1 vector.
    
    :param w: 3x1 vector (numpy array)
    :return: 3x3 skew-symmetric matrix
    """
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])

def quat_2_Rot(q):
    """
    Converts JPL quaternion to SO(3) rotation matrix.
    Based on equation 62 in Trawny05b.
    
    :param q: 4x1 JPL quaternion [x, y, z, w]
    :return: 3x3 SO(3) rotation matrix
    """
    # Extract vector part (x, y, z)
    q_vec = q[0:3]
    q_w = q[3]
    
    q_x = skew_x(q_vec)
    
    # R = (2*w^2 - 1)*I - 2*w*skew(v) + 2*v*v^T
    Rot = ((2 * (q_w**2) - 1) * np.eye(3)) - (2 * q_w * q_x) + (2 * np.outer(q_vec, q_vec))
    return Rot

def quat_multiply(q, p):
    """
    Multiply two JPL quaternions.
    Based on equation 9 in Trawny05b.
    
    :param q: First JPL quaternion (4x1)
    :param p: Second JPL quaternion (4x1)
    :return: Resulting q*p quaternion (4x1)
    """
    # Extract components
    q_vec = q[0:3]
    q_w = q[3]
    
    # Create Big L matrix (4x4)
    # [ w*I - skew(v)   v ]
    # [ -v^T            w ]
    
    Qm = np.zeros((4, 4))
    
    # Top-Left 3x3: w*I - skew(v)
    Qm[0:3, 0:3] = q_w * np.eye(3) - skew_x(q_vec)
    
    # Top-Right 3x1: v
    Qm[0:3, 3] = q_vec
    
    # Bottom-Left 1x3: -v^T
    Qm[3, 0:3] = -q_vec
    
    # Bottom-Right 1x1: w
    Qm[3, 3] = q_w
    
    q_t = np.dot(Qm, p)
    
    # Ensure unique by forcing q_4 (scalar) to be > 0
    if q_t[3] < 0:
        q_t *= -1
        
    return q_t / np.linalg.norm(q_t)

def vee(w_x):
    """
    Returns vector portion of skew-symmetric matrix.
    Inverse of skew_x.
    
    :param w_x: 3x3 skew-symmetric matrix
    :return: 3x1 vector
    """
    return np.array([w_x[2, 1], w_x[0, 2], w_x[1, 0]])

def exp_so3(w):
    """
    SO(3) matrix exponential mapping from vector to rotation matrix.
    Rodrigues formula.
    
    :param w: 3x1 axis-angle vector
    :return: 3x3 rotation matrix
    """
    w_x = skew_x(w)
    theta = np.linalg.norm(w)
    
    # Handle small angle values to avoid division by zero
    if theta < 1e-7:
        A = 1
        B = 0.5
    else:
        A = math.sin(theta) / theta
        B = (1 - math.cos(theta)) / (theta * theta)
        
    if theta == 0:
        return np.eye(3)
    else:
        return np.eye(3) + A * w_x + B * np.dot(w_x, w_x)

def log_so3(R):
    """
    SO(3) matrix logarithm. Maps Rotation Matrix to axis-angle vector.
    Based on Ethan Eade and GTSAM implementation for numerical stability.
    
    :param R: 3x3 SO(3) rotation matrix
    :return: 3x1 axis-angle vector
    """
    R11, R12, R13 = R[0, 0], R[0, 1], R[0, 2]
    R21, R22, R23 = R[1, 0], R[1, 1], R[1, 2]
    R31, R32, R33 = R[2, 0], R[2, 1], R[2, 2]
    
    tr = np.trace(R)
    omega = np.zeros(3)
    
    # when trace == -1 (theta near pi)
    if tr + 1.0 < 1e-10:
        if abs(R33 + 1.0) > 1e-5:
            omega = (math.pi / math.sqrt(2.0 + 2.0 * R33)) * np.array([R13, R23, 1.0 + R33])
        elif abs(R22 + 1.0) > 1e-5:
            omega = (math.pi / math.sqrt(2.0 + 2.0 * R22)) * np.array([R12, 1.0 + R22, R32])
        else:
            omega = (math.pi / math.sqrt(2.0 + 2.0 * R11)) * np.array([1.0 + R11, R21, R31])
    else:
        magnitude = 0.0
        tr_3 = tr - 3.0 # always negative
        
        if tr_3 < -1e-7:
            theta = math.acos((tr - 1.0) / 2.0)
            magnitude = theta / (2.0 * math.sin(theta))
        else:
            # Taylor expansion for small angles
            magnitude = 0.5 - tr_3 / 12.0
            
        omega = magnitude * np.array([R32 - R23, R13 - R31, R21 - R12])
        
    return omega

def exp_se3(vec):
    """
    SE(3) matrix exponential function.
    Maps 6x1 vector [omega, u] to 4x4 Transformation Matrix.
    
    :param vec: 6x1 vector [omega (3), u (3)]
    :return: 4x4 SE(3) matrix
    """
    w = vec[0:3]
    u = vec[3:6]
    theta = np.linalg.norm(w)
    wskew = skew_x(w)
    
    # Handle small angles
    if theta < 1e-7:
        A = 1.0
        B = 0.5
        C = 1.0 / 6.0
    else:
        A = math.sin(theta) / theta
        B = (1 - math.cos(theta)) / (theta * theta)
        C = (1 - A) / (theta * theta)
        
    I_33 = np.eye(3)
    V = I_33 + B * wskew + C * np.dot(wskew, wskew)
    
    mat = np.zeros((4, 4))
    mat[0:3, 0:3] = I_33 + A * wskew + B * np.dot(wskew, wskew)
    mat[0:3, 3] = np.dot(V, u)
    mat[3, 3] = 1.0
    
    return mat

def log_se3(mat):
    """
    SE(3) matrix logarithm.
    Maps 4x4 Transformation Matrix to 6x1 vector [omega, u].
    
    :param mat: 4x4 SE(3) matrix
    :return: 6x1 vector [omega, u]
    """
    w = log_so3(mat[0:3, 0:3])
    T = mat[0:3, 3]
    t = np.linalg.norm(w)
    
    if t < 1e-10:
        return np.concatenate((w, T))
    else:
        W = skew_x(w / t)
        Tan = math.tan(0.5 * t)
        WT = np.dot(W, T)
        
        # u = T - (0.5*t)*WT + (1 - t/(2*tan))* (W*WT)
        u = T - (0.5 * t) * WT + (1 - t / (2.0 * Tan)) * np.dot(W, WT)
        
        return np.concatenate((w, u))

def hat_se3(vec):
    """
    Hat operator for R^6 -> Lie Algebra se(3).
    Converts 6x1 vector to 4x4 matrix representation.
    
    :param vec: 6x1 vector [omega, u]
    :return: 4x4 matrix
    """
    mat = np.zeros((4, 4))
    mat[0:3, 0:3] = skew_x(vec[0:3])
    mat[0:3, 3] = vec[3:6]
    return mat

def Inv_se3(T):
    """
    SE(3) matrix analytical inverse.
    [ R  t ]^-1   [ R^T  -R^T*t ]
    [ 0  1 ]    = [ 0      1    ]
    
    :param T: 4x4 SE(3) matrix
    :return: Inverse 4x4 matrix
    """
    Tinv = np.eye(4)
    R_t = T[0:3, 0:3].T
    Tinv[0:3, 0:3] = R_t
    Tinv[0:3, 3] = -np.dot(R_t, T[0:3, 3])
    return Tinv

def Inv(q):
    """
    JPL Quaternion inverse.
    q_inv = [-v, w]
    
    :param q: 4x1 quaternion [x, y, z, w]
    :return: Inverse quaternion
    """
    qinv = q.copy()
    qinv[0:3] = -q[0:3] # Negate vector part
    # Scalar part remains same
    return qinv

def Omega(w):
    """
    Integrated quaternion matrix from angular velocity.
    Used for time derivative of quaternion.
    
    [ -skew(w)   w ]
    [ -w^T       0 ]
    
    :param w: 3x1 angular velocity
    :return: 4x4 Omega matrix
    """
    mat = np.zeros((4, 4))
    mat[0:3, 0:3] = -skew_x(w)
    mat[0:3, 3] = w
    mat[3, 0:3] = -w
    mat[3, 3] = 0
    return mat

def quatnorm(q_t):
    """
    Normalizes a quaternion and ensures unique representation (w > 0).
    
    :param q_t: 4x1 quaternion
    :return: Normalized unique quaternion
    """
    # Enforce positive scalar
    if q_t[3] < 0:
        q_t = -q_t
        
    return q_t / np.linalg.norm(q_t)

def Jl_so3(w):
    """
    Computes left Jacobian of SO(3).
    Used for pre-integration and error propagation.
    
    :param w: 3x1 axis-angle vector
    :return: 3x3 Left Jacobian matrix
    """
    theta = np.linalg.norm(w)
    if theta < 1e-6:
        return np.eye(3)
    else:
        a = w / theta
        # J = (sin(t)/t)*I + (1 - sin(t)/t)*a*a^T + ((1-cos(t))/t)*skew(a)
        J = (math.sin(theta) / theta) * np.eye(3) + \
            (1 - math.sin(theta) / theta) * np.outer(a, a) + \
            ((1 - math.cos(theta)) / theta) * skew_x(a)
        return J

def Jr_so3(w):
    """
    Computes right Jacobian of SO(3).
    Jr(w) = Jl(-w)
    """
    return Jl_so3(-w)

def rot2rpy(rot):
    """
    Gets roll, pitch, yaw from rotation matrix (in that order).
    R = Rz(yaw) * Ry(pitch) * Rx(roll)
    
    :param rot: 3x3 rotation matrix
    :return: 3x1 vector [roll, pitch, yaw]
    """
    rpy = np.zeros(3)
    
    # Pitch (y-axis)
    rpy[1] = math.atan2(-rot[2, 0], math.sqrt(rot[0, 0]**2 + rot[1, 0]**2))
    
    if abs(math.cos(rpy[1])) > 1.0e-12:
        rpy[2] = math.atan2(rot[1, 0] / math.cos(rpy[1]), rot[0, 0] / math.cos(rpy[1]))
        rpy[0] = math.atan2(rot[2, 1] / math.cos(rpy[1]), rot[2, 2] / math.cos(rpy[1]))
    else:
        rpy[2] = 0.0
        rpy[0] = math.atan2(rot[0, 1], rot[1, 1])
        
    return rpy

def rot_x(t):
    """
    Rotation matrix from roll angle (X-axis).
    """
    ct = math.cos(t)
    st = math.sin(t)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,  ct, -st],
        [0.0,  st,  ct]
    ])

def rot_y(t):
    """
    Rotation matrix from pitch angle (Y-axis).
    """
    ct = math.cos(t)
    st = math.sin(t)
    return np.array([
        [ ct, 0.0, st],
        [0.0, 1.0, 0.0],
        [-st, 0.0, ct]
    ])

def rot_z(t):
    """
    Rotation matrix from yaw angle (Z-axis).
    """
    ct = math.cos(t)
    st = math.sin(t)
    return np.array([
        [ ct, -st, 0.0],
        [ st,  ct, 0.0],
        [0.0, 0.0, 1.0]
    ])