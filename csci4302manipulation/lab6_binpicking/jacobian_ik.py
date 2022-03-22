#!/usr/bin/env python
# coding: utf-8

# In[10]:


#!jupyter nbconvert --to script jacobian_ik.ipynb


# In[2]:


import numpy as np
from math import cos, sin, atan2, acos, asin, sqrt, pi
from spatialmath import SE3, Twist3
from webots import *


# From Ryan Keating via mc-capolei
# https://raw.githubusercontent.com/mc-capolei/python-Universal-robot-kinematics/master/universal_robot_kinematics.py

# https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
# ****** Coefficients ******

# Jacobian IK solution by Michael Lauria, CU Boulder

# UR 5e

d1 = 0.1625
a2 = -0.425
a3 = -0.3922
d4 = 0.1333
d5 = 0.0997
d6 = 0.0996

d = np.array([d1, 0, 0, d4, d5, d6])
a = np.array([0, a2, a3, 0, 0, 0])
alph = np.array([pi / 2, 0, 0, pi / 2, -pi / 2, 0])


# In[3]:


def AH(n, th):

    T_a = np.array(np.identity(4), copy=False)
    T_a[0, 3] = a[n - 1]
    T_d = np.array(np.identity(4), copy=False)
    T_d[2, 3] = d[n - 1]

    Rzt = np.array(
        [
            [cos(th[n - 1]), -sin(th[n - 1]), 0, 0],
            [sin(th[n - 1]), cos(th[n - 1]), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        copy=False,
    )

    Rxa = np.array(
        [
            [1, 0, 0, 0],
            [0, cos(alph[n - 1]), -sin(alph[n - 1]), 0],
            [0, sin(alph[n - 1]), cos(alph[n - 1]), 0],
            [0, 0, 0, 1],
        ],
        copy=False,
    )

    A_i = T_d @ Rzt @ T_a @ Rxa

    return A_i


# In[4]:


def HTrans(th):
    A_1 = AH(1, th)
    A_2 = AH(2, th)
    A_3 = AH(3, th)
    A_4 = AH(4, th)
    A_5 = AH(5, th)
    A_6 = AH(6, th)

    T_01 = A_1
    T_02 = T_01 @ A_2
    T_03 = T_02 @ A_3
    T_04 = T_03 @ A_4 
    T_05 = T_04 @ A_5   
    T_06 = T_05 @ A_6
  
    transforms = [T_01, T_02, T_03, T_04, T_05, T_06]

    return transforms


# In[5]:


def get_joint_twists():
    # everything in the space frame aka base frame
    joint_twists = []
    
    # first joint
    axis = np.array([0, 0, 1]) # rotates around z, right hand rule
    point = np.array([0, 0, 0]) # a point on the axis of rotation
    twist = Twist3.UnitRevolute(axis, point)
    joint_twists.append(twist)
    
    # second joint
    axis = np.array([0, -1, 0])
    point = np.array([0, 0, d1])
    twist = Twist3.UnitRevolute(axis, point)
    joint_twists.append(twist)
    
    # third joint
    axis = np.array([0, -1, 0])
    point = np.array([a2, 0, d1])
    twist = Twist3.UnitRevolute(axis, point)
    joint_twists.append(twist)
    
    # fourth joint
    axis = np.array([0, -1, 0])
    point = np.array([a2 + a3, -d4, d1])
    twist = Twist3.UnitRevolute(axis, point)
    joint_twists.append(twist)
    
    # fifth joint
    axis = np.array([0, 0, -1])
    point = np.array([a2 + a3, -d4, d1 - d5])
    twist = Twist3.UnitRevolute(axis, point)
    joint_twists.append(twist)
    
    # sixth joint
    axis = np.array([0, -1, 0])
    point = np.array([a2 + a3, -d4 - d6, d1 - d5])
    twist = Twist3.UnitRevolute(axis, point)
    joint_twists.append(twist)
    
    return joint_twists

zero_config_fk = HTrans([0]*6)[-1]
zero_config_fk = SE3(zero_config_fk)    

def get_fk_from_twists(joint_angles):
    joint_twists = get_joint_twists()
    relative_transforms = []
    for idx, joint_twist in enumerate(joint_twists):
        angle = joint_angles[idx]
        transform = SE3(joint_twist.exp(angle))
        relative_transforms.append(transform)
        
    fk = zero_config_fk
    for transform in relative_transforms[::-1]:  # apply in reverse order
        fk = transform * fk
    return fk

def get_ur5e_jacobian_from_twists(angles, frame=None):
    if frame is None:
        frame = "body"
    joint_twists = get_joint_twists()
    relative_transforms = []
    for idx, joint_twist in enumerate(joint_twists):
        angle = angles[idx]
        relative_transforms.append(SE3(joint_twist.exp(angle)))
    jacobian = np.zeros([6, 6])
    twist_transform = SE3(np.eye(4))
    for idx in range(6):
        if idx > 0:
            twist_transform = twist_transform @ relative_transforms[idx-1]
        jacobian[:, idx] = twist_transform.Ad() @ joint_twists[idx].A  
    
    if frame == "space":
        return jacobian
    elif frame == "body":
        fk = zero_config_fk
        for transform in relative_transforms[::-1]:  # apply in reverse order
            fk = transform * fk
        return fk.inv().Ad() @ jacobian
    else:
        raise Exception(f"frame: {frame} not in (space, body)")

def get_adjoint(angles):
    current_transform = get_fk_from_twists(angles).A
    adjoint = SE3(current_transform).Ad()
    return adjoint

def get_adjoint_inverse(angles):
    current_transform = get_fk_from_twists(angles).A
    adjoint_inverse = SE3(current_transform).inv().Ad()
    return adjoint_inverse

def get_body_twist_from_transform(desired_transform, current_transform):
    """
    Even though both desired_transform and current_transform are in space frame,
    this returns a twist in the body frame.
    """
    transform_from_desired = SE3(current_transform).inv().A @ desired_transform
    twist = SE3(transform_from_desired).log(twist=True)
    return twist

def get_body_twist(angles, desired_transform):
    transforms = HTrans(angles)
    current_transform = transforms[-1]
    body_twist = get_body_twist_from_transform(desired_transform, current_transform)
    return body_twist

def get_space_twist(angles, desired_transform):
    body_twist = get_body_twist(angles, desired_transform)
    space_twist = get_adjoint(angles) @ body_twist
    return space_twist

def get_twist(angles, desired_transform, frame=None):
    if frame is None or frame == "body":
        return get_body_twist(angles, desired_transform)
    elif frame == "space":
        return get_space_twist(angles, desired_transform)
    else:
        raise Exception(f"frame: {frame} not in (space, body)")


# In[6]:


def damped_pinv(J, rho=1e-4):
    assert J.shape == (6, 6) # for UR5e, remove otherwise
    rho_squared = rho * rho
    output = J.T @ np.linalg.pinv(J @ J.T + rho_squared * np.eye(J.shape[0]))
    return output
                               
def damped_scaled_pinv(J, rho=1e-3):
    assert J.shape == (6, 6) # for UR5e, remove otherwise
    rho_squared = rho * rho
    jjt = J @ J.T
    diag_j = np.diag(np.diag(jjt)) # call np.diag twice, first to get diagonal, second to reshape
    output = J.T @ np.linalg.pinv(jjt + rho_squared * diag_j)
    return output

def get_trajectory(target, 
                   joint_angles=None, 
                   pinv_func=None, 
                   debug=False, 
                   max_iter=100,
                   learning_rate=0.1
                  ):
    if joint_angles is None:
        joint_angles = [0, 0, 0, 0, 0, 0]
        
    if pinv_func is None:
        pinv_func = np.linalg.pinv
    
    epsilon_v = 1e-4
    epsilon_w = 1e-4
    output = [joint_angles]
    
    joint_angles = np.array(joint_angles)
    FRAME = "space"
#     FRAME = "body"
    J = get_ur5e_jacobian_from_twists(joint_angles, frame=FRAME)
    J_pinv = pinv_func(J)
    twist = get_twist(joint_angles, target, frame=FRAME)
    twist[np.isnan(twist)] = 0
    
    count = 0
    norm = np.linalg.norm
    while (count < max_iter and 
           (norm(twist[:3]) > epsilon_v or norm(twist[3:]) > epsilon_w)
          ):
        step = J_pinv @ twist
        if debug:
            print(f"step: {step.round(3)}")
        joint_angles = joint_angles + learning_rate * step
        if debug:
            print(HTrans(joint_angles)[-1].round(3))
        
        J = get_ur5e_jacobian_from_twists(joint_angles, frame=FRAME)
        J_pinv = pinv_func(J)
        twist = get_twist(joint_angles, target, frame=FRAME)
        twist[np.isnan(twist)] = 0.
        if debug:
            print(f"twist: {twist.round(3)}")
        output.append(joint_angles)
        count += 1
    return output, twist


# In[7]:


def _moveTo(target, joint_angles):
    """
    move to the target homogeneous transform from current joint_angles
    """
    output, error = get_trajectory(target, 
                                   joint_angles, 
                                   pinv_func=damped_scaled_pinv)

    last_config = output[-1]
    set_motor_pos(last_config)
    for i in range(100):
        robot.step(32)
    return output

def moveToPose(pose, joint_angles):
    _moveTo(pose, joint_angles)
    
axis_map = {'x': 0, 'y': 1, 'z': 2}

def getCurrentPose():
    joint_angles = getCurrentJointAngles()
    currentPose = get_fk_from_twists(joint_angles)
    return joint_angles, currentPose.A

def moveTo(axis, value):
    idx = axis_map[axis]
    joint_angles, currentPose = getCurrentPose()
    desiredPose = currentPose.copy()
    desiredPose[idx, 3] += value
    return _moveTo(desiredPose, joint_angles), desiredPose

def getSpaceFrameError(target):
    joint_angles = getCurrentJointAngles()
    twist = get_space_twist(joint_angles, target)
    return twist

