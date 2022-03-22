#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!jupyter nbconvert --to script webots.ipynb


# In[ ]:


import numpy as np

from controller import Robot
robot = Robot()
TIME_STEP=32

ur_motors = []
ur_motors.append(robot.getDevice('shoulder_pan_joint'))
ur_motors.append(robot.getDevice('shoulder_lift_joint'))
ur_motors.append(robot.getDevice('elbow_joint'))

ur_motors.append(robot.getDevice('wrist_1_joint'))
ur_motors.append(robot.getDevice('wrist_2_joint'))
ur_motors.append(robot.getDevice('wrist_3_joint'))


position_sensors = []
position_sensors.append(robot.getDevice('shoulder_pan_joint_sensor'))
position_sensors.append(robot.getDevice('shoulder_lift_joint_sensor'))
position_sensors.append(robot.getDevice('elbow_joint_sensor'))

position_sensors.append(robot.getDevice('wrist_1_joint_sensor'))
position_sensors.append(robot.getDevice('wrist_2_joint_sensor'))
position_sensors.append(robot.getDevice('wrist_3_joint_sensor'))

for ps in position_sensors:
    ps.enable(TIME_STEP)
    
for i, ur_motor in enumerate(ur_motors):
    ur_motor.setVelocity(1)
    
def getCurrentJointAngles():
    return np.array([ps.getValue() for ps in position_sensors])

def goToPose(config):
    for i, ur_motor in enumerate(ur_motors):
        ur_motor.setVelocity(1)
        ur_motor.setPosition(config[i])

    for i in range(100):
        robot.step(32)
        
def set_motor_pos(config):
    for i,um in enumerate(ur_motors):
        um.setPosition(config[i])

