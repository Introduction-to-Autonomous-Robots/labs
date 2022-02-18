#!/usr/bin/env python
# coding: utf-8

# In[7]:


try:
    get_ipython().system('jupyter nbconvert --to script supervisor_test.ipynb')
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    pass


# In[1]:


def updateCanStatus():
    for can in cans:
        x,y,z=can["translation"].getSFVec3f()
        if(x>-0.75 and x<6.8 and y>-1 and y<-0.36):
            can["status"]="CONVEYOR"
        elif(x>-0.25 and x<0.25 and y>0.48 and y<0.8):
            can["status"]="BASKET"
        else:
            can["status"]="LOST"
    


# In[2]:


#  from controller import Robot, Motor, DistanceSensor
from controller import Supervisor
import random

# create the Robot instance.
sv = Supervisor()

# get the time step of the current world.
timestep = int(sv.getBasicTimeStep())*8


ncans = 14
spacing=5


# In[3]:



# Initialize can datastructure
global cans
cans=[]
for i in range(ncans):
    can_node=sv.getFromDef('can'+str(i))
    can={}
    can["translation"]=can_node.getField("translation")
    can["rotation"]=can_node.getField("rotation")
    can["status"]="unknown"
    cans.append(can)

updateCanStatus()
sv.step(timestep)
    
nextAction = sv.getTime()
score=0


# In[4]:


while sv.step(timestep) != -1 and sv.getTime()<100:
    if(sv.getTime()>=nextAction):
        updateCanStatus()
        for can in cans:
            if(can["status"]=="LOST"):
                can["translation"].setSFVec3f([6.8,random.randrange(-130,-80)/100,0.65])
                can["rotation"].setSFRotation([0,1,0,0])
                nextAction=nextAction+spacing
                break
#    print("Time: {} Score: {}".format(sv.getTime(),score))
    pass


# In[5]:


score=0
updateCanStatus()
sv.step(timestep)


for can in cans:
    if(can["status"]=="BASKET"):
        score=score+1

print(score)


# In[6]:


for can in cans:
    print(can["status"])

