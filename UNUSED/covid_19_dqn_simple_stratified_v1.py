def dqnv1():
    # -*- coding: utf-8 -*-
    """Covid-19 DQN Simple Stratified v1.ipynb
    
    Automatically generated by Colaboratory.
    
    Original file is located at
        https://colab.research.google.com/drive/1rlqAp6jWWBmoUHUoxAAIV31xrqxSnPVJ
    """
    
    #STRATIFIED VERSION
    #ELU invece che ReLU, terminate reward con win/loose, few episodes, done in memoria (come usarlo? reward molto diversi per done e not done)
    import torch
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    import time
    import math
    import random
    from numpy import exp
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    import os
    import shutil
    
    my_path = os.path.dirname(os.path.abspath(__file__))
    
    MAX_RUN = 52
    FRESH_TIME = 0.1
    
    #torch.cuda.get_device_name(0)
    
    model_save_name_1 = 'SimpleStratifiedWeights2.pth'
    pathMS1 = os.path.join(my_path,model_save_name_1)
    images_dir = os.path.join(my_path,'static\\images\\Simple Stratified 3 Layers')
    
    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
        os.mkdir(images_dir)
    else:
        os.mkdir(images_dir)
    
    def F(x_0, t,beta0):
      """
      Time derivative of the state vector.
    
        * x is the state vector (array_like)
        * t is time (scalar)
        * beta0 is the passed parameter 
    
      """
      beta0_u, beta0_m, beta0_o = beta0
      #b3_u, b3_m, b3_o = b3  #b3 variable depending on action
      #v_u, v_m, v_o = v      #vaccine variable depending on action
    
    
      x_u=x_0[0:5]
      x_m=x_0[5:10]
      x_o=x_0[10:15]
      s_u, i1_u, i2_u, r1_u, r2_u = x_u   #unpack the state
      s_m, i1_m, i2_m, r1_m, r2_m = x_m 
      s_o, i1_o, i2_o, r1_o, r2_o = x_o
    
      #r3_tot=r3_u+r3_m...=1-s_u-s_m-s_o-i1_u-..... $Da verificare
      ds_u=-beta0_u*s_u*(i1_u+i1_m+i1_o+i2_u+i2_m+i2_o)+d1*r2_u
      di1_u=beta0_u*s_u*(i1_u+i1_m+i1_o+i2_u+i2_m+i2_o)-(b1+b2)*i1_u
      di2_u=b2*i1_u-c1*i2_u-b3_u*i2_u
      dr1_u=b1*i1_u-(c2+c3_u)*r1_u
      dr2_u=c1*i2_u+c2*r1_u-d1*r2_u
    
      ds_m=-beta0_m*s_m*(i1_u+i1_m+i1_o+i2_u+i2_m+i2_o)+d1*r2_m
      di1_m=beta0_m*s_m*(i1_u+i1_m+i1_o+i2_u+i2_m+i2_o)-(b1+b2)*i1_m
      di2_m=b2*i1_m-c1*i2_m-b3_m*i2_m
      dr1_m=b1*i1_m-(c2+c3_m)*r1_m
      dr2_m=c1*i2_m+c2*r1_m-d1*r2_m
    
      ds_o=-beta0_o*s_o*(i1_u+i1_m+i1_o+i2_u+i2_m+i2_o)+d1*r2_o
      di1_o=beta0_o*s_o*(i1_u+i1_m+i1_o+i2_u+i2_m+i2_o)-(b1+b2)*i1_o
      di2_o=b2*i1_o-c1*i2_o-b3_o*i2_o
      dr1_o=b1*i1_o-(c2+c3_o)*r1_o
      dr2_o=c1*i2_o+c2*r1_o-d1*r2_o
    
      #state equations
      #ds=-beta0*s*(i1+i2)+d1*r2
      #di1=beta0*s*(i1+i2)-(b1+b2)*i1
      #di2=b2*i1-c1*i2-b_3*i2
      #dr2=(c1+c2*b1/(b2-b1))*i2-d1*r2
    
      return ds_u, di1_u, di2_u, dr1_u, dr2_u, ds_m, di1_m, di2_m, dr1_m, dr2_m, ds_o, di1_o, di2_o, dr1_o, dr2_o
    
    
    
    
    def spread_virus(t, beta0, x):  #funzione di simulazione pandemia
      #G = lambda x, t: F(x, t, beta0,b_3)
      x_0=x[0:-2]
      G = lambda x, t: F(x, t, beta0)#non considero b_3 (tampone)
      s_u, i1_u, i2_u, r1_u, r2_u, s_m, i1_m, i2_m, r1_m, r2_m, s_o, i1_o, i2_o, r1_o, r2_o = odeint(G, x_0, t).transpose() #solve ode system
            # per risultati cumulativi:
            # for i in range (len(s)):
            #     i1_tot += beta0*s(i)*(i1(i)+i2(i))
            #     guariti_tot +=(c1+c2*b1/b2)*i2(i)
        #r1_path = (1/(b2/b1-1))*i2_path     # cumulative cases
        #r3_path = c3*r1_path
      return s_u, i1_u, i2_u, r1_u, r2_u, s_m, i1_m, i2_m, r1_m, r2_m, s_o, i1_o, i2_o, r1_o, r2_o
    
    def update_env(INFECTED, STEP, ACTION, REWARD):#outuput stream
        print("Week %d, Action: %s, %.5f Infected, Reward %.2f" % (STEP, ACTION, INFECTED, REWARD))
        time.sleep(FRESH_TIME)
    
    class Environment(object):
        def __init__(self,perc):
            super(Environment, self).__init__()
            self.action_space = ['all_open', 'OS_HB' ,'OS_LB', 'HS_OB', 'HS_HB', 'HS_LB', 'LS_OB','LS_HB', 'all_closed']  # possible actions 
            self.n_actions = len(self.action_space) 
            perc_u,perc_o,perc_m=perc
            #$$percentuali di popolazione (https://www.tuttitalia.it/statistiche/popolazione-eta-sesso-stato-civile-2019/)
            self.perc_u=perc_u #percentage of under 30 y.o italian people
            self.perc_o=perc_o #percentage of over 65 y.o italian people
            self.perc_m=1-self.perc_u-self.perc_o #percentage of over 30 and under 65 y.o italian people
            #This proportion will be used in assesing the number of infected people and to plot results
    
            self.i1_u_0 = random.uniform(5e-2, 2e-1)
            self.i1_m_0 = random.uniform(5e-2, 2e-1)
            self.i1_o_0 = random.uniform(5e-2, 2e-1)
    
            #state x= s,i1,i2,r1,r2
            self.x_u = (1.-self.i1_u_0,self.i1_u_0,0,0,0)
            self.x_m = (1.-self.i1_m_0,self.i1_m_0,0,0,0)
            self.x_o = (1.-self.i1_o_0,self.i1_o_0,0,0,0)
    
            self.state=np.concatenate((self.x_u,self.x_m),axis=None)
            self.state=np.concatenate((self.state,self.x_o),axis=None)
            self.state=np.concatenate((self.state,0),axis=None) #add step info to the state
            self.state=np.concatenate((self.state,0),axis=None) #add done info to the state
    
            self.beta_scuole=0.16
            self.beta_bus=0.04
            self.beta=self.beta_scuole+self.beta_bus
            self.beta_u= self.beta*10 #under 30 are more probable to get/spread the virus
            self.beta_m= self.beta
            self.beta_o= self.beta/10 #elder people should be the more careful ones
            self.beta_0=(self.beta_u,self.beta_m,self.beta_o)
    
            #self.previous_infected=self.x_u[1]+self.x_m[1]+self.x_o[1] 
            self.last_action = None
            t_stop=5 #batch size
            self.t_prev=2*t_stop
            self.t_vec=np.linspace(0,self.t_prev-1,self.t_prev)
            self.c3_u = 0 #$devo fare una media pesata
            self.c3_m = 0.0063*self.perc_m
            self.c3_o = 0.1423*self.perc_o
            self.w_school=1
            self.w_bus=1
            self.end_episode=0.00001 #1 person every 100 thousand people
            self.rew_cum=0
            self.action_list='Start'
            self.history_u=np.array([self.x_u[0],
                                    self.x_u[1],
                                    self.x_u[2],
                                    self.x_u[3],
                                    self.x_u[4]])
                
            self.history_m=np.array([self.x_m[0],
                                    self.x_m[1],
                                    self.x_m[2],
                                    self.x_m[3],
                                    self.x_m[4]])
                
            self.history_o=np.array([self.x_o[0],
                                    self.x_o[1],
                                    self.x_o[2],
                                    self.x_o[3],
                                    self.x_o[4]])
            self.history_prev_u=self.history_u
            self.history_prev_m=self.history_m
            self.history_prev_o=self.history_o
        def reset(self):
            self.i1_u_0 = random.uniform(5e-2, 3e-1)
            self.i1_m_0 = random.uniform(5e-2, 3e-1)
            self.i1_o_0 = random.uniform(5e-2, 3e-1)
            #state x= s,i1,i2,r1,r2
            self.x_u = (1.-self.i1_u_0,self.i1_u_0,0,0,0)
            self.x_m = (1.-self.i1_m_0,self.i1_m_0,0,0,0)
            self.x_o = (1.-self.i1_o_0,self.i1_o_0,0,0,0)
    
            self.state=np.concatenate((self.x_u,self.x_m),axis=None)
            self.state=np.concatenate((self.state,self.x_o),axis=None)
            self.state=np.concatenate((self.state,0),axis=None) #add step info to the state
            self.state=np.concatenate((self.state,0),axis=None) #add done info to the state
    
            self.beta_scuole=0.16
            self.beta_bus=0.04
            self.beta=self.beta_scuole+self.beta_bus
            self.beta_u= self.beta*10 #under 30 are more probable to get/spread the virus
            self.beta_m= self.beta
            self.beta_o= self.beta/10 #elder people should be the more careful ones
            self.beta_0=(self.beta_u,self.beta_m,self.beta_o)
            self.w_school=1
            self.w_bus=1
            self.rew_cum=0
            #self.previous_infected=self.x_u[1]+self.x_m[1]+self.x_o[1] 
            self.action_list='Start'
    
            self.history_u=np.array([self.x_u[0],
                                    self.x_u[1],
                                    self.x_u[2],
                                    self.x_u[3],
                                    self.x_u[4]])
                
            self.history_m=np.array([self.x_m[0],
                                    self.x_m[1],
                                    self.x_m[2],
                                    self.x_m[3],
                                    self.x_m[4]])
                
            self.history_o=np.array([self.x_o[0],
                                    self.x_o[1],
                                    self.x_o[2],
                                    self.x_o[3],
                                    self.x_o[4]])
            self.history_prev_u=self.history_u
            self.history_prev_m=self.history_m
            self.history_prev_o=self.history_o
            
            #return np.array([self.x_u,self.x_m,self.x_o])
            return self.state
        
        def instant_reward(self, delta_infect, step):
          #r1 = 1 - infect # [0, 1), sono i "non infetti", tanti piu` infetti, tanto minore e` il reward conseguente
          #if self.infected < 1e-3:
          #  weight_delta=50
          weight_delta=1
          weight_socioecon=1/50
          weight_time=20
          self.i1_0=self.i1_u_0 + self.i1_m_0 + self.i1_o_0 
          #rew_inf=-self.infected/self.i1_0*weight_delta#reward for infect improvement
          rew_inf=-self.infected/self.end_episode*weight_delta#reward for infect improvement
          
          self.rew_cum=self.rew_cum+rew_inf
          #print(f"rew_inf is {rew_inf}, rew_socioecon is {rew_socioecon}, rew_time is {rew_time} at step {step}")
          return rew_inf
          
        
        def terminate_reward(self, delta_infect, step):
          
          
          return self.rew_cum
          
     
     
     
        def step(self, action, step, show):
          #s suscptible
          #i1 infected
          #i2 undetected
          #r1 hospitalized
          #r2 recovered
    
          #check what would happen if no action are taken (same parameters as last cycle)
          self.prev_s_u, self.prev_i1_u, self.prev_i2_u, self.prev_r1_u, self.prev_r2_u, self.prev_s_m, self.prev_i1_m, self.prev_i2_m, self.prev_r1_m, self.prev_r2_m, self.prev_s_o, self.prev_i1_o, self.prev_i2_o, self.prev_r1_o, self.prev_r2_o = spread_virus(self.t_vec+step*self.t_prev, self.beta_0, self.state) #prevision of the trend if no action is taken
          #self.prev_infected=self.prev_infected[-1]+self.prev_undetected[-1]
          self.prev_infected = self.perc_u*(self.prev_i1_u[-1] + self.prev_i2_u[-1]+self.prev_r1_u[-1]) + self.perc_m*(self.prev_i1_m[-1] + self.prev_i2_m[-1]+self.prev_r1_m[-1]) + self.perc_o*(self.prev_i1_o[-1] + self.prev_i2_o[-1]+self.prev_r1_o[-1]) 
          self.action_list=np.concatenate((self.action_list,self.action_space[action]),axis=None)
          
          if (self.action_space[action] == 'all_open'): # a seconda dell'azione bisogna cambiare il parametro R0 e volendo B_3
          #pl e` patient_list,
            self.beta=0.2
          elif (self.action_space[action] == 'OS_HB'):
            self.beta=0.18
          elif (self.action_space[action] == 'OS_LB'):
            self.beta=0.16
          elif (self.action_space[action] == 'HS_OB'):
            self.beta=0.12
          elif (self.action_space[action] == 'HS_HB'):
            self.beta=0.1
          elif (self.action_space[action] == 'HS_LB'):
            self.beta=0.08
          elif (self.action_space[action] == 'LS_OB'):
            self.beta=0.04
          elif (self.action_space[action] == 'LS_HB'):
            self.beta=0.02
          else: # (action == 'all_closed'):
            self.beta=0 # non faccio niente
          #self.beta=self.w_school*self.beta_scuole + self.w_bus*self.beta_bus
    
          self.beta_u= self.beta*10 #under 30 are more probable to get/spread the virus
          self.beta_m= self.beta
          self.beta_o= self.beta/10 #elder people should be the more careful ones
          self.beta_0=(self.beta_u,self.beta_m,self.beta_o)
    
    
          self.current_s_u, self.current_i1_u, self.current_i2_u, self.current_r1_u, self.current_r2_u, self.current_s_m, self.current_i1_m, self.current_i2_m, self.current_r1_m, self.current_r2_m, self.current_s_o, self.current_i1_o, self.current_i2_o, self.current_r1_o, self.current_r2_o = spread_virus(self.t_vec+step*self.t_prev, self.beta_0, self.state) #solve_path
    
          #self.current_susceptible, self.current_infected, self.current_undetected, self.current_recovered= spread_virus(self.t_vec, self.beta, self.state) #solve_path
          
          self.infected = self.perc_u*(self.current_i1_u[-1] + self.current_i2_u[-1]+self.current_r1_u[-1]) + self.perc_m*(self.current_i1_m[-1] + self.current_i2_m[-1] + self.current_r1_m[-1]) + self.perc_o*(self.current_i1_o[-1] + self.current_i2_o[-1]++self.current_r1_o[-1]) 
          #self.infected=self.current_infected[-1]+self.current_undetected[-1] #total infected 
          
          self.delta_infected =  self.prev_infected-self.infected #check if the prevision is better than the actual infect state, used for reward
          
     
          self.x_u  = (self.current_s_u[-1], self.current_i1_u[-1], self.current_i2_u[-1], self.current_r1_u[-1], self.current_r2_u[-1])
          self.x_m  = (self.current_s_m[-1], self.current_i1_m[-1], self.current_i2_m[-1], self.current_r1_m[-1], self.current_r2_m[-1])
          self.x_o  = (self.current_s_o[-1], self.current_i1_o[-1], self.current_i2_o[-1], self.current_r1_o[-1], self.current_r2_o[-1])
          
          
          #self.state = (self.current_susceptible[-1], self.current_infected[-1], self.current_undetected[-1], self.current_recovered[-1]) #take last element of each block as current state
          #print([self.state,action,self.beta,self.delta_infected])
          reward = self.instant_reward(self.delta_infected, step)
     
          if (step >= 100) or (self.infected >= 0.8) or (self.infected<=self.end_episode) : #end episode condition,
            done = True
            if self.infected<=self.end_episode:
              reward = -1/self.terminate_reward(self.delta_infected, step)
            else:
              reward = self.terminate_reward(self.delta_infected, step)#The whole episode reward is different from each step reward giving more weight to the time passed
            #print("tr:", reward)
          else:
            done = False
            
          self.state=np.concatenate((self.x_u,self.x_m),axis=None)
          self.state=np.concatenate((self.state,self.x_o),axis=None)
          self.state=np.concatenate((self.state,step),axis=None) #add step info to the state
          self.state=np.concatenate((self.state,done),axis=None) #add step info to the state
          update_env(self.infected, step, self.action_space[action], reward)
          time.sleep(0.02)#$ si può togliere?
    
          #save history to plot 
          if show:
            
            self.history_u=np.array([np.concatenate((self.history_u[0],self.current_s_u[1:-1]),axis=None),
                                  np.concatenate((self.history_u[1],self.current_i1_u[1:-1]),axis=None),
                                  np.concatenate((self.history_u[2],self.current_i2_u[1:-1]),axis=None),
                                  np.concatenate((self.history_u[3],self.current_r1_u[1:-1]),axis=None),
                                  np.concatenate((self.history_u[4],self.current_r2_u[1:-1]),axis=None)])
            self.history_m=np.array([np.concatenate((self.history_m[0],self.current_s_m[1:-1]),axis=None),
                                  np.concatenate((self.history_m[1],self.current_i1_m[1:-1]),axis=None),
                                  np.concatenate((self.history_m[2],self.current_i2_m[1:-1]),axis=None),
                                  np.concatenate((self.history_m[3],self.current_r1_m[1:-1]),axis=None),
                                  np.concatenate((self.history_m[4],self.current_r2_m[1:-1]),axis=None)])
            self.history_o=np.array([np.concatenate((self.history_o[0],self.current_s_o[1:-1]),axis=None),
                                  np.concatenate((self.history_o[1],self.current_i1_o[1:-1]),axis=None),
                                  np.concatenate((self.history_o[2],self.current_i2_o[1:-1]),axis=None),
                                  np.concatenate((self.history_o[3],self.current_r1_o[1:-1]),axis=None),
                                  np.concatenate((self.history_o[4],self.current_r2_o[1:-1]),axis=None)])
            
            
            self.history_prev_u=np.array([np.concatenate((self.history_prev_u[0],self.prev_s_u[1:-1]),axis=None),
                                  np.concatenate((self.history_prev_u[1],self.prev_i1_u[1:-1]),axis=None),
                                  np.concatenate((self.history_prev_u[2],self.prev_i2_u[1:-1]),axis=None),
                                  np.concatenate((self.history_prev_u[3],self.prev_r1_u[1:-1]),axis=None),
                                  np.concatenate((self.history_prev_u[4],self.prev_r2_u[1:-1]),axis=None)])
            self.history_prev_m=np.array([np.concatenate((self.history_prev_m[0],self.prev_s_m[1:-1]),axis=None),
                                  np.concatenate((self.history_prev_m[1],self.prev_i1_m[1:-1]),axis=None),
                                  np.concatenate((self.history_prev_m[2],self.prev_i2_m[1:-1]),axis=None),
                                  np.concatenate((self.history_prev_m[3],self.prev_r1_m[1:-1]),axis=None),
                                  np.concatenate((self.history_prev_m[4],self.prev_r2_m[1:-1]),axis=None)])
            self.history_prev_o=np.array([np.concatenate((self.history_prev_o[0],self.prev_s_o[1:-1]),axis=None),
                                  np.concatenate((self.history_prev_o[1],self.prev_i1_o[1:-1]),axis=None),
                                  np.concatenate((self.history_prev_o[2],self.prev_i2_o[1:-1]),axis=None),
                                  np.concatenate((self.history_prev_o[3],self.prev_r1_o[1:-1]),axis=None),
                                  np.concatenate((self.history_prev_o[4],self.prev_r2_o[1:-1]),axis=None)])
            
                                            
      
          return np.array(self.state), reward, done
          
        def action_available(self,action):
          return True
     
          if self.beta==0.2 and self.action_space[action] == 'all_open':
              return False
          if self.beta==0.18 and self.action_space[action] == 'OS_HB':
              return False
          if self.beta==0.16 and self.action_space[action] == 'OS_LB':
              return False
          if self.beta==0.12 and self.action_space[action] == 'HS_OB':
              return False
          if self.beta==0.1 and self.action_space[action] == 'HS_HB':
              return False
          if self.beta==0.08 and self.action_space[action] == 'HS_LB':
              return False
          if self.beta==0.04 and self.action_space[action] == 'LS_OB':
              return False
          if self.beta==0.02 and self.action_space[action] == 'LS_HB':
              return False
          if self.beta==0 and self.action_space[action] == 'all_closed':
              return False
          return True
        def action_available_2(self,b_b,b_a_):
          
          check_vec= np.ones((len(b_a_),), dtype=bool) #initial guess that all action to be taken are available
          return check_vec
          idx=(b_b==0.2 )
          idx1=(b_a_[idx[:,0]]==0) 
          if any(idx1):
            check_vec[idx[:,0]][idx1]=0 #if b_w of that experience is already 1 I can't open bus (next action = 5)
          idx=(b_b==0.18)
          idx1=(b_a_[idx[:,0]]==1) 
          if any(idx1):
            check_vec[idx[:,0]][idx1]=0 #if b_w of that experience is already 0.5 I can't half bus (next action = 4)
          idx=(b_b==0.16)
          idx1=(b_a_[idx[:,0]]==2) 
          if any(idx1):
            check_vec[idx[:,0]][idx1]=0 #if b_w of that experience is already 0 I can't close bus (next action = 3)
          idx=(b_b==0.12)
          idx1=(b_a_[idx[:,0]]==3) 
          if any(idx1):
            check_vec[idx[:,0]][idx1]=0 #if b_school of that experience is already 1 I can't open school (next action = 2)
          idx=(b_b==0.1)
          idx1=(b_a_[idx[:,0]]==4) 
          if any(idx1):
            check_vec[idx[:,0]][idx1]=0 #if b_school of that experience is already 0.5 I can't DAD bus (next action = 1)
          idx=(b_b==0.08)
          idx1=(b_a_[idx[:,0]]==5) 
          if any(idx1):
            check_vec[idx[:,0]][idx1]=0 #if b_school of that experience is already 0 I can't close school  (next action = 0)
          idx=(b_b==0.04)
          idx1=(b_a_[idx[:,0]]==6) 
          if any(idx1):
            check_vec[idx[:,0]][idx1]=0 #if b_school of that experience is already 0 I can't close school  (next action = 0)
          idx=(b_b==0.02)
          idx1=(b_a_[idx[:,0]]==7) 
          if any(idx1):
            check_vec[idx[:,0]][idx1]=0 #if b_school of that experience is already 0 I can't close school  (next action = 0)
          idx=(b_b==0)
          idx1=(b_a_[idx[:,0]]==8) 
          if any(idx1):
            check_vec[idx[:,0]][idx1]=0 #if b_school of that experience is already 0 I can't close school  (next action = 0)
         
          #print(f"idx is {idx[:,0]}")
          #print(f" b_a of idx is {b_a_[idx[:,0]]}")
    
          return check_vec #vec containing zeros for action which can not be taken and ones for actions available
    
    class DeepQNetwork(nn.Module):
      def __init__(self):
        super(DeepQNetwork, self).__init__()#inherit characteristic of DeepQNetwork class
        self.fc = nn.Sequential(
            #nn.Linear(2, 24),
            nn.Linear(17, 24), #17 input =state, 5 for each popoluation age + 1 for time +1 for done
            nn.ELU(),
            nn.Linear(24, 24),
            nn.ELU(),
            nn.Linear(24, 9) #7 output = q values: one for each action to be taken
        )
        self.mls = nn.MSELoss() #mean square error criterion, possible to chose square error or sum of square error
        self.opt = torch.optim.Adam(self.parameters(), lr=0.1)# first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments
        #more about adam at https://arxiv.org/abs/1412.6980
      
      def forward(self, inputs):
        return self.fc(inputs)
    
    class Explore():#explore function, given a time step, it returns a value from end to start which gradually decay to end which is not zero because always explore
      def __init__(self,start,end,decay):
        self.start=start
        self.end=end
        self.decay=decay
    
      def get_exploration_rate(self,current_step):
        return self.end + (self.start - self.end)*math.exp(-1.*current_step*self.decay)
    
    def plot_env(env,txt,episode):
      with plt.style.context('classic'):
        inf=env.history_u[1]
        prev_inf=env.history_prev_u[1]
        fig, ax = plt.subplots(figsize=(12, 4), num='classic')
        plt.plot(inf,'r',linewidth=2,label='actual')
    
        # Add labels to the plot
        style = dict(size=10, color='black')
        for i in range(1,len(env.action_list)):
          if env.action_list[i]!=env.action_list[i-1]:
            idx_inf=max(range(7,len(inf)), key=inf.__getitem__)
            index_action=1+8*(i-1)
            inf_action=inf[index_action-1]
            #print(f"AZIONE DIVERSA: passo {index_action} valore {inf_action}")
            if index_action>idx_inf or index_action==1:
              offset=20
            else:
              offset=-20
            #ax.text(7*i,inf[7*i], env.action_list[i], **style)
            
            ax.annotate(env.action_list[i], xy=(index_action, inf_action),xycoords='data',xytext=(index_action+offset, 0.3+0.1*math.sin(index_action)+inf_action),bbox=dict(boxstyle="round", alpha=0.2),arrowprops=dict(arrowstyle="fancy"))
        # Label the axes
        ax.set(title='INFECTED',
              ylabel='infected %')
        
        ax.set_xlabel(f"days\n\n{txt}")
    
        plt.ylim([0, max(max(inf),max(prev_inf))])
    
        plt.plot(prev_inf,'k--',linewidth=1,label='previsions')
        plt.legend()
        
      
        fig.set_size_inches(20, 10, forward=True)
    
        name = '1_'+str(episode)+'.png'
        ptf = os.path.join(images_dir,name)
        plt.savefig(ptf)
        plt.show()
        #plt.savefig(f"{images_dir}/image1_{episode}.png")
        #plt.savefig("name.png")
        #files.download('name.png')
        #print(env.action_list)
    
    #PARAMETER DEFINITION
    memory_size = 5000 #Maximum size of the memory
    #decline = 0.00006
    update_time = 50 #Set how often update the target network
    gamma = 0.9 #Discount factor: how much we weight the old q-value wrt to the new one
    b_size = 1600 #batch size: dimension of each set of experiences sent to the network to be updated
    memory = np.zeros((memory_size, 38)) # 32 = State(17) + Action(1) + Next State(17) + Reward(1) + Beta(1) + Done(1)
    
    EPISODES=200
    show_every=3
    
    
    perc_u=0.28 #percentage of under 30 y.o italian people
    perc_o=0.23 #percentage of over 65 y.o italian people
    perc_m=1-perc_u-perc_o #percentage of over 30 and under 65 y.o italian people
    perc=np.array([perc_u,perc_o,perc_m])
    
    t1=5          #incubation period
    t2=17         #disease time
    
    
    # $$ COEFFICENTI VARI DA RIVISIONARE, NON REGGE PIù L'IPOTESI tra C1 C2 C3 e b1 b2
    b1=0.3/t1     #detection rate, from i1 to r1
    b2=0.7/t1     #undetection rate, from i1 to i2
    
    
    c1=1/t2       #from i2 to r2, undetected healing ratio
    c2=0.8/t2     #from r1 to r2, detected healing ratio
    c3=1/t2-c2    #Fatality https://www.statista.com/statistics/1106372/coronavirus-death-rate-by-age-group-italy/
    
    c3_u = 0 #$devo fare una media pesata
    c3_m = 0.0063*perc_m
    c3_o = 0.1423*perc_o
    
    d1=0.001      #0.03  %back to susceptible (fine anticorpi)
    b_3=0.05      #simulation of tampon
    b3_u = b_3 
    b3_m = b_3 
    b3_o = b_3
    
    
    
    action_count=np.array([0,0,0,0,0,0,0,0,0])
    
    epsilon = 0.9 #random action to explore stuff
    maxeps=1
    mineps=0.01
    epsilon_decay_value=5e-4
    
    def run_deepQ(env, net, net2):
      convergence_episode=0
      loss_list=np.array([0,0])
      #print(f"FATALITY RATES c3_u is {c3_u} c3_m is {c3_m} c3_o is {c3_o}")
      memory_count = 0
      learn_time = 0
    
      strategy=Explore(maxeps,mineps,epsilon_decay_value)
      show= False
    
      for episode in range(EPISODES):
        if show:
          #$super duper cool https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/04.09-Text-and-Annotation.ipynb#scrollTo=-aadvkYjQWoV
          #fig=plt.figure()
          txt="Stratified With Reward -self.infected/self.end_episode*1 and 3 layer Networks"
          plot_env(env,txt,episode)
        if episode%show_every:
          show= False
        else:
          show= True
    
        if episode>=(EPISODES-1):
          plt.plot(np.log10(loss_list))
          print(f"loss list is {loss_list}")
          print(f"convergence episode is {convergence_episode}")
          print(f"{action_count[0]} all_open, {action_count[1]} OS_HB, {action_count[2]} OS_LB, {action_count[3]} HS_OB, {action_count[4]} HS_HB, {action_count[5]} HS_LB, {action_count[6]} LS_OB, {action_count[7]} LS_HB, {action_count[8]} all_close")
        observation = env.reset()
        step = 0
        print(f" LEARNING PROCESS AT {100*episode/EPISODES}% ")
        while True:
          if (learn_time < 10):
            if (memory_count < memory_size):
              print(learn_time, memory_count)
            print(learn_time)
          #if random.randint(0, 100) < 0.001 * strategy.get_exploration_rate(learn_time): #use this to avoid exploration
          if random.randint(0, 100) < 100 * strategy.get_exploration_rate(learn_time):
            print("Explore")
            action = random.randint(0, env.n_actions-1)  # Randomly pick an action
            while not env.action_available(action):
              action = random.randint(0, env.n_actions-1)  # Randomly pick an available action
          else:
            print("Exploit")
            out = net(torch.Tensor(observation).flatten()).detach()  # [left_rewards_total, right_reward_total]
            action_vec= out.data
            action = torch.argmax(action_vec).item()
            while not env.action_available(action):
              action_vec[action]=torch.min(action_vec).item()
              action = torch.argmax(action_vec).item()
            action_count[action]=  action_count[action] + 1
          observation_, reward, done = env.step(action, step, show)
          
    
          idx = memory_count % memory_size
          memory[idx][0:17] = observation
          memory[idx][17:18] = action
          memory[idx][18:35] = observation_
          memory[idx][35:36] = reward
          memory[idx][36:37] = env.beta
          memory[idx][37:38] = done
          
          memory_count +=1
          
          observation = observation_ 
    
          if (memory_count >= memory_size):  # Start to learn
            learn_time += 1 # Learn once
            if (learn_time % update_time == 0): # Sync two nets
              net2.load_state_dict(net.state_dict())
              print("Sync Two Net")
            else:
              rdp = random.randint(0, memory_size - b_size - 1)
              b_s = torch.Tensor(memory[rdp:rdp+b_size, 0:17])
              b_a = torch.Tensor(memory[rdp:rdp+b_size, 17:18]).long()
              b_s_ = torch.Tensor(memory[rdp:rdp+b_size, 18:35])
              b_r = torch.Tensor(memory[rdp:rdp+b_size, 35:36])
              b_b = torch.Tensor(memory[rdp:rdp+b_size, 36:37])      #batch consequent beta0
              b_d= torch.Tensor(memory[rdp:rdp+b_size, 37:38])  #done
    
              q = net(b_s).gather(1, b_a)       #take the q values as output of the policy net given the actual state as input and chose those of the action taken stored in b_a
              out_available=net2(b_s_).detach() #take best q_next as output of the target net given the next state as input corrisponding to the best action
              b_a_= torch.argmax(out_available,dim=1).numpy()
              check_vec = env.action_available_2(b_b.numpy(),b_a_) #best next_q value only legit if the corrisponding action can be taken
              while not all(check_vec):
                out_available[not check_vec,b_a_]=torch.min(out_available[not check_vec,:]).item()
                b_a_ = torch.argmax(out_available,dim=1).numpy()
                check_vec = env.action_available_2(b_b.numpy(),b_a_)
              
              q_next = out_available.max(1)[0].reshape(b_size, 1) #best possible next_q value
              tq = b_r + gamma * q_next  #Bellman equation to compute the loss function
              loss = net.mls(q, tq) #loss function as MSE between q values and optimal function (rhs of bellman equation)
              #print(f"LOSS is{loss.detach().numpy()}")
              loss_list=np.concatenate((loss_list,loss.detach().numpy()),axis=None)
              if loss.detach().numpy()<0.01 and convergence_episode==0:
                convergence_episode=episode
              net.opt.zero_grad() #optimization (use TRPO instead?)
              loss.backward() #policy net weights updated based on loss function
              net.opt.step()
    
          step +=1
          if done:
            break
    
    net = DeepQNetwork() #define policy net
    net2 = DeepQNetwork() #define target net
    env = Environment(perc) #define environment object
    state_dict=torch.load(pathMS1) #load weight file from drive
    net.load_state_dict(state_dict) #upload weights in policy net 
    net2.load_state_dict(state_dict) #upload weights in target net
    run_deepQ(env, net, net2)
    
    torch.save(net.state_dict(),pathMS1) #save weights file in Drive
    
#dqnv1()