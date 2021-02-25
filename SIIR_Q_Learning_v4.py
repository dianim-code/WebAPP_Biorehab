def siirqt(from0,episodes_IN,beta1_IN,beta2_IN): 
    import numpy as np
    from numpy import exp
    import math
    import random
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    import os
    import shutil

    my_path = os.path.dirname(os.path.abspath(__file__))
    global images_dir
    images_dir = os.path.join(my_path,'static\\images\\SIIRQT')

    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
        os.mkdir(images_dir)
    else:
        os.mkdir(images_dir)
    
    
    
    # ## v4 update: abbiamo usato i1+i2 per prevision,
    # abbiamo usato l'integrale del flusso nel reward
    # reward con 2 premi e 2 penalty
    # reward si comporta diversa con explore vs exploit
    # reward con funzione esponenziale del tempo
    
    
    # TO DO: misura performance
    #     AZIONE NON REALISTICHE, come cambiano i coefficenti in realta'???
    #     
    #     spottare stati non utilizzati (e.g. 50 50 50 50)
    #     pensare se aggiungere lo stato timestep
    #     pensare se usare i morti (desemplificare il modello)
    
    
    
    ## Inizializazione SIR
    pop_size = 1e3
    
    t_fin=550 #days
    t_stop=5 #batch size
    t_prev=2*t_stop
    t_vec=np.linspace(0,t_prev-1,t_prev)
    #print(t_vec)
    
    
    t1=5          #periodo di incubazione
    t2=17         #durata malattia
    
    b1=0.3/t1     #detection rate, da i1 a r1
    b2=0.7/t1     #undetection rate, da i1 a i2
    
    
    c1=1/t2       #Da i2 a r2, rateo guarigione undetected 
    c2=0.8/t2     #Da r1 a r2, rateo guarigione detected
    c3=1/t2-c2    #Fatality 
    
    d1=0.001      #0.03  %ritorno a suscettibili (fine anticorpi)
    
    
    global countsss
    countsss=0
    
    strat=[0,0]#num of exploration and num of exploitation
    
    ## Inizializzazione Reinforcement learning
    LEARNING_RATE= 0.01 #tra 0.1 e 0.001
    DISCOUNT= 0.95 #measure of how we value next q wrt to actual q, from 0 to 1, 0.9 to 0.99
    
    SHOW_EVERY = 9999
    EPISODES = episodes_IN
    
    #MAX=pop_size
    MAX=np.array([1,1,1,1]) #let's normalize it up
    MIN=np.array([0,0,0,0])
    
    YOU_WIN=1e-2
    YOU_LOOSE=9e-1
    
    Discretization_num_steps=50
    num_action=7
    num_states=4
    
    ## Inizializzazione Esplorazione
    epsilon = 0.9 #random action to explore stuff
    maxeps=1
    mineps=0.01
    epsilon_decay_value=1e-5

    STEPS=0
    
    loadit=1   
    
    class Q_Values():
        def __init__(self,Discretization_num_steps,num_actions,MAX,MIN):
            self.MAX=MAX
            self.MIN=MIN
            self.num_actions=num_actions
            self.Discretization_num_steps=Discretization_num_steps
            self.DISCRETE_OS_SIZE=[Discretization_num_steps]*num_states
            #print(self.discrete_os_win_size)
            #print(f"discrete_os_win_size= {discrete_os_win_size}")
            if loadit:
                self.q_table=np.load('Q_Table_SIIR_Variable_Step.npy')
                print("table loaded")
            else:
                self.q_table=np.random.uniform(low=-0.1,high=0, size=(self.DISCRETE_OS_SIZE + [num_action]))#50x50x50x7 ~= 42 MILION QVALUES with initial random values between -2 and 0
                print("random table")
            print(f"Q TABLE SHAPE is {self.q_table.shape}")
            
            self.max=np.array((0,0,0,0))
            self.min=np.array((50,50,50,50))
    
    
        def get_discrete_state(self,state): ##$step a larghezza variabile!
    
            #Vecchia verione stabile:
            #discrete_state = (state - self.MIN)/self.discrete_os_win_size 
    
            discrete_state=np.array((0,0,0,0))
            # bozza discrete_state a step variabile:
            for i in range(len(state)):
                if state[i]<=0.2: #sensibilita a 1% tra 0 e 20% di infetti
                    Discretization_num_steps=2*self.Discretization_num_steps
                    offset1=0
                    offset2 =0
                if state[i]>.2 and state[1]<=0.5: #sensibilita a 2% tra 20 e 50% di infetti
                    Discretization_num_steps=self.Discretization_num_steps
                    offset1=.2
                    offset2=20
                if state[i]>.5: #sensibilita a 3.3% tra 50 e 100% di infetti
                    
                    Discretization_num_steps=2*self.Discretization_num_steps/3
                    offset1=.5
                    offset2= 35
                # i 50 step sono distribuiti come 20 tra 0 e 20%,15 tra 20 e 50% e 15 tra 50 e 100%
    
                discrete_state[i]= (state[i]-offset1)*Discretization_num_steps+offset2
                if discrete_state[i]>self.max[i]:
                    self.max[i]=discrete_state[i]
    
                if discrete_state[i]<self.min[i]:
                    self.min[i]=discrete_state[i]
    
                
            #print(discrete_state)
                
            
            #print(f"state {state} discretized as {tuple(discrete_state.astype(np.int))}")
            return tuple(discrete_state.astype(np.int))
    
        def update_table(self, done, discrete_state, new_state, action, reward, episode):
            new_discrete_state=self.get_discrete_state(new_state)
            if not done:
                max_future_q= np.max(self.q_table[new_discrete_state])
                ##print(f"Infectous = {new_state[1]} still between {YOU_WIN} and {YOU_LOOSE}")
                current_q= self.q_table[discrete_state + (action, )]
                #print(f"Discrete state {discrete_state} and action {action} correspond to {current_q} Q value due to reward {reward}")
                new_q=(1-LEARNING_RATE)*current_q + LEARNING_RATE*(reward + DISCOUNT* max_future_q)#Bellman equation
                self.q_table[discrete_state + (action, )] = new_q
                #print(f"value loaded at {discrete_state + (action, )}")
            elif new_state[1]<=YOU_WIN:
                ##print(f"We made it on episode {episode} in {STEPS} STEPS with a reward of {reward} through {strat[0]} explorations and {strat[1]} exploitation")
                self.q_table[discrete_state + (action, )] = np.max(self.q_table)
            return new_discrete_state
    
    
    class Explore():
      def __init__(self,start,end,decay):
        self.start=start
        self.end=end
        self.decay=decay
    
      def get_exploration_rate(self,current_step):
        return self.end + (self.start - self.end)*math.exp(-1.*current_step*self.decay)
    
    
    class Environment():
        def __init__(self,beta_1,beta_2,b3,x_0):
    
    
            self.tot_infected=0
            self.t_vec=np.linspace(0,t_prev-1,t_prev)
    
            self.s_0, self.i1_0, self.i2_0, self.r2_0=x_0
    
            self.b_3=b3
            self.current_b3=self.b_3
            self.beta_1= beta_1
            self.beta_2= beta_2
            self.beta_1_current= beta_1
            self.beta_2_current= beta_2
            self.beta0=beta_1+beta_2
            self.done = False
    
            self.current_susceptible=np.ones((1,t_stop))*s_0
            self.current_infected=np.ones((1,t_stop))*i1_0
            self.current_undetected=np.ones((1,t_stop))*i2_0
            self.current_recovered=np.ones((1,t_stop))*r2_0
            
            self.susceptible=self.current_susceptible
            self.infected=self.current_infected
            self.undetected=self.current_undetected
            self.recovered=self.current_recovered
    
            self.history=np.array([self.current_susceptible,
                                self.current_infected,
                                self.current_undetected,
                                self.current_recovered])
    
            self.actual_state= np.array([self.current_susceptible[(-1,-1)],
                                self.current_infected[(-1,-1)],
                                self.current_undetected[(-1,-1)],
                                self.current_recovered[(-1,-1)]])
    
            self.prevision=i1_0
            self.new_prevision=i1_0
            
            #print(f"x0 is {x_0} while actual state is {self.actual_state}")
            self.count_action=np.array([0,0,0,0,0,0,0])
            self.k=1
            self.action_taken=[]
            self.rew_1=[]
            self.rew_2=[]
            self.rew_3=[]
            self.rew_4=[]
            self.reward=[]
    
        def reset(self):
    
            #strat[0]= 0
            #strat[1]= 0
    
            self.tot_infected=0
            self.reward.clear()
            self.rew_1.clear()
            self.rew_2.clear()
            self.rew_3.clear()
            self.rew_4.clear()
    
            self.done=False
    
            self.current_b3=self.b_3
            self.beta_1_current=self.beta_1
            self.beta_2_current=self.beta_2
            self.beta0=self.beta_1_current+self.beta_2_current
    
            i1_0 = random.uniform(5e-2, 2e-1)
            s_0 = 1.-i1_0
            i2_0 = 0
            r2_0 = 0
    
            initial_state=s_0,i1_0,i2_0,r2_0
    
            self.actual_state=initial_state
    
            aa=self.compute_next_state() # qua mi calcolo self.new_prevision
            #self.compute_reward(0)
            self.prevision=self.new_prevision
    
            initial_state=s_0,i1_0,i2_0,r2_0
            self.actual_state=initial_state
            self.previous_state=initial_state
    
            self.current_susceptible=np.ones((1,t_stop))*s_0
            self.current_infected=np.ones((1,t_stop))*i1_0
            self.current_undetected=np.ones((1,t_stop))*i2_0
            self.current_recovered=np.ones((1,t_stop))*r2_0
            
    
            self.susceptible=self.current_susceptible
            self.infected=self.current_infected
            self.undetected=self.current_undetected
            self.recovered=self.current_recovered
    
            self.history=np.array([self.current_susceptible,
                                self.current_infected,
                                self.current_undetected,
                                self.current_recovered])
    
            self.action_taken.clear()
            ##print(f"INITIALIZING.........initial state = {initial_state}, initial reward={self.reward}")
    
            
            return initial_state
    
        def take_action(self,action,STEP,explored):
            if action==0:
                self.beta_1_current=0
                ##print("chiudo scuole")
                self.action_taken.append("chiudo scuole")
                self.count_action[0]+=1
    
            if action==1:
                self.beta_1_current=self.beta_1
                ##print("apro scuole")
                self.action_taken.append("apro scuole")
                self.count_action[1]+=1
    
            if action==2:
                self.beta_2_current=0
                ##print("chiudo mezzi")
                self.action_taken.append("chiudo mezzi")
                self.count_action[2]+=1
    
            if action==3:
                self.beta_2_current=self.beta_2
                #print("apro mezzi")
                self.action_taken.append("apro mezzi")
                self.count_action[3]+=1
    
            if action==4:
                self.current_b3=self.b_3
                ##print("tampono")
                self.action_taken.append("tampono")
                self.count_action[4]+=1
    
            if action==5:
                self.current_b3=0
                ##print("non tampono")
                self.action_taken.append("non tampono")
                self.count_action[5]+=1
    
            if action==6:
                self.action_taken.append("niente")
                #print(f"eeeeeeiiii {self.action_taken}")
    
            self.count_action[6]+=1
            self.beta0=self.beta_1_current+self.beta_2_current   
    
            self.compute_next_state()
            reward=self.compute_reward(STEP,explored)
            self.done=self.check_done()
    
            return self.actual_state,reward,self.done
    
        def compute_next_state(self):
            #t_vec=np.linspace(0,len(self.history)-1,len(self.history))
    
            self.current_susceptible, self.current_infected, self.current_undetected, self.current_recovered =solve_path(self.t_vec,self.actual_state,self.beta0,self.current_b3)
            self.new_prevision=self.current_infected[-1]+self.current_undetected[-1]
    
    
    
            self.susceptible=np.concatenate((self.susceptible, self.current_susceptible),axis=None)
            self.infected=np.concatenate((self.infected, self.current_infected),axis=None)
            self.undetected=np.concatenate((self.undetected, self.current_undetected),axis=None)
            self.recovered=np.concatenate((self.recovered, self.current_recovered),axis=None)
            
            self.history=np.array([self.susceptible,
                                    self.infected,
                                    self.undetected,
                                    self.recovered])
            #print(self.history.shape)
            #print(self.history[1])
            #print(self.current_infected[t_stop])
            self.current_susceptible=self.current_susceptible[0:t_stop+1]
            self.current_infected=self.current_infected[0:t_stop+1]
            self.current_undetected=self.current_undetected[0:t_stop+1]
            self.current_recovered =self.current_recovered[0:t_stop+1]
    
            self.previous_state=self.actual_state
            self.actual_state= np.array([self.current_susceptible[-1],
                                self.current_infected[-1],
                                self.current_undetected[-1],
                                self.current_recovered[-1]])
    
            self.tot_infected=self.tot_infected+sum(self.beta0*self.current_susceptible*(self.current_infected+self.current_undetected))
            
            return self.actual_state
    
    
    
        def compute_reward(self,STEP,explore):
            #Da decidere come calcolare il reward, proposta: se la media tra nuovi infetti e vecchi infetti e' maggiore di tot Rew-1, minore di tor Rew+1, altrimenti Rew=0
                
            #Weigths
            w_infectivity=200
            w_socioeconomic=w_infectivity/20
            w_tempo=w_infectivity/200
            w_tot_infected=w_infectivity/50
    
    
            #Impact of action on infectous population
            DeltaI=self.prevision-(self.actual_state[1]+self.actual_state[2])
            self.prevision=self.new_prevision
    
            reward_infectivity=w_infectivity*DeltaI 
            self.rew_1.append(reward_infectivity)
    
            #socio_economic impact 
            reward_socioeconomic=w_socioeconomic*(self.beta0-0.2)
            self.rew_2.append(reward_socioeconomic)
    
            if explore:
                penalty_tempo=0
            else:
                penalty_tempo=-w_tempo* (0.04*math.exp(1.*STEP*0.5))
            #deaths=1-self.current_susceptible[-1]-self.current_undetected[-1]-self.current_infected[-1]-self.current_recovered[-1]
            #print(f"deaths are {deaths}")
            #penalty_death = -w_death*deaths
            self.rew_3.append(penalty_tempo)
    
            #penalty total infected
            if explore:
                w_tot_infected=w_infectivity/200
    
            penalty_infected=-w_tot_infected*self.tot_infected
            self.rew_4.append(penalty_infected)
    
            reward=reward_infectivity+reward_socioeconomic+penalty_tempo+penalty_infected
            self.reward.append(reward)
    
            #print(f"reward dato da {Rew} per previone +  {rew_actions} per l'azione scelta + {rew_morti} per il tempo trascorso nell'episodio")
            return reward
    
        def check_done(self):
            if self.actual_state[1]>= YOU_LOOSE or self.actual_state[1]<YOU_WIN:
                return True
            return False
    
    class Agent():
      def __init__(self,strategy,num_actions):
        self.current_step=0
        self.strategy=strategy #oggetto Explore
        self.num_actions=num_actions #dimensione spazio delle azioni
        self.strat=[0,0]
    
      def select_action(self,state,Q_Learn,beta,b_3):
        rate=strategy.get_exploration_rate(self.current_step)
        self.current_step+=1
        flag=0
        if rate>random.random():
          #print("EXPLORE")
    
          action= random.randrange(self.num_actions)
          self.strat[0]+=1
          flag=1
          while not action_available(beta,b_3,action):
            action= random.randrange(self.num_actions)
            #print(f"beta is {beta} and we chose action {action} getting {action_available(beta,b_3,action)}")
          return action,flag #explore
          
        else:
          #print("EXPLOIT ")
          action=np.argmax(Q_Learn.q_table[state]) #exploitation
          action_vec=Q_Learn.q_table[state]
          #print(f"from action_vec {action_vec} i take the action {action}")
          while not action_available(beta,b_3,action):
            action_vec[action]=np.min(action_vec)
            action=np.argmax(action_vec) #exploitation
        
    
            #print(f"but we can't so from action_vec {action_vec} i take the action {action}")
          
          self.strat[1]+=1
          return action, flag
            
    def F(x, t,beta0,b_3):
        """
        Time derivative of the state vector.
    
            * x is the state vector (array_like)
            * t is time (scalar)
            * R0 is the effective transmission rate, defaulting to a constant
    
        """
        s, i1, i2, r2 = x
    
    
    
        # Time derivatives, ci metto i pop_size???
        ds=-beta0*s*(i1+i2)+d1*r2
        di1=beta0*s*(i1+i2)-(b1+b2)*i1
        di2=b2*i1-c1*i2-b_3*i2
        dr2=(c1+c2*b1/(b2-b1))*i2-d1*r2
    
        return ds, di1, di2, dr2
    
    
    #se b_3=0, di2 cresce, di1 cresce, dr2 cresce, ds cala
    #Note that R0 can be either constant or a given function of time.
    
    
    def solve_path(t_vec, x_init,beta0,b_3):
        """
        Solve for i(t) and c(t) via numerical integration,
        given the time path for R0.
        
        """
        #print(f"x_init is {x_init}")
        G = lambda x, t: F(x, t, beta0,b_3)
        s_path, i1_path, i2_path, r2_path = odeint(G, x_init, t_vec).transpose()
            # per risultati cumulativi:
            # for i in range (len(s)):
            #     i1_tot += beta0*s(i)*(i1(i)+i2(i))
            #     guariti_tot +=(c1+c2*b1/b2)*i2(i)
        #r1_path = (1/(b2/b1-1))*i2_path     # cumulative cases
        #r3_path = c3*r1_path
        return s_path, i1_path, i2_path, r2_path
    
    def action_available(beta,b_3,action):
    
        if beta>0.16:
            if action==1 or action==3:#everything already open
                return False
        if beta<=0.16 and beta >0.04:
            if action==1 or action==2:#means already closed
    
                return False
        if beta<=0.04 and beta>0:
            if action==0 or action==3:#schools already closed
               
                return False
        if beta==0:
            if action==0 or action==2:#everything already closed
    
                return False
    
        if b_3==0.05:
            if action==4: #sto gia tamponando
                return False
    
        if b_3==0:
            if action==5: #non sto gia tamponando
                return False
    
        return True
    
    
    
    def plotit(Env,i_episode): ##$$$$ DA METTERE A POSTO
        #t_vec=np.linspace(0,len(s_path)-1,len(s_path)) DAAAAAAMNNNN c'era questooooooooooooo!!!
        name = '1_'+str(i_episode)+'.png'
        ptf = os.path.join(images_dir,name)
    
        fig, axs = plt.subplots(2,3)
    
        length=len(Env.history[1][t_stop:-1])
        Range=length/t_prev
        s_vec=Env.history[0][t_stop:-1]
        i1_vec=Env.history[1][t_stop:-1]
        i2_vec=Env.history[2][t_stop:-1]
        r_vec=Env.history[3][t_stop:-1]
    
    
        t_vec_plot=np.linspace(0,t_prev-1,t_prev).astype(int)
    
        s_path=[]
        i1_path=[]
        i2_path=[]
        r_path=[]
        r2_path=[]
        rew_path=[]
        t_vec_v=[]
        t_vec_v2=[]
    
        colors=['b--','g','k:','y']
        for i in range(int(Range)):
            
            t_vec_v.append((t_stop)*i + t_vec_plot)
            t_vec_v2.append((t_prev)*i + t_vec_plot)
            s_path.append(s_vec[t_vec_v2[i].astype(int)])
            i1_path.append(i1_vec[t_vec_v2[i].astype(int)])
            i2_path.append(i2_vec[t_vec_v2[i].astype(int)])
            r_path.append(r_vec[t_vec_v2[i].astype(int)])
            r2_path.append((1/(b2/b1-1))*i2_vec[t_vec_v2[i].astype(int)])
    
            axs[0,0].plot(t_vec_v[i],s_path[i],colors[i%4],linewidth=2, label=Env.action_taken[i])
            axs[0,0].set_title('susceptibles')
            axs[0,0].legend(Env.action_taken)
            axs[0,1].plot(t_vec_v[i],i1_path[i],colors[i%4],linewidth=2, label=Env.action_taken[i])
            axs[0,1].set_title('infectous')
            axs[0,1].legend(Env.action_taken)
            axs[1,0].plot(t_vec_v[i],i2_path[i],colors[i%4],linewidth=2, label=Env.action_taken[i])
            axs[1,0].set_title('undetected')
            axs[1,0].legend(Env.action_taken)
            axs[1,1].plot(t_vec_v[i],r_path[i],colors[i%4],linewidth=2, label=Env.action_taken[i])
            axs[1,1].set_title('recovered')
            axs[1,1].legend(Env.action_taken)
            axs[0,2].plot(t_vec_v[i],r2_path[i],colors[i%4],linewidth=2, label=Env.action_taken[i])
            axs[0,2].set_title('detected')
            axs[0,2].legend(Env.action_taken)
            
            #plt.plot(t_vec_v[i],p_vec[i],colors[i%4], )
            #plt.title("I1")
        idx=np.linspace(0,len(Env.reward)-1,len(Env.reward))
        axs[1,2].plot(idx,Env.reward,linewidth=2)
        axs[1,2].plot(idx,Env.rew_1)
        axs[1,2].plot(idx,Env.rew_2)
        axs[1,2].plot(idx,Env.rew_3)
        axs[1,2].plot(idx,Env.rew_4)
        axs[1,2].legend(['Totale','Prevision','SocioEconomic','Deaths','Tot Infected'])
        axs[1,2].set_title("reward")
        #print(Env.action_taken[1])
        F = plt.gcf() 
        Size = F.get_size_inches() 
        F.set_size_inches(Size[0]*2, Size[1]*2, forward=True)
        plt.savefig(ptf)
        #plt.show()
    
    
        #vecchia versione del plot
        # r_path= np.ones((1,len(s_path)))-s_path-i_path
        # plt.plot(t_vec,s_path,t_vec,i_path)
        # plt.title("S-I")
        # #plt.plot(t_vec,s_path,t_vec,i_path,t_vec,r_path)
        # plt.show()
    
        # idx=np.linspace(1,len(List),len(List))
        # plt.plot(idx,List)
        # plt.title("STEPS")
        # plt.show()
    
    
        # t_vec1=np.linspace(0,len(Env.history[1,:])-1,len(Env.history[1,:]))
        # fig, axs = plt.subplots(2,2)
        # # axs[0,0].plot(t_vec,s_path,label='susceptibles')
        # # axs[0,0].set_title('susceptibles')
        # # axs[0,1].plot(t_vec,i_path,label = 'infected')
        # # axs[0,1].set_title('infected')
        # count=Env.count_action
        # count_niente=count[6]-count[5]-count[4]-count[3]-count[2]-count[1]-count[0]
        # print(f"total Actions: {count[0]} chiudo scuole; {count[1]} apro scuole; {count[2]} chiudo mezzi; {count[3]} apro mezzi;{count[4]} tampono; {count[5]} non tampono; {count_niente}  niente")
        # axs[0,0].plot(idx,List)
        # axs[0,0].set_title('STEPS')
        # axs[0,1].hist(count.transpose(),5)
        # axs[0,1].set_title('actions')
    
        # axs[1,0].plot(t_vec1,Env.history[0,:])
        # axs[1,0].set_title('susceptibles')
        # axs[1,1].plot(t_vec1,Env.history[1,:])
        # axs[1,1].set_title('infected')
        # #axs[1,0].plot(t_vec,r_path, label='recovered')
        # #axs[1,0].set_title('recovered')
    
        # plt.show()
    
    
    
    
    ##MAIN
    beta_1=beta1_IN
    beta_2=beta2_IN
    b3=.05
    
    i1_0 = 1e-1
    s_0 = 1-i1_0
    i2_0 = 0
    r2_0 = 0
    
    x_0 = s_0, i1_0, i2_0, r2_0
    
    Env=Environment(beta_1,beta_2,b3,x_0)
    strategy=Explore(maxeps,mineps,epsilon_decay_value)
    Agent= Agent(strategy,num_action)
    Q_Learn=Q_Values(Discretization_num_steps,num_action,MAX,MIN)
    
    STEPS_List=[]
    for i_episode in range(EPISODES):
        STEPS = 0
        explored =0
        done=False
        if i_episode%SHOW_EVERY ==0:
            wonnaplot=True
        else:
            wonnaplot=False
        initial_state=Env.reset()
        state=Q_Learn.get_discrete_state(initial_state)
        # QUA dovrei genere uno stato iniziale random
        # obs = env.reset()
        # #state = tuple(obs.astype(np.int))
        # state = get_discrete_state(obs[2:])
        
        while not done:
            action,flag=Agent.select_action(state,Q_Learn,Env.beta0,Env.current_b3)
            if flag:
                explored=1
    
            #print(f"action is {action}")
            #count+=1
            STEPS+=1
            new_state,reward,done = Env.take_action(action,STEPS,explored)
 
            #print(f"New State is {new_state}")
            #new_state=tuple(new_obs.astype(np.int))
            state=Q_Learn.update_table(done, state, new_state, action, reward, i_episode)
        
        #print(Env.action_taken)
        if wonnaplot and i_episode:
            print("discrete states between")
            print(Q_Learn.min)
            print(Q_Learn.max)
            plotit(Env,i_episode)
            count=Env.count_action
            count_niente=count[6]-count[5]-count[4]-count[3]-count[2]-count[1]-count[0]
            print(f"total Actions: {count[0]} chiudo scuole; {count[1]} apro scuole; {count[2]} chiudo mezzi; {count[3]} apro mezzi;{count[4]} tampono; {count[5]} non tampono; {count_niente}  niente")
        
            print(f"{Agent.strat[0]} esplorazioni e {Agent.strat[1]} exploitations")
            #plotit(Env.current_susceptible,Env.current_infected,STEPS_List,Env)
        if not i_episode%5000:
            print(f"still going on episode {i_episode}")
        STEPS_List.append(STEPS)
    np.save('Q_Table_SIIR_Variable_Step', Q_Learn.q_table)
    print("saved")
    # labels = [f'susceptibles',f'infected',f'recovered']
    # s_paths, i_paths, r_paths = [], [], []
    
    # s_path, i_path, r_path = solve_path(t_vec)
    # s_paths.append(s_path)
    # i_paths.append(i_path)
    # r_paths.append(r_path)
    
    #plotit(s_path,i_path,r_path)
    
    
    
    
    #TO DO
    #   class environment = take action(prova a cambiare piu parametri per azione (not explicable)),REWARD FUNCTION!!!,
    #   class QValues = table initialization
    #   plot function = istogram
    
    ## cambiare valore iniziale ode
    ## cambiare beta ode
    ## done pochi infetti

#siirqt(from0,20000,0.16,0.04)