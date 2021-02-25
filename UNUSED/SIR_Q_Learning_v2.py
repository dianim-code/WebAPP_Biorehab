def sirqt():
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
    images_dir = os.path.join(my_path,'static\\images\\SIRQT')

    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
        os.mkdir(images_dir)
    else:
        os.mkdir(images_dir)
    
    ## Mi aspetto che il sistema non impari bene perche' non sa cosa e' aperto e cosa no, quindi aggiungerei lo stato beta discretizzato in 4 valori
    
    
    ## Inizializazione SIR
    pop_size = 1e3
    
    t_fin=550 #days
    t_stop=10 #batch size
    t_prev=2*t_stop
    t_vec=np.linspace(0,t_prev-2,t_prev-1)
    #print(t_vec)
    b_1=0.16
    b_2=0.04
    beta0=b_1+b_2   #probabilita di contagio da incontro a rischio
    q=0.05          #rateo di crescita guarigioi
    global countsss
    countsss=0
    
    # initial conditions of s, e, i
    i_0 = 1e-1
    s_0 = 1-i_0
    r_0 = 0
    
    x_0 = s_0, i_0
    
    
    strat=[0,0]#num of exploration and num of exploitation
    
    ## Inizializzazione Reinforcement learning
    LEARNING_RATE= 0.05 #tra 0.1 e 0.001
    DISCOUNT= 0.95 #measure of how we value next q wrt to actual q, from 0 to 1, 0.9 to 0.99
    
    SHOW_EVERY = 9999
    EPISODES = 50000
    
    #MAX=pop_size
    MAX=np.array([1,1]) #let's normalize it up
    MIN=np.array([0,0])
    
    YOU_WIN=1e-2
    YOU_LOOSE=9e-1
    
    Discretization_num_steps=100
    num_action=5
    num_states=2
    
    ## Inizializzazione Esplorazione
    epsilon = 0.9 #random action to explore stuff
    maxeps=1
    mineps=0.005
    epsilon_decay_value=0.0005
    
    
    STEPS=0
    
    loadit=1    
    
    class Q_Values():
        def __init__(self,Discretization_num_steps,num_actions,MAX,MIN):
            self.MAX=MAX
            self.MIN=MIN
            self.num_actions=num_actions
            self.Discretization_num_steps=Discretization_num_steps
            self.DISCRETE_OS_SIZE=[Discretization_num_steps]*num_states
            self.discrete_os_win_size = (MAX-MIN)/self.DISCRETE_OS_SIZE
            #print(self.discrete_os_win_size)
            #print(f"discrete_os_win_size= {discrete_os_win_size}")
            if loadit:
                self.q_table=np.load('Q_Table_SI.npy')
                print("table loaded")
            else:
                self.q_table=np.random.uniform(low=-0.1,high=0, size=(self.DISCRETE_OS_SIZE + [num_action]))#100x100x5 with initial random Q values between -2 and 0
                print("random table")
            print(f"Q TABLE SHAPE is {self.q_table.shape}")
        
        def get_discrete_state(self,state):
            discrete_state = (state - self.MIN)/self.discrete_os_win_size 
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
        def __init__(self,b_1,b_2,q,x_0):
            t_stop=10 #batch size
            t_prev=2*t_stop
            self.t_vec=np.linspace(0,t_prev-2,t_prev-1)
    
            self.s_0, self.i_0=x_0
            self.b_1= b_1
            self.b_2= b_2
            self.b_1_current= b_1
            self.b_2_current= b_2
            self.beta0=b_1+b_2
            self.q=q
            self.done = False
            self.reward=0
            self.reward_List=[]
    
            self.current_susceptible=np.ones((1,t_stop))*s_0
            self.current_infected=np.ones((1,t_stop))*i_0
            
            self.susceptible=self.current_susceptible
            self.infected=self.current_infected
    
            self.history=np.array([self.current_susceptible,
                                self.current_infected])
    
            self.actual_state= np.array([self.current_susceptible[(-1,-1)],
                                self.current_infected[(-1,-1)]])
            #print(f"x0 is {x_0} while actual state is {self.actual_state}")
            self.count_action=np.array([0,0,0,0,0])
            self.k=1
            self.action_taken=[]
    
    
        def reset(self):
    
            strat[0]= 0
            strat[1]=0
    
            self.done=False
            self.reward_List.append(self.reward)
            self.reward=0
            self.b_1_current=self.b_1
            self.b_2_current=self.b_2
            self.beta0=self.b_1_current+self.b_2_current
            i_0 = random.uniform(5e-2, 1e-1)
            s_0 = 1.-i_0
    
            initial_state=s_0,i_0
            self.actual_state=initial_state
    
            aa=self.compute_next_state() # qua mi calcolo self.new_prevision
            self.prevision=self.new_prevision
    
            initial_state=s_0,i_0
            self.actual_state=initial_state
            self.previous_state=initial_state
    
            self.current_susceptible=np.ones((1,t_stop))*s_0
            self.current_infected=np.ones((1,t_stop))*i_0
    
            self.susceptible=self.current_susceptible
            self.infected=self.current_infected
    
            self.history=np.array([self.current_susceptible,
                                self.current_infected])
            ##print(f"INITIALIZING.........initial state = {initial_state}, initial reward={self.reward}")
            self.action_taken.clear()
            return initial_state
    
        def take_action(self,action):
            if action==0:
                self.b_1_current=0
                ##print("chiudo scuole")
                self.action_taken.append("chiudo scuole")
                self.count_action[0]+=1
    
            if action==1:
                self.b_1_current=self.b_1
                ##print("apro scuole")
                self.action_taken.append("apro scuole")
                self.count_action[1]+=1
    
            if action==2:
                self.b_1_current=0
                ##print("chiudo mezzi")
                self.action_taken.append("chiudo mezzi")
                self.count_action[2]+=1
    
            if action==3:
                self.b_1_current=self.b_1
                ##print("apro mezzi")
                self.action_taken.append("apro mezzi")
                self.count_action[3]+=1
    
            self.count_action[4]+=1
            self.beta0=self.b_1_current+self.b_2_current
            beta0=self.beta0     
            self.compute_next_state()
            self.reward=self.compute_reward()
            self.done=self.check_done()
    
            return self.actual_state,self.reward,self.done
    
        def compute_next_state(self):
            #t_vec=np.linspace(0,len(self.history)-1,len(self.history))
            self.current_susceptible, self.current_infected =solve_path(self.t_vec,self.actual_state,self.beta0)
            
            ##$print(f"current_infected {self.current_infected} perche' ho beta = {self.beta0} e ho passato {self.actual_state}")
            # if self.k==0:
            #     plt.plot(self.t_vec[0:t_stop],self.prev_infected,'o',label='prevision')
            #     plt.plot(self.t_vec[0:t_stop],self.current_infected[0:t_stop],label='actual')
            #     plt.legend()
            #     #plt.plot(t_vec[0:t_stop],self.prev_infected,t_vec[0:t_stop-1],self.current_infected[0:t_stop-1])
            #     plt.title("S-I")
            #     #plt.plot(t_vec,s_path,t_vec,i_path,t_vec,r_path)
            #     plt.show()
            # self.k=0
    
            self.prev_infected=self.current_infected[t_stop-1:t_prev]
            #print(f"abbiamo un current_infected {self.current_infected} e ne prende {self.current_infected[0:t_stop]}")
    
            self.new_prevision=self.current_infected[-1]
    
            self.current_susceptible=self.current_susceptible[0:t_stop]
            self.current_infected=self.current_infected[0:t_stop]
            
    
            self.susceptible=np.concatenate((self.susceptible, self.current_susceptible),axis=None)
            self.infected=np.concatenate((self.infected, self.current_infected),axis=None)
    
            self.history=np.array([self.susceptible,
                                    self.infected])
    
    
            #plotit(self.history[0,:],self.history[1,:],self.history[2,:])
            
            self.previous_state=self.actual_state
            self.actual_state= np.array([self.current_susceptible[-1],
                                self.current_infected[-1]])
            return self.actual_state
    
        def compute_reward(self):
            #Da decidere come calcolare il reward, proposta: se la media tra nuovi infetti e vecchi infetti e' maggiore di tot Rew-1, minore di tor Rew+1, altrimenti Rew=0
            DeltaI=self.actual_state[1]-self.prevision
            Rew=-100*DeltaI 
    
            rew_actions=20*(self.beta0-0.2)
    
            self.prevision=self.new_prevision
    
            return Rew+rew_actions
    
        def check_done(self):
            if self.actual_state[1]>= YOU_LOOSE or self.actual_state[1]<YOU_WIN:
                return True
            return False
    
    class Agent():
      def __init__(self,strategy,num_actions):
        self.current_step=0
        self.strategy=strategy #oggetto Explore
        self.num_actions=num_actions #dimensione spazio delle azioni
    
    
      def select_action(self,state,Q_Learn,beta):
        rate=strategy.get_exploration_rate(self.current_step)
        self.current_step+=1
    
        if rate>random.random():
          #print("EXPLORE")
    
          action= random.randrange(self.num_actions)
          strat[0]+=1
          while not action_available(beta,action):
            action= random.randrange(self.num_actions)
          return action #explore
          
        else:
          #print("EXPLOIT ")
          action=np.argmax(Q_Learn.q_table[state]) #exploitation
          action_vec=Q_Learn.q_table[state]
          #print(f"from action_vec {action_vec} i take the action {action}")
          while not action_available(beta,action):
            action_vec[action]=np.min(action_vec)
            action=np.argmax(action_vec) #exploitation
            #print(f"but we can't so from action_vec {action_vec} i take the action {action}")
          
          strat[1]+=1
          return action 
            
    def F(x, t,beta0):
        """
        Time derivative of the state vector.
    
            * x is the state vector (array_like)
            * t is time (scalar)
            * R0 is the effective transmission rate, defaulting to a constant
    
        """
        s, i = x
    
        
    
        # Time derivatives, ci metto i pop_size???
        global countsss
    
        countsss+=1
        ds = -beta0 *s*i
    
        
    
        di = beta0 *s*i - q * i
        #if countsss%20:
            #print(f"dentro beta e' {beta0} e di e' {beta0 *s*i } + {- q * i}")
            #print(x)
        return ds, di
    
    #Note that R0 can be either constant or a given function of time.
    
    
    def solve_path(t_vec, x_init,beta0):
        """
        Solve for i(t) and c(t) via numerical integration,
        given the time path for R0.
        
        """
        #print(f"x_init is {x_init}")
        G = lambda x, t: F(x, t,beta0)
        s_path, i_path= odeint(G, x_init, t_vec).transpose()
    
        #c_path = 1 - s_path - e_path       # cumulative cases
        return s_path, i_path
    
    def action_available(beta,action):
        if beta==0.2:
            if action==1 or action==3:
                return False
        if beta==0.16:
            if action==1 or action==2:
                return False
        if beta==0.04:
            if action==0 or action==3:
                return False
        if beta==0:
            if action==0 or action==2:
                return False
    
        return True
    
    
    
    def plotit(s_path,i_path,List,Env):
        #t_vec=np.linspace(0,len(s_path)-1,len(s_path)) DAAAAAAMNNNN c'era questooooooooooooo!!!
        r_path= np.ones((1,len(s_path)))-s_path-i_path
        plt.plot(t_vec,s_path,t_vec,i_path)
        plt.title("S-I")
        #plt.plot(t_vec,s_path,t_vec,i_path,t_vec,r_path)
        plt.show()
    
        idx=np.linspace(1,len(List),len(List))
        plt.plot(idx,List)
        plt.title("STEPS")
        plt.show()
    
    
        t_vec1=np.linspace(0,len(Env.history[1,:])-1,len(Env.history[1,:]))
        fig, axs = plt.subplots(2,2)
        # axs[0,0].plot(t_vec,s_path,label='susceptibles')
        # axs[0,0].set_title('susceptibles')
        # axs[0,1].plot(t_vec,i_path,label = 'infected')
        # axs[0,1].set_title('infected')
        count=Env.count_action
        count_niente=count[4]-count[3]-count[2]-count[1]-count[0]
        print(f"total Actions: {count[0]} chiudo scuole; {count[1]} apro scuole; {count[2]} chiudo mezzi; {count[3]} apro mezzi; {count_niente}  niente")
        axs[0,0].plot(idx,List)
        axs[0,0].set_title('STEPS')
        axs[0,1].hist(count.transpose(),5)
        axs[0,1].set_title('actions')
    
        axs[1,0].plot(t_vec1,Env.history[0,:])
        axs[1,0].set_title('susceptibles')
        axs[1,1].plot(t_vec1,Env.history[1,:])
        axs[1,1].set_title('infected')
        #axs[1,0].plot(t_vec,r_path, label='recovered')
        #axs[1,0].set_title('recovered')
    
        plt.show()
    
    def plotit3(Env,i_episode):
        name = '1_'+str(i_episode)+'.png'
        ptf = os.path.join(images_dir,name)

        #t_vec=np.linspace(0,len(s_path)-1,len(s_path)) DAAAAAAMNNNN c'era questooooooooooooo!!!
        fig, axs = plt.subplots(1,3)

        length=len(Env.history[1][t_stop:-1])
        Range=length/t_prev
        s_vec=Env.history[0][t_stop:-1]
        i_vec=Env.history[1][t_stop:-1]
        r_vec=np.ones(np.size(s_vec)) - (s_vec + i_vec)
    
        t_vec_plot=np.linspace(0,t_prev-1,t_prev).astype(int)
    
        s_path=[]
        i_path=[]
        r_path=[]
        new_path=[]
        t_vec_v=[]
        t_vec_v2=[]
    
        colors=['b--','b','k--','k','y']
    
        lblplt = Env.action_taken
        #lblplt = list(dict.fromkeys(Env.action_taken))
    
        for i in range(int(Range)):
            
            if Env.action_taken[i] == "chiudo scuole":
                styplt = colors[0]
                actplt = "chiudo scuole"
            else:
                if Env.action_taken[i] == "apro scuole":
                    styplt = colors[1]
                    actplt = "apro scuole"
                else:
                    if Env.action_taken[i] == "chiudo mezzi":
                        styplt = colors[2]
                        actplt = "chiudo mezzi"
                    else:
                        if Env.action_taken[i] == "apro mezzi":
                            styplt = colors[3]
                            actplt = "apro mezzi"
                        else:
                            styplt = colors[4]
                            actplt = "niente"
            
            #print(actplt)
    
            t_vec_v.append((t_stop)*i + t_vec_plot)
            t_vec_v2.append((t_prev)*i + t_vec_plot)
            s_path.append(s_vec[t_vec_v2[i].astype(int)])
            i_path.append(i_vec[t_vec_v2[i].astype(int)])
            r_path.append(r_vec[t_vec_v2[i].astype(int)])
    
            axs[0].plot(t_vec_v[i],s_path[i],styplt,linewidth=2, label=actplt)
            axs[0].set_title('susceptibles')
            axs[0].legend(lblplt)
            axs[1].plot(t_vec_v[i],i_path[i],styplt,linewidth=2, label=actplt)
            axs[1].set_title('infectous')
            axs[1].legend(lblplt)
            axs[2].plot(t_vec_v[i],r_path[i],styplt,linewidth=2, label=actplt)
            axs[2].set_title('recovered')
            axs[2].legend(lblplt)

            #plt.plot(t_vec_v[i],p_vec[i],colors[i%4], )
            #plt.title("I1")
        
        #print(Env.action_taken[1])
        F = plt.gcf() 
        Size = F.get_size_inches() 
        F.set_size_inches(Size[0]*2, Size[1]*2, forward=True)
        plt.savefig(ptf)
        plt.show()
    

    ##MAIN
    Env=Environment(b_1,b_2,q,x_0)
    strategy=Explore(maxeps,mineps,epsilon_decay_value)
    Agent= Agent(strategy,num_action)
    Q_Learn=Q_Values(Discretization_num_steps,num_action,MAX,MIN)

    STEPS_List=[]
    for i_episode in range(EPISODES):
        STEPS =0
    
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
            action=Agent.select_action(state,Q_Learn,Env.beta0)
            #print(f"action is {action}")
            #count+=1
            STEPS+=1
            new_state,reward,done = Env.take_action(action)
            
    
            
            #print(f"New State is {new_state}")
            #new_state=tuple(new_obs.astype(np.int))
            state=Q_Learn.update_table(done, state, new_state, action, reward, i_episode)
        
        if wonnaplot and i_episode:
            #env.render()
            #plotit(Env.history[0,:],Env.history[1,:],Env.history[2,:])
            #plt.plot(t_vec,Env.current_infected)
            plotit3(Env,i_episode)
            #plt.show()
            count=Env.count_action
            count_niente=count[4]-count[3]-count[2]-count[1]-count[0]
            print(f"total Actions: {count[0]} chiudo scuole; {count[1]} apro scuole; {count[2]} chiudo mezzi; {count[3]} apro mezzi; {count_niente}  niente")
            #plotit(Env.current_susceptible,Env.current_infected,STEPS_List,Env)
        if not i_episode%50000:
            print("still going")
        STEPS_List.append(STEPS)
    np.save('Q_Table_SI', Q_Learn.q_table)
    print("saved")
    # labels = [f'susceptibles',f'infected',f'recovered']
    # s_paths, i_paths, r_paths = [], [], []
    
    # s_path, i_path, r_path = solve_path(t_vec)
    # s_paths.append(s_path)
    # i_paths.append(i_path)
    # r_paths.append(r_path)
    
    #plotit(s_path,i_path,r_path)
    
    
    
    
    #TO DO
    #   class environment = take action(prova a cambiare piu parametri per azione (not explicable)),REWARD FUNCTION!!!, Generate random initial state
    #   class QValues = table initialization, save table, load table.
    #   plot function = istogram
    #   (class DQNN,class Memory, advanced class QValues )
    
    ## cambiare valore iniziale ode
    ## cambiare beta ode
    ## done pochi infetti

sirqt()
