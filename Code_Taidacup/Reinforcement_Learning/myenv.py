import gym
from gym import spaces
import numpy as np
import config
import time
from gen_filter_matrix import gen
import utils

primes = np.array([
    2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101, 
    103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197, 
    199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311, 
    313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431, 
    433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557, 
    563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661, 
    673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809, 
    811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937, 
    941,947,953,967,971,977,983,991,997,1009,1013,1019,1021,1031,1033,1039,1049, 
    1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153, 
    1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277, 
])

log_primes = np.log(primes)

class CircuitEnv(gym.Env):
    def __init__(self, node_num, connect_num, valid_res=[], time_res=[], vp=None):
        self.node_num = node_num
        self.connect_num = connect_num
        self.valid_res = valid_res
        self.time_res = time_res
        self.circuit_filter = CircuitFilter(node_num, connect_num, valid_res, time_res,vp)
        self.observation_space = spaces.MultiDiscrete([connect_num] * node_num + [connect_num + 1])
        self.action_space = spaces.Discrete(connect_num)
        self._current_state = None
        self._current_node = None
        self.info = {}
        self.reset()

    def reset(self,seed=None, **kwargs) -> np.array:
        super().reset(seed=seed,** kwargs)
        self._current_state = np.zeros((self.node_num + 1,))
        self._current_node = 1
        return self._current_state

    def step(self, action: np.array) -> np.array:
        if self._current_node > self.node_num:
            self.reset()

        self._current_state[self._current_node-1] = action
        self._current_state[-1] = self._current_node

        state = self._current_state
        reward = self.circuit_filter(state)
        done = False
        info = {}

        if self._current_node == self.node_num:
            done = True

        self._current_node += 1
        return state, reward, done, info

    def render(self, mode='human'):
        pass

class CircuitFilter(object):
    def __init__(self, node_num, connect_num, valid_res=[], time_res=[],vp=None):
        self.node_num = node_num
        self.connect_num = connect_num
        self.time_res = time_res
        self.valid_res = valid_res

        print("New Circuit Filter with res_array:",id(self.valid_res))

        self.paral_filter_m = np.loadtxt("paral_filter.txt", dtype=int)
        self.short_filter_m = np.loadtxt("short_filter.txt", dtype=int)

        self.V_val = np.loadtxt("V_val.txt", dtype=int)
        self.V_sw_val = np.loadtxt("V_sw_val.txt", dtype=int)

        self.V_val = self.V_val - 1
        self.V_sw_val = self.V_sw_val - 1

        self.V_node_avg = log_primes[self.V_val]
        self.V_node_sw = np.array([ log_primes[x] for x in self.V_sw_val])

 
        #self.vp = log_primes[:self.node_num//2][::-1]
        #self.vp = np.array([12,24,36,48])
        self.vp = vp 
        self.ip = np.ones_like(self.vp)
        self.ip[0] = -np.sum(self.vp[1:])/self.vp[0]


    def __call__(self, s: np.array):
        state = np.array(s)
        reward = 0
        if state[-1] == self.node_num:
            x = state[:-1].astype(np.int)
            #reward = self.score_current(x, self.ip)
            #reward = self.score_voltage(x, self.vp)
            #reward = self.score_hard(x)
            #reward = self.score_inductor_current(x, self.ip)
            reward = self.score_voltage_current(x, self.vp, self.ip)

            if (not config.paral) and reward > 0:
                res = (x+1).tolist()
                if not res in self.valid_res:
                    self.valid_res.append(res)
                    self.time_res.append(time.time())
                else:
                    #reward-=5
                    pass

        return reward

    def get_inductor_current_matrix(self, x : np.array):
        res = np.zeros((self.node_num//2 - 1, self.node_num//2)) # (inductor, iL_left by iport)
        for i in range(2, self.connect_num, 2):
            for j in range(1, self.node_num+1):
                if x[j-1]==i-1:
                    if j%2==0:
                        p = j//2
                        d = 1
                    else:
                        p = (j+1)//2
                        d = -1
                    res[i//2-1, p-1] = d
        
        res2 = np.zeros((self.node_num//2 - 1, self.node_num//2)) # (inductor, iL_right by iport)
        for i in range(3, self.connect_num, 2):
            for j in range(1, self.node_num+1):
                if x[j-1]==i-1:
                    if j%2==0:
                        p = j//2
                        d = 1
                    else:
                        p = (j+1)//2
                        d = -1
                    res2[(i-1)//2-1, p-1] = d
        
        return np.concatenate( (res, res2), axis = 0)
    
    def get_inductor_current(self, x, ip):
        return np.matmul( self.get_inductor_current_matrix(x)[ : self.connect_num//2-1 ,:] , ip )

    def get_DS_current_matrix(self,x ):
        res = np.zeros((self.node_num//2, self.node_num//2, self.node_num//2 - 1)) # (stage, switch, iDS by iL_left)
        #res2 = np.zeros((self.node_num//2, self.node_num//2, self.node_num//2 - 1)) # (stage, switch, iDS by iL_right)

        for stage in range(1, self.node_num//2+1):
            for switch in range(1, self.node_num//2+1):
                if stage != switch:
                    if switch<stage:
                        res[stage-1, switch-1, switch-1:stage-1] = 1
                    else:
                        res[stage-1, switch-1, stage-1:switch-1] = -1
        return np.concatenate( (res, res), axis = 2)

    def get_current_stress_matrix(self, x):
        Apl = self.get_inductor_current_matrix(x)
        Als = self.get_DS_current_matrix(x)
        Aps = np.matmul(Als,Apl)
        return Aps

    def get_current_stress(self, x, ip):
        """
            return matrix(switch, stage)
        """
        Aps = self.get_current_stress_matrix(x)
        i_s = np.matmul(Aps,ip)
        return i_s
    
    def get_current_stress_rms(self, x, ip, ds):
        i_s = self.get_current_stress(x, ip) # (switch, stage)
        ds = 1 - ds
        ds = ds.reshape((-1,1))
        rms = np.sqrt( np.matmul( (i_s**2) , ds ) )
        return rms

    def get_voltage_stress_matrix(self, x):
        ans = [0]*(self.node_num//2)
        visited = [0]*(self.node_num//2)
        end = self.connect_num-1

        cnt = [0]
        def dfs(node, state): # state == 1 for converter node; -1 for port node
            if cnt[0]>1000:
                return False
            cnt[0] += 1
            if state==1 and node==end:
                return True
            
            if state==1: # node is converter node
                for p_node, c_node in enumerate(x):
                    if c_node == node:
                        if dfs(p_node, -state):
                            return True
            else:
                t_node = node + 1
                if t_node%2!=0:
                    t_node +=1
                
                port = t_node // 2 - 1

                if visited[port]:
                    return False

                visited[port]=1
                
                t_node = node + 1
                if t_node%2!=0:
                    t_node +=1
                    ans[port]=1
                else:
                    t_node -=1
                    ans[port]=-1
                
                t_node = t_node - 1

                if dfs(x[t_node], -state):
                    return True

                ans[port]=0
                visited[port]=0

            return False
        
        if not dfs(0,1):
            ans = [0]*(self.node_num//2)
       
        return np.array(ans)
    
    def get_voltage_stress(self, x, vp):
        Aps = self.get_voltage_stress_matrix(x)
        return np.dot( Aps, vp )

    def score_hard(self, x):
        sc_hard = self.filter_each(x)
        if sc_hard == 0:
            sc_hard += 500
        return sc_hard

    def score_inductor_current(self, x, ip):
        sc_hard = self.filter_each(x)
        i_s = np.matmul( self.get_inductor_current_matrix(x)[:self.node_num-1,:], ip )
        sc_soft = -np.max( np.abs( i_s.flatten() ) )/np.sum(np.abs(ip))
        alpha = 0
        beta = 200
        e0 = 0
        e1 = 500

        if sc_hard<0:
            return sc_hard + alpha * sc_soft + e0
        else:
            return beta * sc_soft + e1

    def score_current(self, x, ip):
        sc_hard = self.filter_each(x)
        i_s = self.get_current_stress(x, ip)
        sc_soft = -np.max( np.abs( i_s.flatten() ) )/np.sum(np.abs(ip))
        alpha = 10
        beta = 200
        e0 = 0
        e1 = 500

        if sc_hard<0:
            return sc_hard + alpha * sc_soft + e0
        else:
            return beta * sc_soft + e1

    def score_voltage(self, x, vp):
        vs = self.get_voltage_stress(x, vp)

        alpha_basic = 100
        alpha_vs = 10
        alpha_duty_cycle = 10

        sc_basic = self.filter_each(x)
        sc_vs = -(vs<vp).sum()
        if sc_vs!=0:
            sc_duty_cycle = -1
        else:
            sc_duty_cycle = -1
            try:
                duty_cycle = utils.get_duty_cycle(x, vp)
                sc_duty_cycle = -np.maximum( np.tanh( np.abs(1/0.99*(duty_cycle - 0.5)) - 0.5) , 0).mean()
            except:
                pass
        sc_vs /= len(vp)
        
        sc_hard = alpha_basic * sc_basic + \
                  alpha_vs * sc_vs + \
                  alpha_duty_cycle * sc_duty_cycle
         
        
        sc_soft = -np.max( np.abs( vs.flatten() ) )/np.sum(np.abs(vp))
        #sc_soft = 0

        alpha = 10
        beta = 200
        e0 = 0
        e1 = 500

        if sc_hard<0:
            return sc_hard + alpha * sc_soft + e0
        else:
            return beta * sc_soft + e1

    def score_voltage_current(self, x, vp, ip):
        vs = self.get_voltage_stress(x, vp)
        i_s = np.sum(np.abs(ip))

        alpha_basic = 100
        alpha_vs = 10
        alpha_duty_cycle = 10

        sc_basic = self.filter_each(x)
        sc_vs = -(vs<vp).sum()
        if sc_vs!=0:
            sc_duty_cycle = -1
        else:
            sc_duty_cycle = -1
            try:
                duty_cycle = utils.get_duty_cycle(x, vp)
                if (duty_cycle<1).all() and (duty_cycle>0).all():
                    i_s = self.get_current_stress_rms(x, ip, duty_cycle)
                sc_duty_cycle = -np.maximum( np.tanh( np.abs(1/0.99*(duty_cycle - 0.5)) - 0.5) , 0).mean()
            except:
                pass
        sc_vs /= len(vp)
        
        sc_hard = alpha_basic * sc_basic + \
                  alpha_vs * sc_vs + \
                  alpha_duty_cycle * sc_duty_cycle
         
        
        if (vs==0).all():
            vs = np.sum(np.abs(vp))
            
        sc_soft = -1 * np.max( np.abs( vs.flatten() ) )/np.sum(np.abs(vp)) 
        sc_soft+= -0 * np.max( np.abs( i_s.flatten() ) )/np.sum(np.abs(ip))

        alpha = 0
        beta = 200
        e0 = 0
        e1 = 500

        if sc_hard<0:
            return sc_hard + alpha * sc_soft + e0
        else:
            return beta * sc_soft + e1
        
    def cnt_comp(self,x):
        vs = (x+1).reshape((-1,2))
        results = []
        for v in vs:
            res = 0
            if v[0]%2==0:
                res+=1
                v[0]+=1
            if v[1]%2==0:
                if v[1]==self.connect_num:
                    v[1]=self.connect_num+1
                else:
                    res+=1
                    v[1]+=1
            res+=(v[1]-v[0])/2
            results.append(res)
        return results

    def filter_each(self, x):
        self.V_avg = self.V_node_avg[ x[::2] ] - self.V_node_avg[ x[1::2] ]
        self.V_sw = self.V_node_sw[:,x[::2] ] - self.V_node_sw[:, x[1::2] ]

        V_avg_dir_check = ((x[::2]-x[1::2])>=0).sum() + ( (x[::2]%2==1) & (x[::2]-x[1::2]==-1) ).sum()

        V_avg_paral_check = np.matmul( self.V_avg, self.paral_filter_m.T)

        V_sw_paral_check = np.matmul( self.V_sw, self.paral_filter_m.T)
        V_sw_short_check = np.matmul( self.V_sw, self.short_filter_m.T)

        eps = 1e-5
        reward = -1/4* ( 
                V_avg_dir_check / (len(x)//2) +
                ( abs(V_avg_paral_check)<=eps ).sum() / len(self.paral_filter_m) +
                ( abs(V_sw_paral_check)<=eps ).sum() / len(self.paral_filter_m) +
                ( abs(V_sw_short_check)<=eps ).sum() / len(self.short_filter_m)
        )

        return reward


i_str = ["i_"+str(i)+" " for i in range(1,9)]
def show_i_stress(x):
    ans = []
    for i,d in enumerate(x):
        if d!=0:
            if d==1:
                ans.append("+ "+i_str[i])
            else:
                ans.append("- "+i_str[i])
    
    ans = "".join(ans)
    if ans[0]=="+":
        ans = ans[1:]
    return ans

v_str = ["v_"+str(i)+" " for i in range(1,9)]
def show_v_stress(x):
    ans = []
    for i,d in enumerate(x):
        if d!=0:
            if d==1:
                ans.append("+ "+v_str[i])
            else:
                ans.append("- "+v_str[i])
    ans = "".join(ans)
    if ans[0]=="+":
        ans = ans[1:]
    return ans

def create_table(x_val , x_m, indices, stress_type, constraint_type):
    ans = ""
    if stress_type=="current":
        for i,x in enumerate(indices):
            r,c = np.where(x_val[x]==np.max(x_val[x]))
            r,c = r[0], c[0]               
            ans += r"Fig.~\ref{%s%d} & %d & %d & $%s$ \\" % ("hc" if constraint_type=="hard" else "hsc" , i, c+1, r+1, show_i_stress( x_m[x][r][c]) ) + "\n"
            ans += r"\midrule" + "\n"

            print(x_val[x][r][c])
    else:
        for i,x in enumerate(indices):             
            ans += r"Fig.~\ref{%s%d} & $%s$ \\" % ("hv" if constraint_type=="hard" else "hsv" , i, show_v_stress( x_m[x] )) + "\n"
            ans += r"\midrule" + "\n"

            print(x_val[x])
    print(ans)

# def create_table(x_val , x_m, indices, constraint_type):
#     ans = ""
#     for i,x in enumerate(indices):
#         r,c = np.where(x_val[x]==np.max(x_val[x]))
#         r,c = r[0], c[0]               
#         ans += r"Fig.~\ref{%s%d} & %d & $%s$ \\" % ("hc" if constraint_type=="hard" else "hsc" , i, r+1, show_i_stress( x_m[x][r]) ) + "\n"
#         ans += r"\midrule" + "\n"

#         print(x_val[x][r][c])

#     print(ans)

if __name__ == "__main__":
    gen(8)
    ip = log_primes[:8][::-1].reshape((-1,1))
    #vp = log_primes[:8][::-1].reshape((-1,1))
    vp = np.array([12,8,3,6,15,4,8,8])

    # print("-------------------------soft constraints of inductor circuit stresses------------------------------")
    # x125 = np.loadtxt("exp/res_hard_soft/RL_port_8_hard.csv",dtype=int,delimiter=",")-1
    # xhs = np.loadtxt("exp/res_inductor/RL_port_8.csv",dtype=int,delimiter=",")-1


    # cf = CircuitFilter(16,16,[])
    # s125_m = [cf.get_inductor_current_matrix(c)[:7,:] for c in x125]
    # shs_m = [cf.get_inductor_current_matrix(c)[:7,:] for c in xhs]

    # x125_i = [cf.get_inductor_current(c,ip) for c in x125]
    # xhs_i = [cf.get_inductor_current(c,ip) for c in xhs]

    # s125_s = [(cf.score_inductor_current(c,ip),i) for i,c in enumerate(x125)]
    # shs_s = [(cf.score_inductor_current(c,ip),i) for i,c in enumerate(xhs)]

    # s125_s_sort = sorted(s125_s,reverse=True)
    # shs_s_sort = sorted(shs_s,reverse=True)
    
    # s125_s_sort_idx = np.array(s125_s_sort, dtype=int)[:,1] # 10
    # shs_s_sort_idx = np.array(shs_s_sort, dtype=int)[:,1] # 20


                    
    # t = 3
    # print("i1 to i8:", ip)
    # print("只有硬约束的情况：")
    # print("【电流压力】最小的前{}个电路：".format(t))
    # create_table(x125_i, s125_m, s125_s_sort_idx[0:t], "hard")
    # print(x125[s125_s_sort_idx[0:t]]+1)

    # print("\n")
    # print("包含硬约束和软约束的情况：")
    # print("【电流压力】最小的前{}个电路：".format(t))
    # create_table(xhs_i, shs_m, shs_s_sort_idx[0:t], "hard_soft")
    # print(xhs[shs_s_sort_idx[0:t]]+1)

    # print("-------------------------soft constraints of circuit stresses------------------------------")
    # x125 = np.loadtxt("exp/res_hard_soft/RL_port_8_hard.csv",dtype=int,delimiter=",")-1
    # xhs = np.loadtxt("exp/res_hard_soft/RL_port_8_current.csv",dtype=int,delimiter=",")-1


    # cf = CircuitFilter(16,16,[])
    # s125_m = [cf.get_current_stress_matrix(c) for c in x125]
    # shs_m = [cf.get_current_stress_matrix(c) for c in xhs]
    # s125_i = [np.abs( cf.get_current_stress(c, ip) ) for c in x125]
    # shs_i = [np.abs( cf.get_current_stress(c, ip) ) for c in xhs]
    # s125_s = [(cf.score_current(c,ip),i) for i,c in enumerate(x125)]
    # shs_s = [(cf.score_current(c,ip),i) for i,c in enumerate(xhs)]

    # s125_s_sort = sorted(s125_s,reverse=True)
    # shs_s_sort = sorted(shs_s,reverse=True)
    
    # s125_s_sort_idx = np.array(s125_s_sort, dtype=int)[:,1]
    # shs_s_sort_idx = np.array(shs_s_sort, dtype=int)[:,1]


                    
    # t = 3
    # print("i1 to i8:", ip)
    # print("只有硬约束的情况：")
    # print("【电流压力】最小的前{}个电路：".format(t))
    # create_table(s125_i, s125_m, s125_s_sort_idx[0:t], "current", "hard")
    # print(x125[s125_s_sort_idx[0:t]]+1)

    # print("\n")
    # print("包含硬约束和软约束的情况：")
    # print("【电流压力】最小的前{}个电路：".format(t))
    # create_table(shs_i, shs_m, shs_s_sort_idx[0:t], "current", "hard_soft")
    # print(xhs[shs_s_sort_idx[0:t]]+1)

    print("-------------------------soft constraints of voltage stresses------------------------------")
    x125 = np.loadtxt("exp/res_hard_soft/RL_port_8_hard.csv",dtype=int,delimiter=",")-1
    xhs = np.loadtxt("exp/res_v2/RL_port_8.csv",dtype=int,delimiter=",")-1
    

    cf = CircuitFilter(16,16,[])
    s125_m = [cf.get_voltage_stress_matrix(c) for c in x125]
    shs_m = [cf.get_voltage_stress_matrix(c) for c in xhs]
    s125_v = [np.abs( cf.get_voltage_stress(c,vp) ) for c in x125]
    shs_v = [np.abs( cf.get_voltage_stress(c,vp) ) for c in xhs]
    s125_s = [(cf.score_voltage(c,vp),i) for i,c in enumerate(x125)]
    shs_s = [(cf.score_voltage(c,vp),i) for i,c in enumerate(xhs)]

    s125_s_sort = sorted(s125_s,reverse=True)
    shs_s_sort = sorted(shs_s,reverse=True)
    
    s125_s_sort_idx = np.array(s125_s_sort, dtype=int)[:,1]
    shs_s_sort_idx = np.array(shs_s_sort, dtype=int)[:,1]


                    
    t = 10
    print("v1 to v8:", vp)
    print("只有硬约束的情况：")
    print("【电压压力】最小的前{}个电路：".format(t))
    create_table(s125_v, s125_m, s125_s_sort_idx[0:t], "voltage", "hard")
    print(x125[s125_s_sort_idx[0:t]]+1)


    print("\n")
    print("包含硬约束和软约束的情况：")
    print("【电压压力】最小的前{}个电路：".format(t))
    create_table(shs_v, shs_m, shs_s_sort_idx[0:t], "voltage", "hard_soft")
    print(xhs[shs_s_sort_idx[0:t]]+1)



    
