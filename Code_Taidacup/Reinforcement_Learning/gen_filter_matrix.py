import numpy as np
def gen(n):
    res = []
    res2 = []
    t = []
    t2 = []

    def gen(ones, negs, zeros, place):
        if place == n:
            if (not t in res) and (not [-x for x in t] in res):
                res.append(t.copy())
            return
        
        if ones:
            t[place] = 1
            gen(ones-1, negs, zeros, place+1)
        if negs:
            t[place] = -1
            gen(ones, negs-1, zeros, place+1)
        if zeros:
            t[place] = 0
            gen(ones, negs, zeros-1, place+1)

    def gen2(place):
        if place == n:
            if not t2 in res2:
                res2.append(t2.copy())
            return
        
        t2[place]=0
        gen2(place+1)
        t2[place]=1
        gen2(place+1)

    for zeros in range(0, n-2 + 1):
        for negs in range(1, n-zeros-1 + 1):
            for ones in range(1, n-zeros-negs + 1):
                t = [0]*n
                t2 = [0]*n
                gen(ones, negs, zeros, 0)
                gen2(0)

    V_val = []
    for i in range(n,1,-1):
        V_val.append(i)
        V_val.append(i)
    V_val = [n+1] + V_val + [1]

    V_sw_val_1 = list(range(3,3+n-1))
    V_sw_val_1.append(2)

    V_sw_val_2 = []
    for i in range(1,n+1):
        t_sw = [1]*i + [2]*(n-i)
        V_sw_val_2.append(t_sw)

    V_sw_val = []
    for x in V_sw_val_2:
        t_sw = np.array(list(zip(x,V_sw_val_1)))
        V_sw_val.append(t_sw.flatten().tolist())

    V_val = np.array(V_val).reshape((1,-1))
    V_sw_val = np.array(V_sw_val)

    np.savetxt("paral_filter.txt", np.array(res), fmt="%d")
    np.savetxt("short_filter.txt", np.array(res2[1:]), fmt="%d")
    np.savetxt("V_val.txt", V_val, fmt="%d")
    np.savetxt("V_sw_val.txt", V_sw_val, fmt="%d")
    print("param for port %d generated." % n)

if __name__=="__main__":
    gen(8)
