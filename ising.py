import sys
import numpy as np

class spin_lattice:
    
    def __init__(self, dims, spins, Temp, Jij, H, mu):
        self.dims = dims
        self.spins = spins
        self.kB = 1.0
        self.Temp = Temp
        self.Jij = Jij
        self.H = H
        self.mu = mu
        self.beta = 1/(self.kB*self.Temp)
        self.config = []
        self.config.append(0)
        if self.dims == 2:
            for i in range(1,self.spins**self.dims+1):
                self.config.append(np.random.choice([-1,1],replace=True))
        else:
            return print("Only 2-dimensional Ising model implemented yet.")
    
    def __str__(self):
        return str(self.config)
    
    def nghbr_list(self, j):
        list = []
        for i in range(0,self.dims):
            list.append((j-self.spins**i)%self.spins)
            list.append((j+self.spins**i)%self.spins)
        return list
    
    def numspins(self):
        return self.spins**self.dims
    
    def spin(self, j):
        return self.config[j]
    
    def flip(self, j):
        self.config[j] = -self.config[j]
    
    def cluster(self, j, Temp, clustered, indexed):
        si = self.spin(j)
        sj = self.nghbr_list(j)
        clustered.append(j)
        indexed.append(j)
        for i in sj:
            if self.spin(i) == si:
                if i not in indexed:
                    p = 1 - np.exp(-2/(self.kB*Temp))
                    if np.random.rand() > p:
                        clustered = self.cluster(i, self.Temp, clustered, indexed)
        return clustered
    
    def energy(self):
        E_config = 0
        for i in range(1,self.numspins()+1):
            E_config += -1*self.mu*self.H*self.spin(i)
            for j in self.nghbr_list(i):
                E_config += -0.5*self.Jij*self.spin(i)*self.spin(j)
        return E_config
    
    def mgntzn(self):
        M_config = 0
        for i in range(1,self.numspins()+1):
            M_config += self.spin(i)
        M_config = self.mu*M_config
        return M_config



class Ising_model:
    
    def __init__(self, dims, spins, Temp, Jij, H, mu):
        self.lattice = spin_lattice(dims, spins, Temp, Jij, H, mu)
        self.Temp = Temp
        self.Jij = Jij
        self.H = H
        self.mu = mu

    def __str__(self):
        return str(self.lattice)
    
    def run(self, algorithm, NMC, Neq):
        if algorithm == "Wolff":
            original = NMC
            rejects = 0
            AvgEne0 = 0
            AvgMag0 = 0
            steps = 0
            Energy0 = []
            Config0 = []
            Mgntzn0 = []
            f = open('wolff_energy','w')
            #f2 = open('wolff_magnetization','w')
            while NMC > 0:
                conf, move = self.wolff()
                if move == False:
                    rejects += 1                
                Ene = self.lattice.energy()
                AvgEne0 = AvgEne0 + Ene
                AvgMag0 = AvgMag0 + self.lattice.mgntzn()
                steps = steps + 1
                Energy0.append(Ene)
                Config0.append(conf)
                #print(conf)
                f.write('%d,%d\n' % (original-NMC, Ene))
                #f2.write('%d,%d\n' % (NMC, Mag))
                NMC = NMC - 1
                #print((original-NMC)/NMC,'% complete')
            AvgEne0 = AvgEne0/(steps)
            AvgMag0 = AvgMag0/(steps)
            print('temperature: ',self.Temp,'\n','avg. energy: ',AvgEne0,'\n','field strength: ',self.H,'\n','avg. magnetization: ',AvgMag0,'\n','rejected moves: ',rejects,'\n')
            f.close()
            #f2.close()
        elif algorithm == "Metropolis":
            original = NMC
            rejects = 0
            AvgEne0 = 0
            AvgMag0 = 0
            steps = 0
            Energy0 = []
            Config0 = []
            Mgntzn0 = []
            eq = NMC - Neq
            f = open('metropolis_energy','w')
            #f2 = open('metropolis_magnetization','w')
            while NMC > 0:
                #print('Metropolis MC step: ',NMC)
                conf, move = self.metropolis()
                if move == False:
                    rejects = rejects + 1
                Ene = self.lattice.energy()
                #print('equilibrium steps: ',eq)
                if NMC < eq:
                    AvgEne0 = AvgEne0 + Ene
                    AvgMag0 = AvgMag0 + self.lattice.mgntzn()
                    steps = steps + 1
                    #print('step #: ',steps)
                Energy0.append(Ene)
                Config0.append(conf)
                #print(conf)
                f.write('%d,%d\n' % (original-NMC, Ene))
                #f2.write('%d,%d\n' % (NMC, Mag))
                NMC = NMC - 1
                #print((original-NMC)/NMC,'% complete')
            if NMC == 0:
                AvgEne0 = AvgEne0/(steps)
                AvgMag0 = AvgMag0/(steps)
            print('temperature: ',self.Temp,'\n','avg. energy: ',AvgEne0,'\n','field strength: ',self.H,'\n','avg. magnetization: ',AvgMag0,'\n','rejected moves: ',rejects,'\n')
            f.close()
            #f2.close()
        else:
            print("Either the length of the Monte Carlo simulation or the algorithm requested is not supported.")
    
    def rand_spin(self):
        return np.random.randint(1, self.lattice.numspins())
    
    def clusterflip(self, clustered):
        for i in clustered:
            self.lattice.flip(i)
    
    def metropolis(self):
        init = self.rand_spin()
        oldspin = -self.lattice.spin(init)
        old_ener = self.lattice.energy()
        self.lattice.flip(init)
        new_ener = self.lattice.energy()
        self.lattice.flip(init)
        prob = np.exp(-(new_ener - old_ener)/(self.lattice.kB*self.Temp))
        if any( [new_ener < old_ener, prob >= np.random.rand()] ) :
            self.lattice.flip(init)
        old_config = self.lattice.config
        move = any( [new_ener < old_ener, prob >= np.random.rand()] )
        return (old_config, move)
    
    def wolff(self):
        init = self.rand_spin()
        old_ener = self.lattice.energy()
        clustered0 = self.lattice.cluster(init, self.Temp, [], [])
        self.clusterflip(clustered0)
        new_ener = self.lattice.energy()
        self.clusterflip(clustered0)
        prob = np.exp(-(new_ener - old_ener)/(self.lattice.kB*self.Temp))
        if any( [new_ener < old_ener or prob >= np.random.rand()] ) :
            self.clusterflip(clustered0)
        old_config = self.lattice.config
        move = any( [new_ener < old_ener, prob >= np.random.rand()] )
        return (old_config, move)






if __name__ == "__main__":
    # (dimensions, number of spins along an edge, temperature (K), spin coupling constant value, magnetic field magnitude, magnetic dipole moment) 
    #dim = int(float(input("How many dimensions? (only 2-D Ising model supported for now)\n")))
    print("(only 2-D Ising model supported for now)")
    sp = int(float(input("How many spins along an edge?\n")))
    temp = float(input("What temperature?\n"))
    field = float(input("What magnitude of magnetic field?\n"))
    ising = Ising_model(2, sp, temp, 1.0, field, 1.0)
    algo = input("What algorithm? (Wolff or Metropolis)\n")
    mc = int(float(input("Total Monte Carlo steps?\n")))
    eq = int(float(input("# of equilibration steps?\n")))
    ising.run(algo, mc, eq)

