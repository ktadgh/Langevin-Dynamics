import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit
from numba.types import pyobject
import numba
import time

class Scheme:
    def __init__(self, step_funtion, force, potential):
        self.step_function = step_funtion
        self.force = force
        self.potential = potential
        self.run_numba_convergence = make_numba_convergence(self.step_function)
        self.run_numba_average_convergence = make_numba_average_convergence(self.step_function)
        self.run_simulation = make_simulation(self.step_function)
        self.simulate_trajectories = make_simulate_trajectories(self.step_function, self.run_simulation)

    def convergence_graph(self, n_steps, stepsize, sims,gammas, q_init = None, p_init = None):
        if q_init is None:
            q_init = np.zeros(sims)
        if p_init is None:
            p_init = np.zeros(sims)
        gammas, trajectories = self.simulate_trajectories(n_steps, stepsize, gammas, q_init, p_init)
        for i in range(len(trajectories)):
            q_traj= trajectories[i]
            #print(np.shape(q_traj))
            q_hists = [np.histogram(q,bins=50,range=[-3,3], density=True)[0] for q in q_traj]
            q_diff = np.diff(q_hists)
            q_diffs = np.array([np.linalg.norm(diff) for diff in q_diff])
            plt.plot(q_diffs, alpha = 0.3, label=f'Friction = {gammas[i]}')
            plt.xlabel('$q$')
            plt.ylabel('2-norm of difference')
            #plt.yscale('log')

        plt.title("Convergence of $q$ for different values of friction")
        plt.legend()

    def friction_graph(self, n_steps, stepsize, sims,gammas, q_init = None, p_init = None):
        if q_init is None:
            q_init = np.zeros(sims)
        if p_init is None:
            p_init = np.zeros(sims)
        gammas, trajectories = self.simulate_trajectories(n_steps, stepsize, gammas, q_init, p_init)
        fig = plt.figure(figsize=[10,7])

        for i in range(len(trajectories)):
            q_traj= trajectories[i]
            histogram,bins = np.histogram(q_traj,bins=50,range=[-3,3], density=True)
            midx = (bins[0:-1]+bins[1:])/2
            plt.plot(midx,histogram,label=f'Friction = {gammas[i]}')
            plt.xlabel('$q$')
            plt.ylabel('Density')



        plt.title("Distribution of $q$ for different values of friction")
        rho = np.exp( -self.potential(midx))
        rho = rho / ( np.sum(rho) * (midx[1]-midx[0]) )
        plt.plot(midx,rho,'--',label='Truth')
        plt.legend()

    def cum_mean(traj_mean, cut_out):
        traj_2 = traj_mean[cut_out:]
        return np.cumsum(traj_2)/ np.arange(1, len(traj_2)+1)

    def expectation_graph(self, function, n_steps, stepsize, sims,gammas, q_init, p_init):
        if q_init is None:
            q_init = np.zeros(sims)
        if p_init is None:
            p_init = np.zeros(sims)


        fig = plt.figure(figsize=[10,7])
        gammas, trajectories = self.simulate_trajectories(n_steps, stepsize, gammas, q_init, p_init)

        for i in range(len(trajectories)):
            q_traj = np.array(trajectories[i])
            q_traj = function(q_traj)
            gamma = gammas[i]
            expectation = Scheme.cum_mean(np.mean(q_traj, axis = 1), 5000)
            plt.plot(expectation, label = f"Gamma = {gamma}")
        plt.ylabel('$Expectation$')
        plt.xlabel('Number of Iterations')


        plt.title("Expectation Graph")
        plt.legend()
        #plt.show()

    def convergence_time_graph(self,error, function,value, h, gammas,q_init,p_init):
        its1 = []
        for gamma in gammas:
            print(f"Gamma = {gamma}")
            iters = self.run_numba_convergence(error, function,value, h, gamma,q_init,p_init)
            print(f"Iters = {iters}")
            its1.append(iters)
        plt.plot(gammas, its1)
        plt.xlabel('gamma')
        plt.ylabel('iterations to convergence')
        plt.xscale('log')

    def avg_convergence_time_graph(self,title,error, function,value, h, gammas,q_init,p_init):
        its1 = []
        for gamma in gammas:
            print(f"Gamma = {gamma}")
            iters = self.run_numba_average_convergence(error, function,value, h, gamma,q_init,p_init)
            print(f"Iters = {iters}")
            its1.append(iters)
        iters1 = np.array(its1)
        np.save('iterations',iters1)
        plt.plot(gammas, its1)
        plt.title(title)
        plt.savefig(title+ ".pdf", bbox_inches = 'tight')
        plt.xlabel('gamma')
        plt.ylabel('iterations to convergence')
        plt.xscale('log')


def make_obabo(force):
    @njit(parallel=True)
    def A_step( qp , h , factor=1):
        q,p = qp

        q = q + h*p*factor

        return [q,p]

    @njit(parallel=True)
    def B_step( qp , h, factor=1):
        q,p = qp

        F = force(q)

        p = p + h*F*factor

        return [q,p]


    @njit()
    def O_step( qp , h,gamma, factor=1):
        q,p = qp

        alpha = np.exp(-h*gamma)

        R = np.random.randn( q.size ).reshape( q.shape)
        p = np.exp(- gamma*h *factor)*p+ np.sqrt(1-np.exp(-gamma*h))*R

        return [q,p]

    @njit()
    def obabo_step(q,p,h,gamma):
        qp = [q,p]  #this just translates the separate q and p vectors
                    #into a single vector composed from the pair.
        qp = O_step( qp, h, gamma, 0.5)
        qp = B_step( qp, h,0.5)
        qp = A_step(qp , h )
        qp = B_step( qp, h,0.5)
        qp = O_step( qp, h, gamma,  0.5)

        q,p = qp
        return q , p
    return obabo_step

def make_baoab(force):
    @njit(parallel=True)
    def A_step( qp , h , factor=1):
        q,p = qp

        q = q + h*p*factor

        return [q,p]

    @njit(parallel=True)
    def B_step( qp , h, factor=1):
        q,p = qp

        F = force(q)

        p = p + h*F*factor

        return [q,p]


    @njit()
    def O_step( qp , h,gamma, factor=1):
        q,p = qp

        alpha = np.exp(-h*gamma)

        R = np.random.randn( q.size ).reshape( q.shape)
        p = np.exp(- gamma*h *factor)*p+ np.sqrt(1-np.exp(-gamma*h*factor*2))*R

        return [q,p]

    @njit()
    def baoab_step(q,p,h,gamma):
        qp = [q,p]  #this just translates the separate q and p vectors
                    #into a single vector composed from the pair.
        qp = B_step( qp, h,0.5)
        qp = A_step(qp , h , 0.5)
        qp = O_step( qp, h, gamma, 1)
        qp = A_step( qp, h,0.5)
        qp = B_step( qp, h,  0.5)


        q,p = qp
        return q , p
    return baoab_step



def make_simulation(step_function):
    @njit()
    def run_simulation(Nsteps, h, gamma,q_init= np.array([]),p_init= np.array([])):
        q_traj = [q_init]
        q = np.copy(q_init)
        p = np.copy(p_init)
        t = 0

        for n in range(Nsteps):
            q,p = step_function(q, p, h, gamma)
            t = t + h

            q_traj += [q]


        return q_traj
    return run_simulation

def make_numba_convergence(step_function):
    @njit()
    def run_numba_convergence(error, function,value, h, gamma,q_init= np.array([]),p_init= np.array([])):
        mean = 0
        its = 0
        f_its = 0
        totals = is_converged  = np.zeros_like(q_init)
        not_converged = q_means = np.ones_like(q_init)
        q = np.copy(q_init)
        p = np.copy(p_init)
        t = 0

        while np.any(not_converged) and its < 200000:
            its +=1
            q,p = step_function(q, p, h, gamma)
            if its > 10000:
                totals += function(q)
                f_its +=1
            not_converged = (np.abs(totals/its -value) > error).astype(np.float64)


        return f_its
    return run_numba_convergence

def make_numba_average_convergence(step_function):
    @njit()
    def run_numba_average_convergence(error, function,value, h, gamma,q_init= np.array([]),p_init= np.array([])):
        its = f_its = 0
        totals = np.zeros_like(q_init)
        means = np.ones_like(q_init)*100
        not_converged = True
        q = np.copy(q_init)
        p = np.copy(p_init)

        while not_converged and f_its < 100000:
            its +=1
            #print(q)
            q,p = step_function(q, p, h, gamma)
            if its > 10000:
                f_its += 1
                totals += function(q)
                means = totals / f_its
            not_converged = (np.abs(means -value)).mean() > error
        return its
    return run_numba_average_convergence

def make_simulate_trajectories(step_function, run_simulation):
    @njit()
    def simulate_trajectories(n_steps, stepsize, gamms, q_init = np.array([]), p_init = np.array([])):
        trajectories = []
        for gamm in gamms:
            q_traj = run_simulation(n_steps , stepsize, gamm,q_init, p_init)
            trajectories.append(q_traj)
        return gamms, trajectories
    return simulate_trajectories



@jit(nopython=True)
def square(x):
    return x**2

@jit(nopython=True)
def nothing(x):
    return x