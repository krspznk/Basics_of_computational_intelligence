import function as function
import numpy as np
import matplotlib.pyplot as plt


def function(x):
    return (100 * np.sqrt(100 - x * x) * np.cos(x * x) * np.cos(x)) / ((x * x + 10) * np.log(100 - x * x))



class Star:
    def __init__(self,  n_population:int, min_population:int,max_population:int,  sigma:float):
        self.n_population=n_population
        self.min_population=min_population
        self.max_population=max_population
        self.sigma=sigma
        self.population_t = list(np.random.uniform(self.min_population, self.max_population) for x in range (self.n_population))
    def create_plot(self):
        X=np.linspace(self.min_population, self.max_population, self.n_population)
        Y=function(X)
        plt.plot(X, Y)

    def iterations(self, n:int):
        t=0
        while t<n:
            population_z=[]
            population_s=[]
            fit_t=[]
            fit_z=[]
            fit_s=[]
            for i in range (self.n_population):
                fit_t.append(function(self.population_t[i]))
                temp = self.population_t[i]+np.random.uniform(0,1)*self.sigma
                if(temp<-5):  temp=temp + 10;
                if (temp>5): temp=temp - 10;
                population_z.append(temp)
                fit_z.append(function(population_z[i]))
            for i in range (self.n_population):
                temp = (population_z[np.random.randint(0, self.n_population-1)] + population_z[np.random.randint(0, self.n_population-1)])/2
                population_s.append(temp)
                fit_s.append(function(population_s[i]))
            pop = np.hstack((np.array(self.population_t), np.array(population_z), np.array(population_s) ))
            fit = np.hstack((np.array(fit_t), np.array(fit_z), np.array(fit_z) ))
            fit, pop = zip(*sorted(zip(fit, pop), reverse = False))
            for i in range (self.n_population):
                self.population_t[i] = pop[i]
            print('{:^10}{:<25}{:<25}'.format(str(t), str(pop[0]), str(fit[0])))
            t+=1




Stars = Star( 20, -5, 5, 0.4)
Stars.create_plot()
print('{:^10}{:^25}{:^25}'.format('Iteration', ' X', ' Y'))
Stars.iterations(30)
plt.show()
