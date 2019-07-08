# PSO-
用于多变量的目标函数优化
#encoding=utf8
import random
import math
import matplotlib.pyplot as plt
from optimal1 import optimal_parameter
import numpy as np

a=[150,450]
b=[50,60]
l0=[150,280]
l=[100,250]
theta=[30,60]
boudary=[a,b,l0,l,theta]

class PSO:
    def __init__(self,dim,size,iter_num,boudary,C1,C2,W):
        self.C1=C1
        self.C2=C2
        self.W=W     #惯性权重
        self.dim=dim      #搜索空间维度
        self.size=size       #粒子群数量
        self.iter_num=iter_num    #迭代次数
        self.boudary=boudary      #搜索空间
        #粒子群位置的初始化
        self.pos=[]
        for i in range(self.size):
            x=[]
            for i in range(len(self.boudary)):
                x.append(random.randrange(boudary[i][0],boudary[i][1]))     
            self.pos.append(x)
        self.pos=np.array(self.pos)#list转化为array
        #粒子群速度的初始化
        self.v=[]
        for i in range(self.size):
            x=[]
            for i in range(len(self.boudary)):
                x.append(random.randrange(0,(boudary[i][1]-boudary[i][0])))     
            self.v.append(x) 
        self.v=np.array(self.v)
        fitness=self.calculate_fitness(self.pos)
        self.p=self.pos #个体最佳位置
        self.pg=self.pos[np.argmax(fitness)]#全局最佳位置
        self.individual_best_fitness=fitness
        self.global_best_fitness=np.max(fitness)
        self.fitness_val_list=[]
        
    def calculate_fitness(self,x):
        result=[]
        for i in range(len(x)):
            result.append(optimal_parameter(x[i][0],x[i][1],x[i][2],x[i][3],x[i][4]))
        return result
    
    def evolve(self):
        for step in range(self.iter_num):
            r1=np.random.rand(self.size,self.dim)
            r2=np.random.rand(self.size,self.dim)
            #更新权重和向量
            self.v=self.W*self.v+self.C1*r1*(self.p-self.pos)+self.C2*r2*(self.pg-self.pos)
            self.pos=self.v+self.pos
            #限制位置的范围
            for i in range(len(self.pos)):
                for j in range(len(boudary)):
                    if self.pos[i][j]<np.min(boudary[j]):
                        self.pos[i][j]=np.min(boudary[j])
                    if self.pos[i][j]>np.max(boudary[j]):
                        self.pos[i][j]=np.max(boudary[j])
                        
            fitness=self.calculate_fitness(self.pos)
            for i in range(len(fitness)):
                if self.individual_best_fitness[i]<fitness[i]:
                    self.p[i]=self.pos[i]
                    self.individual_best_fitness[i]=fitness[i]
            if np.max(fitness)>self.global_best_fitness:
                self.pg=self.pos[np.argmax(fitness)]
                self.global_best_fitness=np.max(fitness)
            self.fitness_val_list.append(self.global_best_fitness)
        return self.pg,self.fitness_val_list
dim=5
size=10
iter_num=20
C1=C2=2
W=0.6

pso=PSO(dim,size,iter_num,boudary,C1,C2,W)
result,fitness_val_list=pso.evolve()
plt.rc('font',family='Times New Roman',size=12)
plt.plot(np.linspace(0, iter_num, iter_num), fitness_val_list, c="R", alpha=0.5)
plt.title("PSO search for v")
plt.show()
print(result)
        
    
    


