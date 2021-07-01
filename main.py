import matplotlib.pyplot as plt
from scipy.optimize import fsolve,leastsq
import numpy as np
import scipy.integrate

def equation(x):
    return (2*np.arctan(x)-1/(2*x**3))
def system_of_equation(xy):
    x,y=xy
    return (np.cos(x+0.5)-y-2,np.sin(y)-2*x-1)
def task_2_1():
    print('solution of the equation:')
    x0=fsolve(equation,1)
    print('x=',*x0)
    print('solution of the system of equations')
    x1,y1=fsolve(system_of_equation,(1,1))
    print('x=',x1)
    print('y=',y1)
    print('----')
def function(p,t):
    return p[0]*t+p[1]
def inaccuracy(p,t,r):
    return r-function(p,t)
def task_2_2_1():
    t=np.array([20,22,24,26,28])
    r=np.array([40,22.8,13.1,7.5,4.3])
    p0=np.array([0,0])
    p0,tmp=leastsq(inaccuracy,p0,args=(t,r))
    a,b=p0[0],p0[1]
    print('result:')
    print('R=',p0[0],'*T+',p0[1])
    print('-----------------')
    plt.scatter(t,r,c='r')
    plt.plot(t,t*p0[0]+p0[1])
    plt.show()
def func(p,t,NU):
    return np.log(NU)-p*t
def inaccur(p,t,n):
    NU = n[0]
    return np.log(n)-func(p,t,NU)
def task_2_2_2():
    t=np.array([0,0.747072,1.494144,2.241216,2.988288,3.73536])
    n=np.array([1000595,9434482,892400,840393,795430,748698])
    p0=0
    p0,tmp=leastsq(inaccur,p0,args=(t,n))
    print('result:')
    print('N(t)=N(0)*exp(-',*p0,'*t)')
    print('T(1/2)=',np.log(2)/p0[0])
    print('_________________')
    plt.scatter(t,np.log(n),c='r')
    plt.plot(t,np.log(n[0])-p0*t)
    plt.show()
def SIR(y,t,betta,gamma):
    S,I,R=y
    dS_dt=-betta*S*I
    dI_dt=betta*S*I-gamma*I
    dR_dt=gamma*I
    return(dS_dt,dI_dt,dR_dt)
def task_3_1():
    S0=0.9
    I0=0.1
    R0=0.0
    betta=0.35
    gamma=0.1
    t=np.linspace(0,100,10000)
    solution=scipy.integrate.odeint(SIR,[S0,I0,R0],t,args=(betta,gamma))
    solution=np.array(solution)
    print('result:')
    print(solution)
    plt.plot(t,solution[:,0],label='S(t)')
    plt.plot(t, solution[:,1],label='I(t)')
    plt.plot(t, solution[:,2],label='R(t)')
    plt.legend()
    plt.show()
def task_3_2():
    pass
def main():
    task_2_1()
    task_2_2_1()
    task_2_2_2()
    task_3_1()
if __name__ == '__main__':
    main()
