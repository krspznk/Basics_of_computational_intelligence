import matplotlib.pyplot as plt

class ustanova:
    def __init__(self, m_lower:int, m_upper:int, alpha:int, beta:int, h:float):
        self.m_lower=m_lower
        self.m_upper=m_upper
        self.alpha=alpha
        self.beta=beta
        self.h=h
    def prt(self):
        print(self.m_lower, self.m_upper, self.alpha, self.beta, self.h)

def counter(*args: ustanova)->ustanova:
    h=min(x.h for x in args)
    alpha=0
    for i in args:
        alpha+=i.alpha/i.h
    alpha*=h
    beta=0
    for i in args:
        beta+=i.beta/i.h
    beta*=h
    m_lower=alpha
    for i in args:
        m_lower+=(i.m_lower-i.alpha)
    m_upper=-beta
    for i in args:
        m_upper+=(i.m_upper+i.beta)
    return ustanova(m_lower, m_upper, alpha, beta, h)


def function(answer:ustanova):
    a1=[answer.m_lower-answer.alpha, 0]
    a2=[answer.m_lower, answer.h]
    a3=[answer.m_upper, answer.h]
    a4=[answer.m_upper+answer.beta, 0]
    a5=[answer.m_lower, 0]
    a6 = [answer.m_upper, 0]
    return [a1, a2, a3, a4, a2, a5, a3, a6]

a=ustanova(430, 430, 0, 0, 1)
b=ustanova(380, 480, 15, 50, 1)
c=[ustanova(300, 300, 100, 0, 0.8), ustanova(0, 0, 0, 0, 0.2)]
d=[ustanova(340, 350, 10, 20, 0.8), ustanova(0, 0, 0, 0, 0.2)]
e=[ustanova(365, 365, 0, 200, 0.2), ustanova(0,0,0, 0, 0.8)]
answer=[]
h_max=0
m_lower_max=0
m_upper_max=0
for ci in range(len(c)):
    for di in range(len(d)):
        for ei in range(len(e)):
            anw=counter(a, b, c[ci], d[di],  e[ei])
            if anw.h>h_max:
                h_max=anw.h
                m_lower_max=anw.m_lower
                m_upper_max=anw.m_upper
            print(f'A + B + C{ci+1} + D{di+1} + E{ei+1} = {anw.m_lower, anw.m_upper, anw.alpha, anw.beta, anw.h}')
            answer.append(anw)
print(f'Обсяг найбільш ймовірного фінансування в діапазоні [{m_lower_max}; {m_upper_max}]')
for anw in answer:
    point=function(anw)
    x=[]
    y=[]
    for i in point[0: 4]:
        x.append(i[0])
        y.append(i[1])
    plt.plot(x, y)
    x=[]
    y=[]
    for i in point[4: 6]:
        x.append(i[0])
        y.append(i[1])
    plt.plot(x, y, "--", alpha=0.1, c="black")
    x = []
    y = []
    for i in point[6: 8]:
        x.append(i[0])
        y.append(i[1])
    plt.plot(x, y, "--", alpha=0.1, c="black")
plt.show()