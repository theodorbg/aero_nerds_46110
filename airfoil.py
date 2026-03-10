import numpy as np
import matplotlib.pyplot as plt

class Airfoil:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.m 

    def camber_line(self, x, p, m, c):
        if x>=0 and x<=p:
            yc = m/p**2 * (2*p*x/c-(x/c)**2)
        if x>p and x<=1:
            yc = m/(1-p)**2 * ((1-2*p)+2*p*x/c-(x/c)**2)
        
        return yc

    def plot(self):
        plt.plot(self.x, self.y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Airfoil Shape')
        plt.axis('equal')
        plt.grid()
        plt.show()