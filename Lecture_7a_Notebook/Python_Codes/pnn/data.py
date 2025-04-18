"""
@author: jpzxshi
"""
import numpy as np
import learner as ln
from learner.integrator.hamiltonian import SV
import matplotlib.pyplot as plt

class LVData(ln.Data):
    '''H(u,v)=u−ln(u)+v−2ln(v), B(u,v)=[[0, uv], [-uv, 0]].
    (p,q)=(ln(u),ln(v)): K(p,q)=p-exp(p)+2q-exp(q) 
    '''
    def __init__(self, z0, h, train_num, test_num):
        super(LVData, self).__init__()
        self.f = lambda t, y: y * (y @ np.array([[0, -1], [1, 0]]) + np.array([-2, 1]))
        self.dK = lambda p, q: (np.ones_like(p) - np.exp(p), 2 * np.ones_like(q) - np.exp(q))
        self.solver = SV(None, self.dK, iterations=1, order=4, N=10)
        self.z0 = np.array(z0)
        self.h = h
        self.train_num = train_num
        self.test_num = test_num
        self.__init_data()
        
    def generate_flow(self, z0, h, num):
        x0 = np.log(np.array(z0))
        return np.exp(self.solver.flow(x0, h, num))
        
    def __init_data(self):
        train_flow = self.generate_flow(self.z0, self.h, self.train_num)
        test_flow = self.generate_flow(train_flow[..., -1, :], self.h, self.test_num)
        self.X_train = train_flow[..., :-1, :].reshape(-1, 2)
        self.y_train = train_flow[..., 1:, :].reshape(-1, 2)
        self.X_test = test_flow[..., :-1, :].reshape(-1, 2)
        self.y_test = test_flow[..., 1:, :].reshape(-1, 2)
    
class PDData(ln.Data):
    '''H(u,v,r)=u^2/2-cos(v)+ur-u^3-uv^2, 
    B(u,v,r)=[[0, -1, -2v], [1, 0, 2u], [2v, -2u, 0]].
    '''
    def __init__(self, z0, h, train_num, test_num):
        super(PDData, self).__init__()
        self.z0 = np.array(z0)
        self.h = h
        self.train_num = train_num
        self.test_num = test_num
        self.__init_data()
        
    @property
    def latent_dim(self):
        return 2
    
    def solve_flow(self, x, h, num):
        x = x.reshape([-1, 3])
        res = []
        for x0 in x:
            dH = lambda p, q: (p + x0[-1], np.sin(q))
            res.append(SV(None, dH, iterations=1, order=4, N=10).flow(x0[:2], h, num))
        return np.array(res) if len(res) > 1 else res[0]
        
    def generate_flow(self, z0, h, num):
        # (u,v,r) to (p,q,c)
        x0 = z0 - np.array([0, 0, 1]) * (z0[..., 0:1] ** 2 + z0[..., 1:2] ** 2)
        # solve
        X = self.solve_flow(x0, h, num)
        shape = (X.shape[0], X.shape[1], 1) if len(X.shape) == 3 else (X.shape[0], 1)
        X = np.concatenate((X, np.tile(x0[..., -1:], [1, num + 1]).reshape(shape)), axis=-1)
        # (p,q,c) to (u,v,r)
        Z = X + np.array([0, 0, 1]) * (X[..., 0:1] ** 2 + X[..., 1:2] ** 2)
        return Z
        
    def __init_data(self):
        train_flow = self.generate_flow(self.z0, self.h, self.train_num)
        test_flow = self.generate_flow(train_flow[..., -1, :], self.h, self.test_num)
        self.X_train = train_flow[..., :-1, :].reshape(-1, 3)
        self.y_train = train_flow[..., 1:, :].reshape(-1, 3)
        self.X_test = test_flow[..., :-1, :].reshape(-1, 3)
        self.y_test = test_flow[..., 1:, :].reshape(-1, 3)
    
class LFData(ln.Data):
    '''The two-dimensional dynamics of the charged particle in the electromagnetic field 
    governed by the Lorentz force.
    m * x_tt = q * (E + x_t cross B)
    H(v,x)=v^T*v/2+1/(100*sqrt(x1^2+x2^2)), B(v,x)=[[-B_hat(x), -I], [I, 0]]
    B_hat(x)=[[0, -sqrt(x1^2+x2^2)], [sqrt(x1^2+x2^2), 0]]
    (p,x)=(v+A(x),x), A(x)=sqrt(x1^2+x2^2)/3*(-x2,x1)^T
    K(p,x)=(p-A(x))^T*(p-A(x))/2+1/(100*sqrt(x1^2+x2^2))
    v=(v1, v2), x=(x1, x2), z=(v, x)=(v1, v2, x1, x2)
    '''
    def __init__(self, z0, h, train_num, test_num):
        super(LFData, self).__init__()
        self.z0 = z0
        self.h = h
        self.train_num = train_num
        self.test_num = test_num
        
        self.solver = SV(None, self.dK, iterations=10, order=4, N=max(int(self.h * 1e1), 1))
        self.__init_data()
        
    def dK(self, p, x):
        R = np.linalg.norm(x, axis=-1, keepdims=True)
        A = x @ np.array([[0, 1], [-1, 0]]) * (R / 3)
        dA = np.hstack([
                - x[..., :1] * x[..., 1:],
                - x[..., :1] ** 2 - 2 * x[..., 1:] ** 2,
                2 * x[..., :1] ** 2 + x[..., 1:] ** 2,
                x[..., :1] * x[..., 1:]
                ]) / (R * 3)
        dA = dA.reshape([-1, 2, 2])
        dphi = x / (- 100 * R ** 3)
        dp = p - A
        dx = dphi - (np.expand_dims(dp, axis=-2) @ dA).squeeze()
        return dp, dx
    
    def generate_flow(self, z0, h, num):
        z0 = np.array(z0)
        # (v,x) to (p,x)
        v, x = z0[:2], z0[2:]
        p = v + np.array([-x[1], x[0]]) * (np.linalg.norm(x) / 3)
        # solve
        flow = self.solver.flow(np.hstack((p, x)), h, num)
        # (p,x) to (v,x)
        p, x = flow[..., :2], flow[..., 2:]
        v = p - x @ np.array([[0, 1], [-1, 0]]) * (np.linalg.norm(x, axis=-1, keepdims=True) / 3)
        return np.hstack((v, x))
    
    def __init_data(self):
        train_flow = self.generate_flow(self.z0, self.h, self.train_num)
        test_flow = self.generate_flow(train_flow[-1], self.h, self.test_num)
        self.X_train, self.y_train = train_flow[:-1], train_flow[1:]
        self.X_test, self.y_test = test_flow[:-1], test_flow[1:]
    
class ALData(ln.Data):
    '''Ablowitz–Ladik Discrete Nonlinear Schrodinger Equation.
    i*w_t+w_xx+2|w|^2*w=0, w(x,0)=w0(x), w0(x+1)=w0(x), w0=u0+i*v0.
    z0=(u0,v0)=(u0(1/N),u0(2/N),...,u0(N/N),v0(1/N),...,v0(N/N)).
    '''
    def __init__(self, z0, h, train_num, test_num):
        super(ALData, self).__init__()
        self.z0 = z0
        self.h = h
        self.train_num = train_num
        self.test_num = test_num
        
        self.solver = SV(None, self.dK, iterations=10, order=4, N=max(int(self.h * 1e4), 1))
        self.__init_data()
        
    def dK(self, p, q):
        N = len(self.z0) // 2
        dx = 1 / N
        
        p0 = np.roll(p, 1, axis=-1)
        p1 = p
        p2 = np.roll(p, -1, axis=-1)
        q0 = np.roll(q, 1, axis=-1)
        q1 = q
        q2 = np.roll(q, -1, axis=-1)
        
        a0 = dx * dx * (p0 * p0 + q0 * q0)
        a1 = dx * dx * (p1 * p1 + q1 * q1)
        a2 = dx * dx * (p2 * p2 + q2 * q2)
        
        tau0 = (np.exp(a0) - 1) / a0
        tau1 = (np.exp(a1) - 1) / a1
        tau2 = (np.exp(a2) - 1) / a2
        
        dtau1 = (np.exp(a1) * (a1 - 1) + 1) / (a1 ** 2)
        
        dp = N * N * (2 * dx * dx * p1 * dtau1 * (tau0 * (p1 * p0 + q1 * q0) + tau2 * (p1 * p2 + q1 * q2))
             + tau1 * (tau0 * p0 + tau2 * p2) - 2 * p1)
        dq = N * N * (2 * dx * dx * q1 * dtau1 * (tau0 * (p1 * p0 + q1 * q0) + tau2 * (p1 * p2 + q1 * q2))
             + tau1 * (tau0 * q0 + tau2 * q2) - 2 * q1)
        return dp, dq
        
    def generate_flow(self, z0, h, num):
        z0 = np.array(z0)
        N = len(z0) // 2
        dx = 1 / N
        # (u,v) to (p,q)
        u, v = z0[:N], z0[N:]
        a = dx * dx * (u * u + v * v)
        sigma = np.sqrt(np.log(1 + a) / a)
        p, q = u * sigma, v * sigma
        # solve
        flow = self.solver.flow(np.hstack((p, q)), h, num)
        # (p,q) to (u,v)
        p, q = flow[..., :N], flow[..., N:]
        b = dx * dx * (p * p + q * q)
        tau = (np.exp(b) - 1) / b
        u, v = p * tau, q * tau
        return np.hstack((u, v))
        
    def __init_data(self):
        train_flow = self.generate_flow(self.z0, self.h, self.train_num)
        test_flow = self.generate_flow(train_flow[-1], self.h, self.test_num)
        self.X_train, self.y_train = train_flow[:-1], train_flow[1:]
        self.X_test, self.y_test = test_flow[:-1], test_flow[1:]
    
class TBData(ln.Data):
    '''Images of two-body.
    '''
    def __init__(self, h, train_num, test_num, size=50):
        super(TBData, self).__init__()
        self.h = h
        self.train_num = train_num
        self.test_num = test_num
        
        self.x0 = np.array([0.3, 0, -0.3, 0, 0.5, 0.2, 0.5, 1.8])
        self.r = 0.07
        self.size = size
        self.solver = SV(None, self.dH, iterations=1, order=4, N=max(int(self.h * 1e3), 1))
        self.__init_data()
        
    def dH(self, p, q):
        q1 = q[..., :2]
        q2 = q[..., 2:]
        r = np.linalg.norm(q1 - q2, axis=-1, keepdims=True)
        dp = p
        dq = (r ** -3) * np.hstack((q1 - q2, q2 - q1))
        return dp, dq

    def show_image(self, flat_data, save_path=None):
        plt.figure()
        plt.imshow(flat_data.reshape(2 * self.size, 1 * self.size), cmap='gray')
        plt.axis('off')
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        plt.close()

    def image(self, x):
        size_x = 1 * self.size
        size_y = 2 * self.size
        A = np.ones(size_y)[:, None] * np.arange(size_x) / self.size
        B = np.arange(size_y)[::-1, None] * np.ones(size_x) / self.size 
        sigmoid = lambda x: 1/(1+np.exp(-x))
        b1 = sigmoid((self.r ** 2 - ((A - x[4]) ** 2 + (B - x[5]) ** 2)) * 100)
        b2 = sigmoid((self.r ** 2 - ((A - x[6]) ** 2 + (B - x[7]) ** 2)) * 100)
        return np.maximum(b1, b2)
        
    def __init_data(self):
        flow_train = self.solver.flow(self.x0, self.h, self.train_num)
        flow_test = self.solver.flow(flow_train[-1], self.h, self.test_num)
        flow_train_image = np.array(list(map(self.image, flow_train))).reshape(flow_train.shape[0], -1)
        flow_test_image = np.array(list(map(self.image, flow_test))).reshape(flow_test.shape[0], -1)
        self.X_train_raw, self.y_train_raw = flow_train[:-1], flow_train[1:]
        self.X_test_raw, self.y_test_raw = flow_test[:-1], flow_test[1:]
        self.X_train, self.y_train = flow_train_image[:-1], flow_train_image[1:]
        self.X_test, self.y_test = flow_test_image[:-1], flow_test_image[1:]