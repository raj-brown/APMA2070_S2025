"""
@author: jpzxshi
"""
import numpy as np
import learner as ln
from data import LVData, PDData, LFData, ALData, TBData
from postprocess import LV_plot, PD_plot, LF_plot, AL_plot, TB_plot
    
def LV():
    device = 'cpu' # 'cpu' or 'gpu'
    # data
    z0 = [[1, 0.8], [1, 1], [1, 1.2]]
    h = 0.1
    train_num = 100
    test_num = 100
    # PNN
    inn_volume_preserving = False
    inn_layers = 3
    inn_sublayers = 2
    inn_subwidth = 30
    inn_activation = 'sigmoid'
    symp_type = 'G' # 'LA' or 'G'
    symp_LAlayers = 3
    symp_LAsublayers = 2
    symp_Glayers = 3
    symp_Gwidth = 30
    symp_activation = 'sigmoid'
    # training
    lr = 0.001
    iterations = 200000
    print_every = 1000
    
    data = LVData(z0, h, train_num, test_num)
    inn = ln.nn.INN(data.dim, data.dim // 2, inn_layers, inn_sublayers, inn_subwidth, inn_activation, 
                    volume_preserving=inn_volume_preserving)
    if symp_type == 'LA':
        sympnet = ln.nn.LASympNet(data.dim, symp_LAlayers, symp_LAsublayers, symp_activation)
    elif symp_type == 'G':
        sympnet = ln.nn.GSympNet(data.dim, symp_Glayers, symp_Gwidth, symp_activation)
    net = ln.nn.PNN(inn, sympnet)
    args = {
        'data': data,
        'net': net,
        'criterion': 'MSE',
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': None,
        'print_every': print_every,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }
    
    ln.Brain.Init(**args)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    
    LV_plot(data, ln.Brain.Best_model())
    
def PD():
    device = 'cpu' # 'cpu' or 'gpu'
    # data
    z0 = [[0, 1, 0 + 1 ** 2 + 0.0], 
          [0, 1.5, 0 + 1.5 ** 2 + 0.1],
          [0, 2, 0 + 2 ** 2 + 0.2]]
    h = 0.1
    train_num = 100
    test_num = 100
    # PNN
    inn_volume_preserving = False
    inn_layers = 3
    inn_sublayers = 2
    inn_subwidth = 30
    inn_activation = 'sigmoid'
    #symp_type = 'E' # should be 'E' in this case
    symp_Elayers = 3
    symp_Ewidth = 30
    symp_activation = 'sigmoid'
    # training
    lr = 0.001
    iterations = 100000
    print_every = 1000
    
    data = PDData(z0, h, train_num, test_num)
    inn = ln.nn.INN(data.dim, data.latent_dim, inn_layers, inn_sublayers, inn_subwidth, inn_activation, 
                    volume_preserving=inn_volume_preserving)
    sympnet = ln.nn.ESympNet(data.dim, data.latent_dim, symp_Elayers, symp_Ewidth, symp_activation)
    net = ln.nn.PNN(inn, sympnet)
    args = {
        'data': data,
        'net': net,
        'criterion': 'MSE',
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': None,
        'print_every': print_every,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }
    
    ln.Brain.Init(**args)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    
    PD_plot(data, ln.Brain.Best_model())
    
def LF():
    device = 'gpu' # 'cpu' or 'gpu'
    # data
    z0 = [1, 0.5, 0.5, 1]
    h = 0.1
    train_num = 1500
    test_num = 300
    # PNN
    inn_volume_preserving = True
    inn_layers = 10
    inn_sublayers = 3
    inn_subwidth = 50
    inn_activation = 'sigmoid'
    symp_type = 'G' # 'LA' or 'G'
    symp_LAlayers = 3
    symp_LAsublayers = 2
    symp_Glayers = 10
    symp_Gwidth = 50
    symp_activation = 'sigmoid'
    # training
    lr = 0.001
    iterations = 2000000
    print_every = 1000
    
    data = LFData(z0, h, train_num, test_num)
    inn = ln.nn.INN(data.dim, data.dim // 2, inn_layers, inn_sublayers, inn_subwidth, inn_activation, 
                    volume_preserving=inn_volume_preserving)
    if symp_type == 'LA':
        sympnet = ln.nn.LASympNet(data.dim, symp_LAlayers, symp_LAsublayers, symp_activation)
    elif symp_type == 'G':
        sympnet = ln.nn.GSympNet(data.dim, symp_Glayers, symp_Gwidth, symp_activation)
    net = ln.nn.PNN(inn, sympnet)
    args = {
        'data': data,
        'net': net,
        'criterion': 'MSE',
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': None,
        'print_every': print_every,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }
    
    ln.Brain.Init(**args)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    
    LF_plot(data, ln.Brain.Best_model())
    
def AL():
    device = 'gpu' # 'cpu' or 'gpu'
    # data
    N = 20
    u0 = 2 + 0.2 * np.cos(2 * np.pi * np.linspace(0, 1, num=N+1)[1:])
    v0 = np.zeros_like(u0)
    z0 = np.hstack((u0, v0))
    h = 0.01
    train_num = 500
    test_num = 100
    # PNN
    inn_volume_preserving = False
    inn_layers = 4
    inn_sublayers = 2
    inn_subwidth = 100
    inn_activation = 'sigmoid'
    symp_type = 'G' # 'LA' or 'G'
    symp_LAlayers = 3
    symp_LAsublayers = 2
    symp_Glayers = 10
    symp_Gwidth = 100
    symp_activation = 'sigmoid'
    # training
    lr = 0.0001
    iterations = 1000000
    print_every = 1000
    
    data = ALData(z0, h, train_num, test_num)
    inn = ln.nn.INN(data.dim, data.dim // 2, inn_layers, inn_sublayers, inn_subwidth, inn_activation, 
                    volume_preserving=inn_volume_preserving)
    if symp_type == 'LA':
        sympnet = ln.nn.LASympNet(data.dim, symp_LAlayers, symp_LAsublayers, symp_activation)
    elif symp_type == 'G':
        sympnet = ln.nn.GSympNet(data.dim, symp_Glayers, symp_Gwidth, symp_activation)
    net = ln.nn.PNN(inn, sympnet)
    args = {
        'data': data,
        'net': net,
        'criterion': 'MSE',
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': None,
        'print_every': print_every,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }
    
    ln.Brain.Init(**args)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    
    AL_plot(data, ln.Brain.Best_model())
    
def TB():
    device = 'cpu' # 'cpu' or 'gpu'
    # data
    h = 0.1
    train_num = 40
    test_num = 100
    # AEPNN
    lam = 1
    recurrent = 2
    latent_dim = 2
    ae_depth = 2
    ae_width = 50
    ae_activation = 'sigmoid'
    symp_type = 'LA' # 'LA' or 'G'
    symp_LAlayers = 3
    symp_LAsublayers = 2
    symp_Glayers = 5
    symp_Gwidth = 50
    symp_activation = 'sigmoid'
    # training
    lr = 0.001
    iterations = 5000
    print_every = 1000
    
    data = TBData(h, train_num, test_num)
    ae = ln.nn.AE(data.dim, latent_dim, ae_depth, ae_width, ae_activation)
    if symp_type == 'LA':
        sympnet = ln.nn.LASympNet(latent_dim, symp_LAlayers, symp_LAsublayers, symp_activation)
    elif symp_type == 'G':
        sympnet = ln.nn.GSympNet(latent_dim, symp_Glayers, symp_Gwidth, symp_activation)
    net = ln.nn.AEPNN(ae, sympnet, lam, recurrent)
    args = {
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': None,
        'print_every': print_every,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }
    
    ln.Brain.Init(**args)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    
    TB_plot(data, ln.Brain.Best_model())

def main():
    LV()
    #PD()
    #LF()
    #AL()
    #TB()
    
if __name__ == '__main__':
    main()