"""
@author: jpzxshi
"""
import numpy as np
import matplotlib.pyplot as plt
import learner as ln

def LV_plot(data, net):
    steps = 1500
    z0_test = data.X_test_np.reshape(-1, data.test_num, data.dim)[:, 0, :]
    flow_true = data.generate_flow(z0_test, data.h, steps)
    flow_pred = net.predict(z0_test, steps, keepinitx=True, returnnp=True)
    from scipy.integrate import solve_ivp
    flow_rk45 = np.array([solve_ivp(data.f, [0, data.h * steps], z, 'RK45', np.linspace(0, data.h * steps, steps + 1)).y.T
                          for z in z0_test])
    
    plt.figure(figsize=[6 * 2, 4.8 * 1])
    plt.subplot(121)
    for i in range(z0_test.shape[0]):
        label_true = 'Ground truth' if i == 0 else None
        label_pred = 'PNN' if i == 0 else None
        label_rk45 = 'RK45' if i == 0 else None
        plt.plot(flow_true[i, :, 0], flow_true[i, :, 1], color='b', label=label_true, zorder=1)
        plt.plot(flow_pred[i, :, 0], flow_pred[i, :, 1], color='r', label=label_pred, zorder=2)
        plt.plot(flow_rk45[i, :, 0], flow_rk45[i, :, 1], color='lightgreen', linestyle='--', label=label_rk45, zorder=0)
    plt.title('Original space')
    plt.legend()
    
    plt.subplot(122)
    if not isinstance(net, ln.nn.PNN):
        flow_true_ori = flow_true
        flow_pred_ori = flow_pred
        flow_rk45_ori = flow_rk45
    else:
        flow_true_ori = net.inn.predict(flow_true.reshape(-1, data.dim), returnnp=True).reshape(-1, steps + 1, data.dim)
        flow_pred_ori = net.inn.predict(flow_pred.reshape(-1, data.dim), returnnp=True).reshape(-1, steps + 1, data.dim)
        flow_rk45_ori = net.inn.predict(flow_rk45.reshape(-1, data.dim), returnnp=True).reshape(-1, steps + 1, data.dim)
    for i in range(z0_test.shape[0]):
        label_true = 'Ground truth' if i == 0 else None
        label_pred = 'PNN' if i == 0 else None
        label_rk45 = 'RK45' if i == 0 else None
        plt.plot(flow_true_ori[i, :, 0], flow_true_ori[i, :, 1], color='b', label=label_true, zorder=1)
        plt.plot(flow_pred_ori[i, :, 0], flow_pred_ori[i, :, 1], color='r', label=label_pred, zorder=2)
        plt.plot(flow_rk45_ori[i, :, 0], flow_rk45_ori[i, :, 1], color='lightgreen', linestyle='--', label=label_rk45, zorder=0)
    plt.title('Latent symplectic manifold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('LV.pdf')
    
def PD_plot(data, net):
    n = data.z0.shape[0] if len(data.z0.shape) == 2 else 1
    z = [data.X_test[data.test_num * i] for i in range(n)]
    
    fig = plt.figure(figsize=[6.4 * 2, 4.8 * 1])
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    for i in range(n):
        u = data.y_test_np[data.test_num * i: data.test_num * (i + 1), 0]
        v = data.y_test_np[data.test_num * i: data.test_num * (i + 1), 1]
        r = data.y_test_np[data.test_num * i: data.test_num * (i + 1), 2]
        ax1.plot(u, v, r, color='b')
        pred = net.predict(z[i], steps=data.test_num, keepinitx=False, returnnp=True)
        u = pred[:, 0]
        v = pred[:, 1]
        r = pred[:, 2]
        ax1.plot(u, v, r, color='r')
    ax1.set_xlabel('u')
    ax1.set_ylabel('v')
    ax1.set_zlabel('r')
    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    for i in range(n):
        flow_true = net.inn.predict(data.y_test[data.test_num * i: data.test_num * (i + 1)], returnnp=True)
        u = flow_true[:, 0]
        v = flow_true[:, 1]
        r = flow_true[:, 2]
        ax2.plot(u, v, r, color='b')
        pred = net.inn.predict(net.predict(z[i], steps=data.test_num, keepinitx=False), returnnp=True)
        u = pred[:, 0]
        v = pred[:, 1]
        r = pred[:, 2]
        ax2.plot(u, v, r, color='r')
    ax2.set_xlabel('u')
    ax2.set_ylabel('v')
    ax2.set_zlabel('r')
    
    fig.savefig('PD.pdf')
    
    
def LF_plot(data, net):
    def predict(net, x, step):
        import torch
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=net.dtype, device=net.device)
        pred = [x]
        for _ in range(step):
            pred.append(net(pred[-1]))
        return torch.stack(pred).cpu().detach().numpy()
    steps = 500
    flow_true = data.generate_flow(data.X_test_np[0], data.h, steps)
    if isinstance(net, ln.nn.INN):
        flow_pred = predict(net, data.X_test[0], steps)
    else:
        flow_pred = net.predict(data.X_test[0], steps, keepinitx=True, returnnp=True)
    
    plt.figure(figsize=[6.4 * 2, 4.8 * 1])
    plt.subplot(121)
    plt.plot(flow_true[:, 2], flow_true[:, 3], color='b', label='Reference')
    plt.plot(flow_pred[:, 2], flow_pred[:, 3], color='r', label='PNN')
    plt.xlabel(r'$x_1$', fontsize=13)
    plt.ylabel(r'$x_2$', fontsize=13)
    plt.legend(fontsize=13)
    
    plt.subplot(122)
    plt.plot(np.arange(steps+1), flow_true[:, 2], color='b', label='Reference')
    plt.plot(np.arange(steps+1), flow_pred[:, 2], color='r', label='PNN')
    plt.xlabel(r'Step', fontsize=13)
    plt.ylabel(r'$x_1$', fontsize=13)
    plt.legend(fontsize=13)
    
    plt.savefig('LF.pdf')
    
def AL_plot(data, net):
    flow_true = np.vstack((data.X_train_np, data.X_test_np))
    flow_pred = net.predict(data.X_train[0], steps=flow_true.shape[0]-1, keepinitx=True, returnnp=True)
    flow_true2 = data.X_test_np
    flow_pred2 = net.predict(data.X_test[0], steps=flow_true2.shape[0]-1, keepinitx=True, returnnp=True)
    
    plt.figure(figsize=[20, 10])
    
    plt.subplot(211)
    plt.plot(flow_true[:, 0], label='Reference')
    plt.plot(flow_pred[:, 0], label='PNN')
    plt.legend()
    plt.ylim([-2.5, 2.5])
    
    plt.subplot(212)
    plt.plot(flow_true2[:, 0], label='Reference')
    plt.plot(flow_pred2[:, 0], label='PNN')
    plt.legend()
    plt.ylim([-2.5, 2.5])
    
    plt.savefig('AL.pdf')
    
def TB_plot(data, net):
    import matplotlib.animation as animation 
    def generate_gif(d, path):
        fig = plt.figure(tight_layout=True)
        image = plt.imshow(d[0].reshape(2 * data.size, 1 * data.size), cmap='gray')
        plt.axis('off')
        def update_points(num):
            image.set_data(d[num].reshape(2 * data.size, 1 * data.size))
            return image,
        interval = max(int(50 / net.recurrent), 1)
        anim = animation.FuncAnimation(fig, update_points, np.arange(d.shape[0]), interval=interval, blit=True)
        anim.save(path)
    pred_im = net.predict(data.X_test[0], steps=data.test_num * net.recurrent, keepinitx=False, returnnp=True)
    generate_gif(pred_im, 'TB.gif')