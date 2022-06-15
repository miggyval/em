from random import gauss
from numpy import genfromtxt
import gaussfun
import numpy as np
import scipy.spatial.transform
import matplotlib.pyplot as plt
import cv2
import time
import pyvista as pv


def f_samples(x, w_list, mu_list, sigma_list, J, m):
    N = x.shape[0]
    norm_factor = 1 / np.sqrt(np.linalg.det(2 * np.pi * sigma_list))
    sig_inv_list = np.linalg.inv(sigma_list)
    
    mu_list = np.expand_dims(mu_list, [0])
    mu_list_tile = np.tile(mu_list, [N, 1, 1])

    sig_inv_list = np.expand_dims(sig_inv_list, [0])
    sig_inv_tile = np.tile(sig_inv_list, [N, 1, 1, 1])
    
    x = np.expand_dims(x, len(x.shape) - 1)
    x_tile = np.tile(x, [1, J, 1])
    x_bar_list = x_tile - mu_list_tile

    einsum = np.einsum('iln,ilmn,ilm->il', x_bar_list, sig_inv_tile, x_bar_list)
    exp_factor = np.exp(-einsum / 2)

    retval = np.einsum('il,l->i', exp_factor, norm_factor)

    return retval

def f_meshgrid(x, w_list, mu_list, sigma_list, J, m):
    kx, ky, kz = x.shape[:3]
    norm_factor = 1 / np.sqrt(np.linalg.det(2 * np.pi * sigma_list))
    sig_inv_list = np.linalg.inv(sigma_list)
    
    mu_list = np.expand_dims(mu_list, [0, 1, 2])
    mu_list_tile = np.tile(mu_list, [kx, ky, kz, 1, 1])

    sig_inv_list = np.expand_dims(sig_inv_list, [0, 1, 2])
    sig_inv_tile = np.tile(sig_inv_list, [kx, ky, kz, 1, 1, 1])
    
    x = np.expand_dims(x, len(x.shape) - 1)
    x_tile = np.tile(x, [1, 1, 1, J, 1])
    x_bar_list = x_tile - mu_list_tile

    einsum = np.einsum('ijkln,ijklmn,ijklm->ijkl', x_bar_list, sig_inv_tile, x_bar_list)
    exp_factor = np.exp(-einsum / 2)

    retval = np.einsum('ijkl,l->ijk', exp_factor, norm_factor)

    return retval


def f_j(x, mu_list, sigma_list, J, m):
    N = x.shape[0]

    norm_factor = 1 / np.sqrt(np.linalg.det(2 * np.pi * sigma_list))
    sig_inv_list = np.linalg.inv(sigma_list)
    mu_list = np.expand_dims(mu_list, [0])
    mu_list_tile = np.tile(mu_list, [N, 1, 1])
    
    sig_inv_list = np.expand_dims(sig_inv_list, [0])
    sig_inv_tile = np.tile(sig_inv_list, [N, 1, 1, 1])
    x = np.expand_dims(x, len(x.shape) - 1)
    x_tile = np.tile(x, [1, J, 1])
    x_bar_list = x_tile - mu_list_tile

    einsum = np.einsum('ijl,ijkl,ijk->ij', x_bar_list, sig_inv_list, x_bar_list)
    exp_factor = np.exp(-einsum / 2)

    retval = np.einsum('ij,j->ij', exp_factor, norm_factor)

    return retval

def sample(mu_list, sigma_list, w_list, J, m):
    w_bins = np.cumsum(w_list)
    j = np.digitize(np.random.random(), w_bins)
    mu = mu_list[j, :]
    sigma = sigma_list[j, :, :]
    return np.random.multivariate_normal(mean=mu, cov=sigma)


def em(x, w_list, T, J):
    start_time = time.time()
    N, m = x.shape
    rmse_mu = np.zeros((T,))
    rmse_sigma = np.zeros((T,))
    lr = np.zeros((T,))
    mu_hat = np.zeros((T + 1, J, m))
    sigma_hat = np.zeros((T + 1, J, m, m))
    mu_init = np.einsum('ij->j', x) / N
    sigma_init = np.einsum('ij,ik->jk', x - mu_init, x - mu_init) / N
    mu_hat[0, :, :] = np.random.multivariate_normal(mean=mu_init, cov=sigma_init, size=(J,))
    sigma_hat[0, :, :, :] = np.einsum('i,...->i...', np.ones((J,)), np.eye(m))
    Pc = w_list

    for t in range(T):

        Pxc = f_j(x[:, :], mu_hat[t, :, :], sigma_hat[t, :, :, :], J, m)
        den_bayes = np.einsum('ij,j->i', Pxc, Pc)
        den_bayes_recip = np.reciprocal(den_bayes)
        c = np.einsum('ij,j,i->ji', Pxc, Pc, den_bayes_recip)

        den_stat = np.einsum('ij->i', c)
        den_stat_recip = np.reciprocal(den_stat)
        mu_hat[t + 1, :, :] = np.einsum('ij,jk,i->ik', c, x, den_stat_recip)
        
        mu_hat_list = np.copy(mu_hat[t + 1, :, :])
        mu_hat_list = np.expand_dims(mu_hat_list, [0])
        mu_hat_tile = np.tile(mu_hat_list, [N, 1, 1])

        x_list = np.copy(x)
        x_list = np.expand_dims(x_list, len(x_list.shape) - 1)
        x_tile = np.tile(x_list, [1, J, 1])
        
        x_bar_tile = x_tile - mu_hat_tile
        sigma_hat[t + 1, :, :, :] = np.einsum('lk,kli,klj,l->lij', c, x_bar_tile, x_bar_tile, den_stat_recip)
        #rmse_sigma[t] = np.sqrt(np.mean(np.square(sigma_hat[t + 1, :, :, :] - sigma_hat[t, :, :, :])))
        #rmse_mu[t] = np.sqrt(np.mean(np.square(mu_hat[t + 1, :, :] - mu_hat[t, :, :])))
        #L1 = f_samples(x, w_list, mu_hat[t + 1, :,], sigma_hat[t + 1, :, :], J, m)
        #L2 = f_samples(x, w_list, mu_hat[t, :,], sigma_hat[t, :, :], J, m)
        #lr[t] = np.prod(np.divide(L1, L2))
    end_time = time.time()
    return (mu_hat, sigma_hat, rmse_mu, rmse_sigma, lr)

def em2d():
    m = 3
    data = genfromtxt('data.csv', delimiter=',')
    x = (data[:, 0:2] - data[:, 2:4]) / 100
    z = 0.01 * np.max(x) * np.random.normal(size=(x[:, 0:1].shape))
    x = np.append(x, z, axis=1)
    xdiff = np.diff(x, 1, axis=0)

    for n in range(1, xdiff.shape[0] - 1):
        xdiff[n, 0:2] = (xdiff[n - 1, 0:2] + xdiff[n + 1, 0:2]) / 2


    N = xdiff.shape[0]
    J = 5
    w_list = np.ones((J,))
    w_list /= np.sum(w_list)
    T = 40
    mu_hat, sigma_hat, rmse_mu, rmse_sigma, lr = em(xdiff, w_list, T, J)

    steps = 400
    x_arr = np.linspace(-0.2, 0.2, steps)
    y_arr = np.linspace(-0.2, 0.2, steps)
    z_arr = np.linspace(-0.2, 0.2, steps)
    X, Y, Z = np.meshgrid(x_arr, y_arr, z_arr)
    print("Normal")
    F_hat = f_meshgrid(np.stack([X, Y, Z], axis=3), w_list, mu_hat[-1, :,], sigma_hat[-1, :, :], J, m)
    print("Done!")
    
    norm_factor = np.max(F_hat)

    F_out = F_hat[:, :, steps // 2] / norm_factor
    img_gray = (F_out * 255).astype(np.uint8)
    img = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
    img = cv2.flip(img, 0)
    cv2.imshow("img", img)
    c = cv2.waitKey(0)

    plt.scatter(xdiff[:, 0], xdiff[:, 1])
    plt.xlim([-0.2, 0.2])
    plt.ylim([-0.2, 0.2])
    plt.show()



def em3d():
    m = 3
    J = 5
    Q = scipy.spatial.transform.Rotation.random(J,).as_matrix()
    D = np.zeros((J, m, m))
    for j in range(J):
        D[j, :, :] = np.diag(np.random.uniform(low=2.5, high=5.5, size=(m,)))

    sigma_list = np.einsum('...ij,...jk,...lk->...il', Q, D, Q)
    mu_list = np.random.multivariate_normal(mean=np.zeros((m,)), cov=7.5 * np.eye(m), size=(J,))

    w_list = np.ones((J,))
    w_list /= np.sum(w_list)


    N = 500
    x = np.zeros((N, m))

    for n in range(N):
        x[n, :] = sample(mu_list, sigma_list, w_list, J, m)

    print("Done sampling!")

    T = 1000
    mu_hat, sigma_hat, rmse_mu, rmse_sigma, lr = em(x, w_list, T, J)

    steps = 150
    x_arr = np.linspace(-15, 15, steps)
    y_arr = np.linspace(-15, 15, steps)
    z_arr = np.linspace(-15, 15, steps)
    Y, X, Z = np.meshgrid(x_arr, y_arr, z_arr)
    print("Normal")
    F = f_meshgrid(np.stack([X, Y, Z], axis=3), w_list, mu_list, sigma_list, J, m)
    print("Estimate")
    F_hat = f_meshgrid(np.stack([X, Y, Z], axis=3), w_list, mu_hat[-1, :,], sigma_hat[-1, :, :], J, m)
    print("Done!")
    F_mean = (np.mean(F) + np.mean(F_hat)) / 2.0
    F_max = (np.max(F) + np.max(F_hat)) / 2.0
    F_min = (np.min(F) + np.min(F_hat)) / 2.0

    cdf = []
    vals = f_samples(x, w_list, mu_hat[-1, :,], sigma_hat[-1, :, :], J, m)
    
    i = 0
    step_size = 1e-5
    while True:
        u = i * step_size
        cdf.append(0)
        for n in range(N):
            if vals[n] >= u:
                cdf[i] += 1 / N
        if cdf[i] == 0:
            break
        i += 1

    cdf = np.array(cdf)
    for i in range(cdf.shape[0]):
        if cdf[i] <= 0.95:
            beta = i * step_size
            break

    grid = pv.UniformGrid()
    grid.dimensions = np.array(F.shape)
    grid.origin = (np.min(X), np.min(Y), np.min(Z))
    grid.spacing = (30 / steps, 30 / steps, 30 / steps)
    grid.point_data["values"] = F.flatten(order="F")


    grid_hat = pv.UniformGrid()
    grid_hat.dimensions = np.array(F_hat.shape)
    grid_hat.origin = (np.min(X), np.min(Y), np.min(Z))
    grid_hat.spacing = (30 / steps, 30 / steps, 30 / steps)
    grid_hat.point_data["values"] = F_hat.flatten(order="F")

    pcl = pv.PolyData(x)
    surface = grid.contour(compute_normals=True, isosurfaces=10, rng=[beta, beta])
    surface_hat = grid_hat.contour(compute_normals=True, isosurfaces=10, rng=[beta, beta])
    plotter = pv.Plotter(shape=(1, 2))
    plotter.subplot(0, 0)
    plotter.add_mesh(pcl, color='red', point_size=6.0, render_points_as_spheres=True, opacity=0.8)
    plotter.add_mesh(surface, opacity=0.1)
    plotter.subplot(0, 1)
    plotter.add_mesh(pcl, color='red', point_size=6.0, render_points_as_spheres=True, opacity=0.8)
    plotter.add_mesh(surface_hat, opacity=0.1)

    plotter.show()
    

    '''
    n = 0
    dir = 1
    norm_factor = max(np.max(F), np.max(F_hat))
    while True:
        F_slice = F[:, :, n] / norm_factor
        F_hat_slice = F_hat[:, :, n] / norm_factor
        img_gray = (F_slice * 255).astype(np.uint8)
        img_gray_hat = (F_hat_slice * 255).astype(np.uint8)
        F_diff = np.abs(F[:, :, n] - F_hat[:, :, n])
        img_diff = (F_diff / norm_factor * 255).astype(np.uint8)

        img = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
        img_hat = cv2.applyColorMap(img_gray_hat, cv2.COLORMAP_JET)
        img_diff_color = cv2.applyColorMap(img_diff, cv2.COLORMAP_JET)
        img_combined = cv2.hconcat([img, img_hat, img_diff_color])
        img_combined = cv2.resize(img_combined, (0, 0), fx=4.0, fy=4.0)
        cv2.imshow("img", img_combined)
        
        c = cv2.waitKey(16)
        if c == ord(' '):
            cv2.waitKey()
        n += dir
        if n == steps - 1 or n == 0:
            dir *= -1
    '''

if __name__ == "__main__":
    em3d()
