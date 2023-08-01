import numpy as np
import pandas as pd

def best_fit(pointset_1: np.ndarray, pointset_2: np.ndarray):
    '''
    Returns rotation matrix (r) and translation vector (t).
    Use to report pointset_1 with respect to pointset_2 using
    r@p + t
    where p is the coordinate point you would like to transform
    '''
    A = pointset_1
    B = pointset_2
    Am = np.mean(A, axis=0)
    Bm = np.mean(B, axis=0)
    A = A - Am
    B = B - Bm
    C = A.T @ B
    u, s, vt = np.linalg.svd(C)
    r = vt.T @ u.T
    t = Bm - r@Am
    return r, t

def fit_plane(xyz: np.ndarray):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    
    A = np.array([[np.sum(x**2), np.sum(x*y), np.sum(x)],
              [np.sum(x*y), np.sum(y**2), np.sum(y)],
              [np.sum(x), np.sum(y), xyz.shape[0]]])
    b = np.array([np.sum(x*z), np.sum(y*z), np.sum(z)])

    fit = np.dot(np.linalg.inv(A), b)
    return fit

def line(x, y):
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_dev = x - x_m
    y_dev = y - y_m
    xy_dev = x_dev * y_dev
    xsq_dev = x_dev**2
    m = np.sum(xy_dev)/np.sum(xsq_dev)
    b = y_m - m * x_m
    bf_y = m*x + b
    fitted = y - m*x - b
    return (np.array([m, b]), fitted)

def circle(x, y):
    x_m = np.mean(x)
    y_m = np.mean(y)

    u = x - x_m
    v = y - y_m

    suv = np.sum(u*v)
    suu = np.sum(u**2)
    svv = np.sum(v**2)
    suuv = np.sum(u**2 * v)
    suvv = np.sum(u * v**2)
    suuu = np.sum(u**3)
    svvv = np.sum(v**3)

    a = np.array([[suu, suv], [suv, svv]])
    b = np.array([suuu + suvv, svvv + suuv])/2.0
    uc, vc = np.linalg.solve(a, b)

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    ri_1 = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
    r_1 = np.mean(ri_1)
    #residu_1 = np.sum((ri_1-r_1)**2)

    x_1 = x - xc_1
    y_1 = y - yc_1

    # return xy center coordinates, radius, and xy coordinates about center
    return np.array([xc_1, yc_1]), r_1, np.array([x_1, y_1])

def sphere(x, y, z):
    a = np.zeros((len(x), 4))
    a[:, 0] = 2*x
    a[:, 1] = 2*y
    a[:, 2] = 2*z
    a[:, 3] = 1

    f = np.zeros((len(x), 1))
    f[:, 0] = x**2 + y**2 + z**2
    c, residuals, rank, singval = np.linalg.lstsq(a, f)
    form_deviation = max(residuals) - min(residuals)

    t = c[0]**2 + c[1]**2 + c[2]**2 + c[3]
    radius = np.sqrt(t)
    return radius, form_deviation, np.array([c[0], c[1], c[2]])

def excel2numpy(filename):
    df = pd.read_excel(filename, skiprows=12)
    xyz = np.array([df[df['Characteristic'].str.contains('X Value')].Actual, df[df['Characteristic'].str.contains('Y Value')].Actual, df[df['Characteristic'].str.contains('Z Value')].Actual]).T
    return xyz

def r2euler(r_matrix):
    theta_ry1 = -np.arcsin(r_matrix[2, 0])
    theta_ry2 = np.pi + np.arcsin(r_matrix[2, 0])
    theta_rx1 = np.arctan2(r_matrix[2, 1]/np.cos(theta_ry1), r_matrix[2, 2]/np.cos(theta_ry1))
    theta_rx2 = np.arctan2(r_matrix[2, 1]/np.cos(theta_ry2), r_matrix[2, 2]/np.cos(theta_ry2))
    theta_rz1 = np.arctan2(r_matrix[1, 0]/np.cos(theta_ry1), r_matrix[0, 0]/np.cos(theta_ry1))
    theta_rz2 = np.arctan2(r_matrix[1, 0]/np.cos(theta_ry2), r_matrix[0, 0]/np.cos(theta_ry2))
    return np.array([theta_rx1, theta_ry1, theta_rz1]), np.array([theta_rx2, theta_ry2, theta_rz2])

if __name__ == '__main__':
    pass
