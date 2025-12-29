import numpy as np
import matplotlib.pyplot as plt

X_data = np.array([12.2983, 10.6872, 9.8521, 9.6342, 9.8792, 10.775, 12.9678, 18.9971])
Y_data = np.array([0.0107, 1.0624, 1.9723, 2.9919, 4.1788, 5.886, 8.8884, 15.981])

def get_conic_type(A, B, C, e, is_linear):
    if is_linear:
        discriminant = B**2 - 4*A*C
        if discriminant < 0:
            return "Ellipse"
        elif discriminant == 0:
            return "Parabola"
        else: 
            return "Hyperbola"
    else:
        if 0.0 < e < 1.0:
            return "Ellipse"
        elif e == 1:
            return "Parabola"
        else: # e > 1
            return "Hyperbola"
        
# Linear Regression: Normal Equations and QR Decomposition
def solve_normal_equations(A, y):
    AtA = np.dot(A.T, A)
    Aty = np.dot(A.T, y)
    theta = np.linalg.solve(AtA, Aty)
    return theta

def gram_schmidt_qr(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v)
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] == 0:
            Q[:, j] = v
        else:
            Q[:, j] = v / R[j, j]
    return Q, R

def back_substitution(R, b):
    n = R.shape[0]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] = x[i] - R[i, j] * x[j]
        x[i] = x[i] / R[i, i]
    return x

def solve_qr_decomposition(A, b):
    Q, R = gram_schmidt_qr(A)
    Qtb = np.dot(Q.T, b)
    theta = back_substitution(R, Qtb)
    return theta

M_linear = np.column_stack([X_data**2, X_data * Y_data, Y_data**2, X_data, Y_data])
Y_target = -1 * np.ones(len(X_data))

theta_normal  = solve_normal_equations(M_linear, Y_target)
params_normal = np.append(theta_normal, 1.0)

theta_qr  = solve_qr_decomposition(M_linear, Y_target)
params_qr = np.append(theta_qr, 1.0)

A_n, B_n, C_n, D_n, E_n, F_n = params_normal
A_q, B_q, C_q, D_q, E_q, F_q = params_qr

conic_type_normal = get_conic_type(A_n, B_n, C_n, 0, True)
conic_type_qr     = get_conic_type(A_q, B_q, C_q, 0, True)

print(f"Part A: Linear Regression")
print(f"Normal Eq Fit Type:    {conic_type_normal}")
print(f"Normal Eq Parameters:  A={A_n:.6f}, B={B_n:.6f}, C={C_n:.6f}, D={D_n:.6f}, E={E_n:.6f}, F={F_n:.6f}")

print(f"\nQR Fit Type:           {conic_type_qr}")
print(f"QR Parameters:         A={A_q:.6f}, B={B_q:.6f}, C={C_q:.6f}, D={D_q:.6f}, E={E_q:.6f}, F={F_q:.6f}")

# Nonlinear Regression: Levenberg-Marquardt Algorithm
def get_polar_coords(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta = np.where(theta < 0, theta + 2*np.pi, theta)
    return r, theta

def polar_conic_residuals(params, theta, r_obs):
    p, e, theta0 = params
    denom = 1 + e * np.cos(theta - theta0)

    r_pred = p / denom

    return r_pred - r_obs

def polar_conic_jacobian(params, theta):
    p, e, theta0 = params
    denom = 1 + e * np.cos(theta - theta0)
    
    dr_dp = 1.0 / denom
    dr_de = -p * np.cos(theta - theta0) / denom**2
    dr_dtheta0 = -p * e * np.sin(theta - theta0) / denom**2
    
    return np.column_stack((dr_dp, dr_de, dr_dtheta0))

def solve_levenberg_marquardt(theta_obs, r_obs, init_params, max_iter=500, tol=1e-6):
    theta = init_params.copy() 
    mu = 0.01
    nu = 10.0
    
    r_res = polar_conic_residuals(theta, theta_obs, r_obs)
    current_sse = 0.5 * np.sum(r_res**2)
    
    print("\nFitting using Levenberg-Marquardt algorithm...")
    
    for i in range(max_iter):
        r_res = polar_conic_residuals(theta, theta_obs, r_obs)
        J = polar_conic_jacobian(theta, theta_obs)
        
        gradient_sse = np.dot(J.T, r_res)
        Hessian_sse  = np.dot(J.T, J)
        
        Hessian_lm = Hessian_sse + mu * np.eye(len(theta))
        
        delta = np.linalg.solve(Hessian_lm, -gradient_sse)
        
        # Stopping condition: check if delta is smaller than tolerance
        # delta = ||theta_new - theta||
        if np.linalg.norm(delta) < tol:
            print(f"Converged at Iteration {i}: Parameter step size {np.linalg.norm(delta):.8f}")
            theta = theta + delta
            break
            
        theta_new = theta + delta
        r_res_new = polar_conic_residuals(theta_new, theta_obs, r_obs)
        new_sse = 0.5 * np.sum(r_res_new**2)
        
        # The damping coefficient, mu, is reduced when the SSE is lower
        if new_sse < current_sse:
            theta = theta_new 
            current_sse = new_sse
            mu /= nu
            print(f"Iteration {i}: SSE = {current_sse:.6f}, mu = {mu:.8f}")
        else:
            mu *= nu
            print(f"Iteration {i}: SSE = {current_sse:.6f}, mu = {mu:.8f}")
                
    else:
        print(f"Reached maximum iterations (Iteration {max_iter}) without convergence.")
                
    return theta

r_obs, theta_obs = get_polar_coords(X_data, Y_data)
init_params      = np.array([10.0, 1.0, 0.0])

params_nonlinear = solve_levenberg_marquardt(theta_obs, r_obs, init_params)
p_fit, e_fit, theta0_fit = params_nonlinear

conic_type_nonlinear = get_conic_type(0, 0, 0, e_fit, False)

print(f"\nPart B: Nonlinear Regression")
print(f"Nonlinear Fit Type:    {conic_type_nonlinear}")
print(f"p (semi-latus rectum): {p_fit:.4f}")
print(f"e (eccentricity):      {e_fit:.4f}")
print(f"theta0 (rotation):     {theta0_fit:.4f} rad ({np.degrees(theta0_fit):.2f} deg)")

# Plotting
def get_cartesian_grid(x_range, y_range, resolution=100):
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    return X, Y

X_grid, Y_grid = get_cartesian_grid((0, 22), (0, 22))

# Figure 1: Normal Equations and QR Decomposition
fig_linear, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Normal Equations 
Z_grid_normal = (A_n * X_grid**2 + B_n * X_grid * Y_grid + C_n * Y_grid**2 + 
                 D_n * X_grid + E_n * Y_grid + F_n)

ax1.scatter(X_data, Y_data, color='blue', label='Observed Data', zorder=5)
ax1.contour(X_grid, Y_grid, Z_grid_normal, levels=[0], linestyles='dashed', colors='black')

ax1.set_xlim(0, 20)
ax1.set_ylim(0, 20)
ax1.set_title(f'(a) Normal Equation\nType: {conic_type_normal}')
ax1.set_xlabel('X (AU)')
ax1.set_ylabel('Y (AU)')
ax1.legend()
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')

eq_text_normal = (f"{A_n:.3f}$x^2$ + {B_n:.3f}$xy$ + {C_n:.3f}$y^2$\n"
              f"{D_n:.3f}$x$ + {E_n:.3f}$y$ + {F_n:.3f} = 0")
ax1.text(0.05, 0.05, eq_text_normal, transform=ax1.transAxes, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'), fontsize=9)

# QR
Z_grid_qr = (A_q * X_grid**2 + B_q * X_grid * Y_grid + C_q * Y_grid**2 + 
             D_q * X_grid + E_q * Y_grid + F_q)

ax2.scatter(X_data, Y_data, color='blue', label='Observed Data', zorder=5)
ax2.contour(X_grid, Y_grid, Z_grid_qr, levels=[0], linestyles='dashed', colors='black')

ax2.set_xlim(0, 20)
ax2.set_ylim(0, 20)
ax2.set_title(f'(b) QR Decomposition\nType: {conic_type_qr}')
ax2.set_xlabel('X (AU)')
ax2.set_ylabel('Y (AU)')
ax2.legend()
ax2.grid(True)
ax2.set_aspect('equal', adjustable='box')

eq_text_qr = (f"{A_q:.3f}$x^2$ + {B_q:.3f}$xy$ + {C_q:.3f}$y^2$\n"
              f"{D_q:.3f}$x$ + {E_q:.3f}$y$ + {F_q:.3f} = 0")
ax2.text(0.05, 0.05, eq_text_qr, transform=ax2.transAxes, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'), fontsize=9)

plt.tight_layout()
plt.show()

# Figure 2: Nonlinear Fit (Levenberg-Marquardt)
fig_nonlinear = plt.figure(figsize=(6, 6))
ax3 = fig_nonlinear.add_subplot(111)

theta = np.linspace(0, 2*np.pi, 500)
r = p_fit / (1 + e_fit * np.cos(theta - theta0_fit))

x_fit_curve = r * np.cos(theta)
y_fit_curve = r * np.sin(theta)

# Remove asymptotes of the hyperbola
mask = np.abs(r) < 30.0 
x_fit_curve = np.where(mask, x_fit_curve, np.nan)
y_fit_curve = np.where(mask, y_fit_curve, np.nan)

ax3.scatter(X_data, Y_data, color='blue', label='Observed Data', zorder=5)
ax3.plot(x_fit_curve, y_fit_curve, color='black', linestyle='dashed', linewidth=1.5)

ax3.set_xlim(0, 20)
ax3.set_ylim(0, 20)

ax3.set_title(f'(c) Levenberg-Marquardt Polar Fit\nType: {conic_type_nonlinear}')
ax3.set_xlabel('X (AU)')
ax3.set_ylabel('Y (AU)')
ax3.legend()
ax3.grid(True)
ax3.set_aspect('equal', adjustable='box') 

eq_text_nonlinear = (f"$r = \\frac{{{p_fit:.2f}}}{{1 + {e_fit:.2f} \\cos(\\theta - {theta0_fit:.2f})}}$")
ax3.text(0.05, 0.05, eq_text_nonlinear, transform=ax3.transAxes, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'), fontsize=11)

plt.tight_layout()
plt.show()