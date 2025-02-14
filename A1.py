""" 
STATE ESTIMATION PART 1 
"""

# pre reqs 
"""
 zt = 2xt/vsound 
 ut = xt.. 
 xt = xt0 + ut(t)^2/2 
 ept = N(0,R)
 delt = N(0,Q) 
 X = [xt,xt.] 
 
 326 time stamps with 0.01 interval 
 xt(0)= 0 , var = 1e-4I 
 Q = 0.01 
 var(x) = 0.1^2 , var(x.) = 0.5^2 , R is diag 
 vsound = 3000 km/hr 
 u(t) = ... 
"""
import numpy as np 
import math 
from numpy.random import multivariate_normal  
import plotly.express as px
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
plot_dir = "plots_part_1/"
tol = 0
# Action model 
def u(t):
    if t < 0.25:
        return 400 
    elif t > 3 and t < 3.25:
        return -400 
    else:
        return 0 

""" 
Initializing motion model, kalman filter and parameters 
"""

sig_x_0 = 0.1 
sig_v_0 = 0.5
sig_s_0 = 0.01

sig_x = sig_x_0 
sig_v = sig_v_0
sig_s = sig_s_0

time_step = 0.01
N = 326
v_sound = 3000
time = [(0.01*i-tol) for i in range(0,326)]

mu_0 = np.transpose(np.array([0,0]))
mu_0 = mu_0[:,np.newaxis]
sigma_0 = 1e-4*np.array([[1,0],[0,1]])
 

R = np.array([[sig_x**2,0],[0,sig_v**2]])
Q = np.array([[sig_s**2]])

A_t = np.array([[1,time_step],[0,1]]) 
B_t = np.transpose(np.array([0.5*time_step**2,time_step]))
B_t = B_t[:,np.newaxis]

C_t = np.array([[2/v_sound,0]]) 



def kalman_filter(mu_prev,sigma_prev,u_t,z_t=None,R=R,Q=Q):

    mu_t = A_t@mu_prev + B_t@u_t 
    sigma_t = A_t@sigma_prev@np.transpose(A_t) + R 
    
    K_t = sigma_t@np.transpose(C_t)@np.linalg.inv((C_t@sigma_t@np.transpose(C_t)+Q))
    if z_t is not None:
        mu_t = mu_t + K_t@(z_t-C_t@mu_t)
        # sigma_t = (np.array([[1,0],[0,1]])-K_t@C_t)@sigma_t
        sigma_t = (np.eye(2) - K_t @ C_t) @ sigma_t @ (np.eye(2) - K_t @ C_t).T + K_t @ Q @ K_t.T

    return mu_t,sigma_t,K_t

def get_state_and_measurements_vs_time(R=R,Q=Q):
    state_t = [np.transpose(np.array(multivariate_normal(np.transpose(mu_0)[0],sigma_0,1)))]
    for t in time[1:]:
        noise_sample = np.transpose(np.array(multivariate_normal(np.array([0,0]),R,1)[0]))
        noise_sample = noise_sample[:,np.newaxis]
        state_t.append(A_t@state_t[-1]+B_t@np.array([[u(t)]])+noise_sample)
    z_t = [(C_t@state_t[i]+multivariate_normal([0],Q,1)[0]) for i in range(len(time))]
    return state_t,z_t 

def get_estimates_and_kalman_gain_vs_time(z_t,R=R,Q=Q):
    K_t = []
    mu_t = [mu_0]
    sigma_t = [sigma_0]


    for i,t in enumerate(time[1:]):
        mu,sigma,K = kalman_filter(mu_t[-1],sigma_t[-1],np.array([[u(t)]]),z_t[i],R,Q) 
        K_t.append(K)
        mu_t.append(mu)
        sigma_t.append(sigma) 
    
    return mu_t,sigma_t,K_t 
    
def get_trajectories(mu_t,sigma_t,state_t):
    x_est = [mu[0][0] for mu in mu_t]
    v_est = [mu[1][0] for mu in mu_t]   
    std_dev_xx = [np.sqrt(sigma_t[i][0][0]) for i in range(len(time))]
    std_dev_vv= [np.sqrt(sigma_t[i][1][1]) for i in range(len(time))]
    x_true = [state_t[i][0][0] for i in range(len(time))]
    v_true = [state_t[i][1][0] for i in range(len(time))]

    result = {'x_est':x_est,'v_est':v_est,'std_dev_xx':std_dev_xx,'x_true':x_true,'v_true':v_true,'std_dev_vv':std_dev_vv}
    return result 


    


# Part(a) and Part(b), Plotting the ground truth and estimated position and velocity vs time 
state_t,z_t = get_state_and_measurements_vs_time()
mu_t,sigma_t,K_t = get_estimates_and_kalman_gain_vs_time(z_t) 
simulation = get_trajectories(mu_t,sigma_t,state_t) 

x_est = simulation['x_est']
v_est = simulation['v_est']
std_dev_xx = simulation['std_dev_xx']
std_dev_vv= simulation['std_dev_vv']
x_true = simulation['x_true']
v_true = simulation['v_true']

fig = px.scatter(x=time, y=x_true,title="Ground Truth Position vs Time",labels={'x':'Time (hour)','y':'Positition (km)'})
# fig.show()
fig.write_html(f"{plot_dir}part_ab/ground_truth_position_vs_time.html")
fig.write_image(f"{plot_dir}part_ab/ground_truth_position_vs_time.png")

fig = px.scatter(x=time,y=v_true,title='Ground Truth Velocity vs Time',labels={'x':'Time (hour)','y':'Velocty (km/h)'})
# fig.show()
fig.write_html(f"{plot_dir}part_ab/ground_truth_velocity_vs_time.html")
fig.write_image(f"{plot_dir}part_ab/ground_truth_velocity_vs_time.png")

fig = px.scatter(x=time, y=x_est,title="Estimated Position vs Time",labels={'x':'Time (hour)','y':'Positition (km)'})
# fig.show()
fig.write_html(f"{plot_dir}part_ab/estimated_position_vs_time.html")
fig.write_image(f"{plot_dir}part_ab/estimated_position_vs_time.png")

fig = px.scatter(x=time,y=v_est,title='Estimated Velocity vs Time',labels={'x':'Time (hour)','y':'Velocty (km/h)'})
# fig.show()
fig.write_html(f"{plot_dir}part_ab/estimated_velocity_vs_time.html")
fig.write_image(f"{plot_dir}part_ab/estimated_velocity_vs_time.png")

# Part (c) Jointly plotting the actual and estimated trajectories and velocities vs time 
fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=x_true, mode='lines+markers', name='Ground Truth Position'))
fig.add_trace(go.Scatter(x=time, y=x_est, mode='lines+markers', name='Estimated Position',error_y=dict(
        type='data',
        array=std_dev_xx, 
        color='brown', 
        visible=True
    )))
fig.update_layout(
    title ="Actual and Estimated Trajectory vs Time",
    xaxis_title = "Time (hour)",
    yaxis_title = "Position (km)",
    legend_title = "Legend"
)
# fig.show()
fig.write_html(f"{plot_dir}part_c/actual_and_estimated_trajectory_vs_time.html") 
fig.write_image(f"{plot_dir}part_c/actual_and_estimated_trajectory_vs_time.png")
fig = go.Figure()
fig.add_trace(go.Scatter(x=time,y=v_true,mode='lines+markers',name='Ground Truth Velocity'))
fig.add_trace(go.Scatter(x=time,y=v_est,mode="lines+markers",name="Estimated Velocity",error_y=dict(
    type='data',
    array=std_dev_vv,
    color='brown',
    visible=True
)))
fig.update_layout(
    title="Actual and Estimated Velocity vs Time",
    xaxis_title='Time (hour)',
    yaxis_title="Velocity (km/h)",
    legend_title="Legend"
)
# fig.show()
fig.write_html(f"{plot_dir}part_c/actual_and_estimated_velocity_vs_time.html")
fig.write_image(f"{plot_dir}part_c/actual_and_estimated_velocity_vs_time.png")

# Part (d) Varying noise in the system and observing the effect on trajectories. 
std_x_ranges = [0.0] + [0.01, 0.5, 1.0]
std_v_ranges = [0.0] + [0.05, 0.1, 1.0]
std_s_ranges = [0.01, 0.05, 0.1, 1.0]

def vary_std_dev_subplots(variable, std_dev_ranges, state_param):
    
    n_plots = len(std_dev_ranges)
    fig = make_subplots(
        rows=1, cols=n_plots,
        subplot_titles=[f"std_dev = {std_d}" for std_d in std_dev_ranges]
    )
    
    if variable == 'x':
        legend_other_variable_noise = f"std_dev in v={sig_v_0}, s={sig_s_0}" 
    elif variable == 'v':
        legend_other_variable_noise = f"std_dev in x={sig_x_0}, s={sig_s_0}"
    elif variable == 's':
        legend_other_variable_noise = f"std_dev in x={sig_x_0}, v={sig_v_0}" 
        
    if state_param == "Position":
        yaxis_title = "Position (km)"
    else:
        yaxis_title = "Velocity (km/h)"
        
    # colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta']
    gt_colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta']
    est_colors = ['royalblue', 'tomato', 'limegreen', 'darkorchid', 'darkorange', 'teal', 'deeppink']
    
    for i, std_d in enumerate(std_dev_ranges):
        sig_x = sig_x_0
        sig_v = sig_v_0
        sig_s = sig_s_0
        
        if variable == 'x':
            sig_x = std_d
        elif variable == 'v':
            sig_v = std_d
        elif variable == 's':
            sig_s = std_d
    
        R = np.array([[sig_x**2, 0],[0, sig_v**2]])
        Q = np.array([[sig_s**2]])
        state_t, z_t = get_state_and_measurements_vs_time(R, Q)
        mu_t, sigma_t, K_t = get_estimates_and_kalman_gain_vs_time(z_t, R, Q)
        simulation = get_trajectories(mu_t, sigma_t, state_t)
        
        if state_param == "Position":
            x_true = simulation['x_true']
            x_est  = simulation['x_est']
            error_array = simulation['std_dev_xx']
        else:
            x_true = simulation['v_true']
            x_est  = simulation['v_est']
            error_array = simulation['std_dev_vv']
        
        gt_color = gt_colors[i % len(gt_colors)]
        est_color = est_colors[i % len(est_colors)]
        fig.add_trace(
            go.Scatter(
                x=time,
                y=x_true,
                mode='lines+markers',
                name=f'GT std_dev={std_d}',
                line=dict(color=gt_color)
            ),
            row=1, col=i+1
        )
        fig.add_trace(
            go.Scatter(
                x=time,
                y=x_est,
                mode='lines+markers',
                name=f'Estimated std_dev={std_d}',
                line=dict(color=est_color),
                error_y=dict(
                    type='data',
                    array=error_array,
                    color='brown',
                    visible=True
                )
            ),
            row=1, col=i+1
        )
    
        fig.update_xaxes(title_text="Time (hour)", row=1, col=i+1)
        fig.update_yaxes(title_text=yaxis_title, row=1, col=i+1)
    

    fig.update_layout(
        title=f"Actual and Estimated {state_param} vs Time (varying '{variable}' std_dev)",
        height=500,
        width=400 * n_plots,
        legend_title=f"Legend: Noise in '{variable}'" + f" ,({legend_other_variable_noise})"
    )
    
    # fig.show()
    fig.write_html(f"{plot_dir}part_d/varying_{variable}_std_dev_{state_param.lower()}_vs_time.html")
    fig.write_image(f"{plot_dir}part_d/varying_{variable}_std_dev_{state_param.lower()}_vs_time.png")


vary_std_dev_subplots('x', std_x_ranges, "Position")
vary_std_dev_subplots('v', std_v_ranges, "Position")
vary_std_dev_subplots('s', std_s_ranges, "Position")


vary_std_dev_subplots('x', std_x_ranges, "Velocity")
vary_std_dev_subplots('v', std_v_ranges, "Velocity")
vary_std_dev_subplots('s', std_s_ranges, "Velocity")



# Part (e) Plotting Kalman Gain and its variation with various noise 
def vary_and_plot_kalman_gain(variable,std_dev_ranges,state_param="Position"):
    fig = go.Figure()
    if variable == 'x':
        legend_title = f"Legend, std_dev_v={sig_v_0}, std_dev_s={sig_s_0}"
    elif variable == 'v':
        legend_title = f"Legend, std_dev_x={sig_x_0}, std_dev_s={sig_s_0}"
    elif variable == 's':
        legend_title = f"Legend, std_dev_x={sig_x_0}, std_dev_v={sig_v_0}"
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta']
    def vary_and_graph(state_param):
        for i,std_d in enumerate(std_dev_ranges):
            sig_x = sig_x_0
            sig_v = sig_v_0
            sig_s = sig_s_0
            if variable == 'x':
                sig_x = std_d 
            elif variable == 'v':
                sig_v = std_d 
            elif variable == 's':
                sig_s = std_d 
            R = np.array([[sig_x**2,0],[0,sig_v**2]]) 
            Q = np.array([[sig_s**2]]) 
            state_t,z_t = get_state_and_measurements_vs_time(R,Q)
            mu_t,sigma_t,K_t = get_estimates_and_kalman_gain_vs_time(z_t,R,Q)
            K_position = [K[0][0] for K in K_t]
            K_velocity = [K[1][0] for K in K_t]
            color = colors[i%len(colors)]
            fig.add_trace(go.Scatter(x=time,y=(lambda param:K_position if param=="Position" else K_velocity)(state_param),mode='lines+markers',name=f'Kalman Gain {state_param} sig_{variable}={std_d}',line=dict(color=color)))
        fig.update_layout(
            title=f"Kalman Gain {state_param} vs Time", 
            xaxis_title="Time (hour)",
            yaxis_title=f"Kalman Gain {state_param}",
            legend_title=legend_title 
        )
        # fig.show()
        fig.write_html(f"{plot_dir}part_e/varying_{variable}_std_dev_kalman_gain_{state_param.lower()}_vs_time.html")
        fig.write_image(f"{plot_dir}part_e/varying_{variable}_std_dev_kalman_gain_{state_param.lower()}_vs_time.png")
    vary_and_graph(state_param)

vary_and_plot_kalman_gain('x',std_x_ranges,"Position")
vary_and_plot_kalman_gain('v',std_v_ranges,"Position")
vary_and_plot_kalman_gain('s',std_s_ranges,"Position")

vary_and_plot_kalman_gain('x',std_x_ranges,"Velocity")
vary_and_plot_kalman_gain('v',std_v_ranges,"Velocity")
vary_and_plot_kalman_gain('s',std_s_ranges,"Velocity")


# Part (f), Missing Observations 
state_t,z_t = get_state_and_measurements_vs_time(R,Q)
z_t = [None if 1.5 <= t <= 2.5 else z for t, z in zip(time, z_t)]
mu_t,sigma_t,K_t = get_estimates_and_kalman_gain_vs_time(z_t,R,Q)
simulation = get_trajectories(mu_t,sigma_t,state_t)
x_est = simulation['x_est']
x_true = simulation['x_true']
v_est = simulation['v_est']
v_true = simulation['v_true']
std_dev_xx = simulation['std_dev_xx']
std_dev_vv = simulation['std_dev_vv']

fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=x_true, mode='lines+markers', name='Ground Truth Position'))
fig.add_trace(go.Scatter(x=time, y=x_est, mode='lines+markers', name='Estimated Position',error_y=dict(
        type='data',
        array=std_dev_xx, 
        color='brown', 
        visible=True
    )))
fig.update_layout(
    title ="Actual and Estimated Trajectory vs Time (missing observations at 1.5 <= t <= 2.5)",
    xaxis_title = "Time (hour)",
    yaxis_title = "Position (km)",
    legend_title = "Legend"
)
# fig.show()
fig.write_html(f"{plot_dir}part_f/actual_and_estimated_trajectory_vs_time_with_missing_data.html")
fig.write_image(f"{plot_dir}part_f/actual_and_estimated_trajectory_vs_time_with_missing_data.png")
fig = go.Figure()
fig.add_trace(go.Scatter(x=time,y=v_true,mode='lines+markers',name='Ground Truth Velocity'))
fig.add_trace(go.Scatter(x=time,y=v_est,mode="lines+markers",name="Estimated Velocity",error_y=dict(
    type='data',
    array=std_dev_vv,
    color='brown',
    visible=True
)))
fig.update_layout(
    title="Actual and Estimated Velocity vs Time (missing observations at 1.5 <= t <= 2.5)",
    xaxis_title='Time (hour)',
    yaxis_title="Velocity (km/h)",
    legend_title="Legend"
)
# fig.show()
fig.write_html(f"{plot_dir}part_f/actual_and_estimated_velocity_vs_time_with_missing_data.html")
fig.write_image(f"{plot_dir}part_f/actual_and_estimated_velocity_vs_time_with_missing_data.png")





""" 

STATE ESTIMATION PART 2 

"""

""" 
g = -10 
GPS = [xt,yt,zt]
base station = [D1,D2,D3,D4] from z=10 
IMU = [xt,yt,zt,vxt,vyt,vzt] 
motion noise = N(0,R) ; R is diag(sigx^2,sigy^2,sigz^2,sigvx^2,sigvy^2,sigvz^2) 
sigx=sigy=sigz=0.01
sigvx=sigvy=sigvz=0.1 

measurement noise = N(0,Q) Q = sig^2I 
sigGPS = 0.1 
sigBS = 0.1 
sigIMU = 0.1 

ut is only g in zt direction 

Time in seconds 
N = 130 time steps , delta_time = 0.01 
field is 100 x 64 , 
goal post is 8 x 3 , at (4,50,0);(4,50,3);(-4,50,3);(-4,50,0); 

boal init = [24.0,4.0,0.0,−16.04,36.8,8.61]T , with sigma = 1e-4 I_6 
state Xt = [xt,yt,zt,vxt,vyt,vzt]T 
"""
import numpy as np 
import math
import scipy.stats as stats 
import pandas as pd
from numpy.random import multivariate_normal  
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp  
from plotly.subplots import make_subplots
plot_dir = "plots_part_2/"
tol = 0
# Action model 
g = -10 
def u(t):
    return np.array([g]).reshape(1,1) 
time_step = 0.01 
N = 130 
time = [i*time_step for i in range(N)]

# Part (a) motion model and Part (b) observation model 
sigGPS = 0.1 
sigBS = 0.1 
sigIMU = 0.1  
sigx=sigy=sigz=0.01 
sigvx=sigvy=sigvz=0.1

BS_height =10 
D1_pos = np.transpose(np.array([-32,50,BS_height])).reshape((3,1)) 
D2_pos = np.transpose(np.array([32,50,BS_height])).reshape((3,1))
D3_pos = np.transpose(np.array([32,-50,BS_height])).reshape((3,1))
D4_pos = np.transpose(np.array([-32,-50,BS_height])).reshape((3,1))

R = np.diag([sigx**2,sigy**2,sigz**2,sigvx**2,sigvy**2,sigvz**2])
Q_GPS = sigGPS**2*np.eye(3) 
Q_BS = sigBS**2*np.eye(4) 
Q_IMU = sigIMU**2*np.eye(6) 

mu_0 = np.transpose(np.array([24.0,4.0,0.0,-16.04,36.8,8.61]))
mu_0 = mu_0.reshape((6,1)) 
sigma_0 = 1e-4*np.eye(6) 

A_t = np.array([
    [1,0,0,time_step,0,0],
    [0,1,0,0,time_step,0],
    [0,0,1,0,0,time_step],
    [0,0,0,1,0,0],
    [0,0,0,0,1,0],
    [0,0,0,0,0,1]
    ])
B_t =  np.transpose(np.array([0,0,0.5*(time_step**2),0,0,time_step])) 
B_t = B_t.reshape((6,1))

def initial_state(mu_0,sigma_0):
    state_0 = multivariate_normal(mu_0.reshape(6),sigma_0).reshape((6,1)) 
    return state_0 

def motion_model(mu_prev,u_t,R=R):
    return A_t@mu_prev + B_t@u_t + multivariate_normal(np.zeros(6),R).reshape((6,1))

Ct_GPS = np.array([
    [1,0,0,0,0,0],
    [0,1,0,0,0,0],
    [0,0,1,0,0,0]
])
Ct_IMU = np.eye(6) 
 
def Ht(xt):
    D1 = np.linalg.norm(D1_pos - xt[0:3]).item()
    D2 = np.linalg.norm(D2_pos - xt[0:3]).item()
    D3 = np.linalg.norm(D3_pos - xt[0:3]).item()
    D4 = np.linalg.norm(D4_pos - xt[0:3]).item()

    return np.array([D1,D2,D3,D4]).reshape(4,1) 

def JacHt(xt):
    D1 = np.linalg.norm(D1_pos - xt[0:3]).item()
    D2 = np.linalg.norm(D2_pos - xt[0:3]).item()
    D3 = np.linalg.norm(D3_pos - xt[0:3]).item()
    D4 = np.linalg.norm(D4_pos - xt[0:3]).item()

    J = np.zeros((4,6))

    J[0,0] = (xt[0]-D1_pos[0]).item()/D1
    J[0,1] = (xt[1]-D1_pos[1]).item()/D1
    J[0,2] = (xt[2]-D1_pos[2]).item()/D1
    J[1,0] = (xt[0]-D2_pos[0]).item()/D2
    J[1,1] = (xt[1]-D2_pos[1]).item()/D2
    J[1,2] = (xt[2]-D2_pos[2]).item()/D2
    J[2,0] = (xt[0]-D3_pos[0]).item()/D3
    J[2,1] = (xt[1]-D3_pos[1]).item()/D3
    J[2,2] = (xt[2]-D3_pos[2]).item()/D3
    J[3,0] = (xt[0]-D4_pos[0]).item()/D4
    J[3,1] = (xt[1]-D4_pos[1]).item()/D4
    J[3,2] = (xt[2]-D4_pos[2]).item()/D4
    return J

def extended_kalman_filter(mu_prev,sigma_prev,u_t,z_t=None,R=None,C_t=None,Q=None,extended=False):
    mu_t = A_t@mu_prev + B_t@u_t 
    sigma_t = A_t@sigma_prev@np.transpose(A_t) + R 
    K_t = sigma_t@np.transpose(C_t)@np.linalg.inv((C_t@sigma_t@np.transpose(C_t)+Q))
    if z_t is not None:
        estimate_zt = C_t@mu_t if not extended else Ht(mu_t) 
        mu_t = mu_t + K_t@(z_t-estimate_zt)
        # sigma_t = (np.array([[1,0],[0,1]])-K_t@C_t)@sigma_t
        sigma_t = (np.eye(6) - K_t @ C_t) @ sigma_t @ (np.eye(6) - K_t @ C_t).T + K_t @ Q @ K_t.T
    return mu_t,sigma_t,K_t 

def gen_ground_truth_and_measurements(R=R,Q_GPS=Q_GPS,Q_BS=Q_BS,Q_IMU=Q_IMU):
    state_t = [initial_state(mu_0,sigma_0)]
    for i,t in enumerate(time[1:]):
        state_t.append(motion_model(state_t[i],u(t),R))
    z_t_GPS = [Ct_GPS@state_t[i] + multivariate_normal(np.zeros(3),Q_GPS).reshape((3,1)) for i in range(N)]
    z_t_BS = [Ht(state_t[i]) + multivariate_normal(np.zeros(4),Q_BS).reshape((4,1)) for i in range(N)]
    z_t_IMU = [state_t[i] + multivariate_normal(np.zeros(6),Q_IMU).reshape((6,1)) for i in range(N)]
    return state_t,z_t_GPS,z_t_BS,z_t_IMU 

def get_estimates_vs_time(z_t_GPS,z_t_BS,z_t_IMU,R=R,Q_GPS=Q_GPS,Q_BS=Q_BS,Q_IMU=Q_IMU):
    K_t_GPS = []
    mu_t_GPS = [mu_0]
    sigma_t_GPS = [sigma_0] 
    K_t_BS = []
    mu_t_BS = [mu_0]
    sigma_t_BS = [sigma_0]
    K_t_IMU = []
    mu_t_IMU = [mu_0]
    sigma_t_IMU = [sigma_0]

    for i,t in enumerate(time[1:]):
        mu_GPS,sigma_GPS,K_GPS = extended_kalman_filter(mu_t_GPS[i],sigma_t_GPS[i],u(t),z_t_GPS[i],R=R,C_t=Ct_GPS,Q=Q_GPS)
        mu_t_GPS.append(mu_GPS)
        sigma_t_GPS.append(sigma_GPS)
        K_t_GPS.append(K_GPS)
        mu_BS,sigma_BS,K_BS = extended_kalman_filter(mu_t_BS[i],sigma_t_BS[i],u(t),z_t_BS[i],R=R,C_t=JacHt(mu_t_BS[i]),Q=Q_BS,extended=True)
        mu_t_BS.append(mu_BS)
        sigma_t_BS.append(sigma_BS)
        K_t_BS.append(K_BS)
        mu_IMU,sigma_IMU,K_IMU = extended_kalman_filter(mu_t_IMU[i],sigma_t_IMU[i],u(t),z_t_IMU[i],R=R,C_t=Ct_IMU,Q=Q_IMU)
        mu_t_IMU.append(mu_IMU)
        sigma_t_IMU.append(sigma_IMU)
        K_t_IMU.append(K_IMU)
    estimates  = {'mu_t_GPS':mu_t_GPS,'sigma_t_GPS':sigma_t_GPS,'K_t_GPS':K_t_GPS,'mu_t_BS':mu_t_BS,'sigma_t_BS':sigma_t_BS,'K_t_BS':K_t_BS,'mu_t_IMU':mu_t_IMU,'sigma_t_IMU':sigma_t_IMU,'K_t_IMU':K_t_IMU} 
    return estimates 

def extract_estimates(est,state_t,z_t_GPS,z_t_BS,z_t_IMU):
    simulation = {'mu_t_GPS':[],'sigma_t_GPS':[],'K_t_GPS':[],'mu_t_BS':[],'sigma_t_BS':[],'K_t_BS':[],'mu_t_IMU':[],'sigma_t_IMU':[],'K_t_IMU':[]}
    
    simulation['mu_t_GPS'] = [est['mu_t_GPS'][i].reshape(6) for i in range(N)]
    simulation['sigma_t_GPS'] = [est['sigma_t_GPS'][i].reshape(6,6) for i in range(N)]
    simulation['K_t_GPS'] = [est['K_t_GPS'][i] for i in range(N-1)]
    simulation['mu_t_BS'] = [est['mu_t_BS'][i].reshape(6) for i in range(N)]
    simulation['sigma_t_BS'] = [est['sigma_t_BS'][i].reshape(6,6) for i in range(N)]
    simulation['K_t_BS'] = [est['K_t_BS'][i] for i in range(N-1)]
    simulation['mu_t_IMU'] = [est['mu_t_IMU'][i].reshape(6) for i in range(N)]
    simulation['sigma_t_IMU'] = [est['sigma_t_IMU'][i].reshape(6,6) for i in range(N)]
    simulation['K_t_IMU'] = [est['K_t_IMU'][i] for i in range(N-1)]
    simulation['state_t'] = [state_t[i].reshape(6) for i in range(N)]
    simulation['z_t_GPS'] = [z_t_GPS[i].reshape(3) for i in range(N)]
    simulation['z_t_BS'] = [z_t_BS[i].reshape(4) for i in range(N)]
    simulation['z_t_IMU'] = [z_t_IMU[i].reshape(6) for i in range(N)]
    return simulation

# Part(a) and Part(b) 

corners_of_field = [[-32,50,0],[32,50,0],[32,-50,0],[-32,-50,0]] 
corners_of_field.append(corners_of_field[0]) 
corners_of_goal = [[4,50,0],[4,50,3],[-4,50,3],[-4,50,0]] 

state_t,z_t_GPS,z_t_BS,z_t_IMU = gen_ground_truth_and_measurements(R,Q_GPS,Q_BS,Q_IMU)
estimates = get_estimates_vs_time(z_t_GPS,z_t_BS,z_t_IMU,R,Q_GPS,Q_BS,Q_IMU)
simulation = extract_estimates(estimates,state_t,z_t_GPS,z_t_BS,z_t_IMU)
state_t = simulation['state_t']


fig = go.Figure() 
fig.add_trace(go.Scatter3d(
    x =[corners_of_field[i][0] for i in range(len(corners_of_field))],
    y = [corners_of_field[i][1] for i in range(len(corners_of_field))], 
    z = [corners_of_field[i][2] for i in range(len(corners_of_field))], 
    mode = 'lines',
    name = 'Field Boundary',
    line=dict(color='red')
))
fig.add_trace(go.Scatter3d(
    x=[corners_of_goal[i][0] for i in range(len(corners_of_goal))],
    y=[corners_of_goal[i][1] for i in range(len(corners_of_goal))],
    z=[corners_of_goal[i][2] for i in range(len(corners_of_goal))], 
    mode = 'lines',
    name = 'Goal boundary',      
    line=dict(color='green')
))

fig.add_trace(go.Scatter3d(
    x=[state_t[i][0] for i in range(N)],
    y=[state_t[i][1] for i in range(N)],
    z=[state_t[i][2] for i in range(N)],
    mode='lines+markers',
    line=dict(color='blue', width=2),
    marker=dict(size=1), 
    name='Ground Truth Trajectory'
))

fig.update_layout(
    title = "Ground Truth Trajectory of the Ball", 
    scene=dict(
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1),  
        zaxis=dict(range=[-30, 30]),
        xaxis = dict(range=[-60, 60]),
        yaxis= dict(range=[-60,60]),
    )
)
# fig.show() 
fig.write_html(f"{plot_dir}part_a/ground_truth_trajectory.html") 
fig.write_image(f"{plot_dir}part_a/ground_truth_trajectory.png")

fig.add_trace(go.Scatter3d(
    x = [simulation['mu_t_GPS'][i][0] for i in range(N)],
    y = [simulation['mu_t_GPS'][i][1] for i in range(N)],
    z = [simulation['mu_t_GPS'][i][2] for i in range(N)],
    mode = 'lines+markers',
    line=dict(width=2,color='violet'), 
    marker=dict(size=1),
    name = 'GPS estimated Trajectory'
))
fig.update_layout(
    title = "GPS estimated Trajectory and Ground Truth Trajectory of the Ball",
)
# fig.show() 
fig.write_html(f"{plot_dir}part_a/gps_estimated_trajectory.html")
fig.write_image(f"{plot_dir}part_a/gps_estimated_trajectory.png")
fig.data = fig.data[:-1]

fig.add_trace(go.Scatter3d(
    x = [simulation['mu_t_BS'][i][0] for i in range(N)],
    y = [simulation['mu_t_BS'][i][1] for i in range(N)],
    z = [simulation['mu_t_BS'][i][2] for i in range(N)],
    mode = 'lines+markers',
    line=dict(width=2,color='orange'),
    marker=dict(size=1),
    name = 'BS estimated Trajectory'
))
fig.update_layout(
    title = "BS estimated Trajectory and Ground Truth Trajectory of the Ball",
)
# fig.show()
fig.write_html(f"{plot_dir}part_a/bs_estimated_trajectory.html")
fig.write_image(f"{plot_dir}part_a/bs_estimated_trajectory.png")
fig.data = fig.data[:-1] 

fig.add_trace(go.Scatter3d(
    x = [simulation['mu_t_IMU'][i][0] for i in range(N)],
    y = [simulation['mu_t_IMU'][i][1] for i in range(N)],
    z = [simulation['mu_t_IMU'][i][2] for i in range(N)],
    mode = 'lines+markers',
    line = dict(width=2,color='cyan'),
    marker = dict(size=1), 
    name='IMU estimated Trajectory',
))
fig.update_layout(
    title = "IMU estimated Trajectory and Ground Truth Trajectory of the Ball"
)
# fig.show()
fig.write_html(f"{plot_dir}part_a/imu_estimated_trajectory.html") 
fig.write_image(f"{plot_dir}part_a/imu_estimated_trajectory.png")
fig.add_trace(go.Scatter3d(
    x = [simulation['mu_t_BS'][i][0] for i in range(N)],
    y = [simulation['mu_t_BS'][i][1] for i in range(N)],
    z = [simulation['mu_t_BS'][i][2] for i in range(N)],
    mode = 'lines+markers',
    line=dict(width=2,color='orange'),
    marker=dict(size=1),
    name = 'BS estimated Trajectory'
)) 
fig.add_trace(go.Scatter3d(
    x = [simulation['mu_t_GPS'][i][0] for i in range(N)],
    y = [simulation['mu_t_GPS'][i][1] for i in range(N)],
    z = [simulation['mu_t_GPS'][i][2] for i in range(N)],
    mode = 'lines+markers',
    line=dict(width=2,color='violet'), 
    marker=dict(size=1),
    name = 'GPS estimated Trajectory'
)) 
fig.update_layout(
    title = "Estimated Trajectories and Ground Truth Trajectory of the Ball",
) 
# fig.show()
fig.write_html(f"{plot_dir}part_a/estimated_trajectories.html")
fig.write_image(f"{plot_dir}part_a/estimated_trajectories.png")

# Part(c) Goal detection, Truth, sensor data, estimated trajectory 


def goal_detection(trajectory):
    direction_vectors = [trajectory[i+1]-trajectory[i] for i in range(len(trajectory)-1)]
    direction_vectors.append(direction_vectors[-1]) 
    for i in range(N):
        if direction_vectors[i][1]:
            t = (50 - trajectory[i][1])/direction_vectors[i][1]
            if t<0 or (t>1 and i!=N-1): # check the case last point is before y=50 
                continue 
            x = trajectory[i][0] + t*direction_vectors[i][0]
            z = trajectory[i][2] + t*direction_vectors[i][2]
            if x>=-4 and x<=4 and z>=0 and z<=3:
                return True 
        elif trajectory[i][1] == 50:
            if trajectory[i][0]>=-4 and trajectory[i][0]<=4 and trajectory[i][2]>=0 and trajectory[i][2]<=3:
                return True 
    return False 



def compute_confidence_ellipse(mu_pos, Sigma_pos, confidence=0.95, num_points=1000):
    """
    Computes points on the confidence ellipse for a 2D Gaussian.
    
    mu_pos: 2x1 mean vector [x, z].
    Sigma_pos: 2x2 covariance matrix for x and z.
    confidence: desired confidence level (e.g. 0.95).
    num_points: number of points to sample along the ellipse.
    Returns an (num_points x 2) array of (x, z) points.
    """
    # Chi-square value for 2 degrees of freedom.
    chi2_val = stats.chi2.ppf(confidence, df=2)
    
    # Eigen-decomposition of the 2x2 covariance matrix.
    vals, vecs = np.linalg.eigh(Sigma_pos)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    
    theta = np.linspace(0, 2*np.pi, num_points)
    ellipse = np.zeros((num_points, 2))
    for i in range(num_points):
        # A_t point on the unit circle.
        circle_point = np.array([np.cos(theta[i]), np.sin(theta[i])])
        # Scale and rotate to get a point on the ellipse.
        # (sqrt(vals[i] * chi2_val)) gives the semi-axis lengths.
        ellipse_point = mu_pos.flatten() + vecs @ (np.sqrt(np.diag(vals * chi2_val)) @ circle_point)
        ellipse[i, :] = ellipse_point
    return ellipse 

def check_ellipse_inside_goal(ellipse, goal_bounds):
   
    x_min, x_max = goal_bounds['x_min'], goal_bounds['x_max']
    z_min, z_max = goal_bounds['z_min'], goal_bounds['z_max']
    
    
    inside = np.all((ellipse[:, 0] >= x_min) & (ellipse[:, 0] <= x_max) &
                    (ellipse[:, 1] >= z_min) & (ellipse[:, 1] <= z_max))
    return inside


def goal_detection_hypothesis_test(mu,sigma,goal_bounds={'x_min':-4,'x_max':4,'z_min':0,'z_max':3}):
    direction_vectors = [mu[i+1]-mu[i] for i in range(N-1)]
    direction_vectors.append(direction_vectors[-1]) 
    for i in range(N):
        if direction_vectors[i][1]:
            t = (50 - mu[i][1])/direction_vectors[i][1]
            if t<0 or (t>1 and i!=N-1):
                continue 
            mu_t = mu[i] + t*direction_vectors[i] 
            sigma_t = sigma[i] 
            # if(i!=N-1 and t > 0.5):
            #     # mui+1 +((50-mui+1)/directioni1)*(directioni)
            #     mu_t = mu[i+1] + ((50-mu[i+1][1])/direction_vectors[i][1])*direction_vectors[i]
            #     sigma_t = sigma[i+1]
            mu_t = np.array([mu_t[0],mu_t[2]]).reshape(2,1)
            sigma_t = np.array([[sigma_t[0,0],sigma_t[0,2]],[sigma_t[2,0],sigma_t[2,2]]])
            ellipse = compute_confidence_ellipse(mu_t,sigma_t) 
            inside = check_ellipse_inside_goal(ellipse,goal_bounds)
            if inside:
                return True
        elif mu[i][1] == 50:
            mu_t = np.array([mu[i][0],mu[i][2]]).reshape(2,1)
            sigma_t = np.array([[sigma[i][0,0],sigma[i][0,2]],[sigma[i][2,0],sigma[i][2,2]]])
            ellipse = compute_confidence_ellipse(mu_t,sigma_t)
            inside = check_ellipse_inside_goal(ellipse,goal_bounds)
            if inside:
                return True
    return False
            
            
            
def goal_scoring_simulation(no_of_simulations,R=R,Q_GPS=Q_GPS,Q_BS=Q_BS,Q_IMU=Q_IMU):
    fraction_scored_ground_truth = 0
    fraction_scored_GPS = 0
    fraction_scored_IMU = 0  
    fraction_scored_GPS_estimates = 0
    fraction_scored_BS_estimates = 0
    fraction_scored_IMU_estimates = 0
    # num_GPS_FP = 0 
    # num_GPS_FN = 0 
    # num_IMU_FP = 0
    # num_IMU_FN = 0 
    for i in range(no_of_simulations):
        state_t,z_t_GPS,z_t_BS,z_t_IMU = gen_ground_truth_and_measurements(R,Q_GPS,Q_BS,Q_IMU)
        estimates = get_estimates_vs_time(z_t_GPS,z_t_BS,z_t_IMU,R,Q_GPS,Q_BS,Q_IMU)
        simulation = extract_estimates(estimates,state_t,z_t_GPS,z_t_BS,z_t_IMU)
        goal_ground_truth = goal_detection(simulation['state_t'])
        goal_GPS_measurements = goal_detection(simulation['z_t_GPS'])
        goal_IMU_measurements = goal_detection(simulation['z_t_IMU']) 
        goal_GPS_estimates = goal_detection_hypothesis_test(simulation['mu_t_GPS'],simulation['sigma_t_GPS'])
        goal_BS_estimates = goal_detection_hypothesis_test(simulation['mu_t_BS'],simulation['sigma_t_BS'])
        goal_IMU_estimates = goal_detection_hypothesis_test(simulation['mu_t_IMU'],simulation['sigma_t_IMU'])
        
        # if goal_ground_truth != goal_GPS:
        #     num_GPS_FP += (goal_ground_truth==False)
        #     num_GPS_FN += (goal_ground_truth==True)
        # if goal_ground_truth != goal_IMU:
        #     num_IMU_FP += (goal_ground_truth==False)
        #     num_IMU_FN += (goal_ground_truth==True)
            
        fraction_scored_ground_truth += goal_ground_truth
        fraction_scored_GPS += goal_GPS_measurements 
        fraction_scored_IMU += goal_IMU_measurements
        fraction_scored_GPS_estimates += goal_GPS_estimates
        fraction_scored_BS_estimates += goal_BS_estimates
        fraction_scored_IMU_estimates += goal_IMU_estimates
    
    fraction_scored_ground_truth /= no_of_simulations
    fraction_scored_GPS /= no_of_simulations
    fraction_scored_IMU /= no_of_simulations
    fraction_scored_GPS_estimates /= no_of_simulations
    fraction_scored_BS_estimates /= no_of_simulations
    fraction_scored_IMU_estimates /= no_of_simulations
    # print("GPS FP:",num_GPS_FP)
    # print("GPS FN:",num_GPS_FN)
    # print("IMU FP:",num_IMU_FP)
    # print("IMU FN:",num_IMU_FN)
    goal_simulation = {'fraction_scored_ground_truth':fraction_scored_ground_truth,'fraction_scored_GPS':fraction_scored_GPS,'fraction_scored_IMU':fraction_scored_IMU,
                        'fraction_scored_GPS_estimates':fraction_scored_GPS_estimates,'fraction_scored_BS_estimates':fraction_scored_BS_estimates,'fraction_scored_IMU_estimates':fraction_scored_IMU_estimates}
    return goal_simulation  

no_of_simulations = 10000
goal_simulation = goal_scoring_simulation(no_of_simulations)

print("Fraction of goals scored by ground truth:",goal_simulation['fraction_scored_ground_truth'])
print("Fraction of goals scored by GPS measurements:",goal_simulation['fraction_scored_GPS'])
print("Fraction of goals scored by IMU measurements:",goal_simulation['fraction_scored_IMU'])
print("Fraction of goals scored by GPS estimates:",goal_simulation['fraction_scored_GPS_estimates'])
print("Fraction of goals scored by BS estimates:",goal_simulation['fraction_scored_BS_estimates'])
print("Fraction of goals scored by IMU estimates:",goal_simulation['fraction_scored_IMU_estimates'])


"""
Fraction of goals scored by ground truth: 0.2798
Fraction of goals scored by GPS measurements: 0.2827
Fraction of goals scored by IMU measurements: 0.2822
Fraction of goals scored by GPS estimates: 0.2353
Fraction of goals scored by BS estimates: 0.2092
Fraction of goals scored by IMU estimates: 0.2461
""" 

# Part (d) varying the noise levels 

number_of_simulations = 2000
sig_pos_ranges = [0.0, 0.01, 0.05,1.0,2.0]
sig_vel_ranges = [0.0]+[0.1*2**i for i in range(0,5)]
sig_sensor_ranges = [0.01, 0.05, 0.1, 0.5, 1.0]

def goal_scoring_simulation_varying_noise_GPS(sig_pos_ranges,sig_vel_ranges,sig_sensor_ranges,number_of_simulations,R=R,Q_GPS=Q_GPS,Q_BS=Q_BS,Q_IMU=Q_IMU):
    simulation_table = {'sig_pos':[],'sig_vel':[],'sig_GPS':[],
                        'Ground Truth':[],'GPS measurement':[],'IMU measurement':[],
                        'BS estimates':[],'IMU estimates':[],'GPS estimates':[],
                        'no of simulations':number_of_simulations
                        
                        }
    for sigp in sig_pos_ranges:
        for sigv in sig_vel_ranges: 
            for sigs in sig_sensor_ranges: 
                R = np.diag([sigp**2,sigp**2,sigp**2,sigv**2,sigv**2,sigv**2]) 
                Q_GPS = sigs**2*np.eye(3)
                goal_simulation = goal_scoring_simulation(number_of_simulations,R,Q_GPS,Q_BS,Q_IMU) 
                simulation_table['sig_pos'].append(sigp)
                simulation_table['sig_vel'].append(sigv)
                simulation_table['sig_GPS'].append(sigs)
                simulation_table['Ground Truth'].append(goal_simulation['fraction_scored_ground_truth'])
                simulation_table['GPS measurement'].append(goal_simulation['fraction_scored_GPS'])
                simulation_table['IMU measurement'].append(goal_simulation['fraction_scored_IMU'])
                simulation_table['BS estimates'].append(goal_simulation['fraction_scored_BS_estimates'])
                simulation_table['IMU estimates'].append(goal_simulation['fraction_scored_IMU_estimates'])
                simulation_table['GPS estimates'].append(goal_simulation['fraction_scored_GPS_estimates'])
    return simulation_table
                 
                    
simulation_table = goal_scoring_simulation_varying_noise_GPS(sig_pos_ranges,sig_vel_ranges,sig_sensor_ranges,number_of_simulations,R,Q_GPS,Q_BS,Q_IMU)   
simulation_table_df = pd.DataFrame(simulation_table)
simulation_table_df.to_csv(f"{plot_dir}part_d/simulation_table.csv")       
# print(simulation_table)         


# Part (e) varying position noise and GPS sensor noise, and analysing 2D projection 

sig_pos_ranges = [0.0, 0.01, 0.1, 1.0]
sig_sensor_ranges = [0.01, 0.05, 0.1, 1.0]

def vary_pos_GPS_noise_and_2D_proj(sig_pos_ranges, sig_sensor_ranges, R, Q_GPS, Q_BS, Q_IMU):
    n_rows = len(sig_pos_ranges)
    n_cols = len(sig_sensor_ranges)
    

    subplot_titles = []
    for sigp in sig_pos_ranges:
        for sigs in sig_sensor_ranges:
            subplot_titles.append(f"sig_pos={sigp}, sig_GPS={sigs}")
    

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05, vertical_spacing=0.1
    )
    
    
    legend_added = {
        "Ground Truth": False,
        "GPS Measurements": False,
        "GPS Estimated": False,
        "Uncertainty Ellipse (1-Sigma)": False
    }
    
   
    for i, sigp in enumerate(sig_pos_ranges):
        for j, sigs in enumerate(sig_sensor_ranges):
            current_row = i + 1 
            current_col = j + 1

         
            R_current = np.diag([sigp**2, sigp**2, sigp**2, sigvx**2, sigvy**2, sigvz**2])
            Q_GPS_current = sigs**2 * np.eye(3)
            
         
            state_t, z_t_GPS, z_t_BS, z_t_IMU = gen_ground_truth_and_measurements(R_current, Q_GPS_current, Q_BS, Q_IMU)
            estimates = get_estimates_vs_time(z_t_GPS, z_t_BS, z_t_IMU, R_current, Q_GPS_current, Q_BS, Q_IMU)
            simulation = extract_estimates(estimates, state_t, z_t_GPS, z_t_BS, z_t_IMU)
   
            truth_x = [s[0] for s in simulation['state_t']]
            truth_y = [s[1] for s in simulation['state_t']]
            
            gps_x = [z[0] for z in simulation['z_t_GPS']]
            gps_y = [z[1] for z in simulation['z_t_GPS']]
            
            est_x = [mu[0] for mu in simulation['mu_t_GPS']]
            est_y = [mu[1] for mu in simulation['mu_t_GPS']]
            
           
            showlegend_gt = not legend_added["Ground Truth"]
            if showlegend_gt:
                legend_added["Ground Truth"] = True
            fig.add_trace(
                go.Scatter(
                    x=truth_x,
                    y=truth_y,
                    mode='lines+markers',
                    name="Ground Truth",
                    legendgroup="Ground Truth",
                    showlegend=showlegend_gt,
                    line=dict(color='blue')
                ),
                row=current_row, col=current_col
            )
            
           
            showlegend_meas = not legend_added["GPS Measurements"]
            if showlegend_meas:
                legend_added["GPS Measurements"] = True
            fig.add_trace(
                go.Scatter(
                    x=gps_x,
                    y=gps_y,
                    mode='lines+markers',
                    name="GPS Measurements",
                    legendgroup="GPS Measurements",
                    showlegend=showlegend_meas,
                    marker=dict(color='red', size=4, symbol='circle')
                ),
                row=current_row, col=current_col
            )
            
           
            showlegend_est = not legend_added["GPS Estimated"]
            if showlegend_est:
                legend_added["GPS Estimated"] = True
            fig.add_trace(
                go.Scatter(
                    x=est_x,
                    y=est_y,
                    mode='lines+markers',
                    name="GPS Estimated",
                    legendgroup="GPS Estimated",
                    showlegend=showlegend_est,
                    line=dict(color='green')
                ),
                row=current_row, col=current_col
            )
            
          
            ellipse_legend_added = False
            for idx in range(0,len(simulation['mu_t_GPS']),5):
                mu = simulation['mu_t_GPS'][idx]
                sigma = simulation['sigma_t_GPS'][idx]
                
                # Extract the XY mean and covariance.
                mu_xy = mu[:2]
                sigma_xy = sigma[:2, :2]
                
                # Create a unit circle and scale it according to the eigen-decomposition of sigma_xy.
                t = np.linspace(0, 2*np.pi, 100)
                circle = np.array([np.cos(t), np.sin(t)]) 
                eigenvals, eigenvecs = np.linalg.eigh(sigma_xy)
                ellipse = mu_xy.reshape(2, 1) + eigenvecs @ np.diag(np.sqrt(eigenvals)) @ circle
                
                # Only add a legend entry for the first ellipse trace of this subplot.
                showlegend_ellipse = False
                if not ellipse_legend_added:
                   
                    if not legend_added["Uncertainty Ellipse (1-Sigma)"]:
                        showlegend_ellipse = True
                        legend_added["Uncertainty Ellipse (1-Sigma)"] = True
                    ellipse_legend_added = True
                    
                fig.add_trace(
                    go.Scatter(
                        x=ellipse[0],
                        y=ellipse[1],
                        mode='lines',
                        line=dict(color='black', dash='dash'),
                        name="Uncertainty Ellipse (1-Sigma)",
                        legendgroup="Uncertainty Ellipse (1-Sigma)",
                        showlegend=showlegend_ellipse
                    ),
                    row=current_row, col=current_col
                )
    
    
    fig.update_layout(
        title='XY Projection with Uncertainty Ellipses (1-Sigma) for Varying Noise Parameters (Y-Axis are y_t and X-axis are x_t)',
        height=900,
        width=1200
    )
    fig.for_each_xaxis(lambda axis: axis.update(autorange="reversed"))
    # fig.show()
    fig.write_html(f"{plot_dir}part_e/2D_projection_uncertainty_ellipses.html")
    fig.write_image(f"{plot_dir}part_e/2D_projection_uncertainty_ellipses.png")


vary_pos_GPS_noise_and_2D_proj(sig_pos_ranges, sig_sensor_ranges, R, Q_GPS, Q_BS, Q_IMU)
           

# Part (f) New Simulation with 2D Trajectory and Uncertainty Ellipses 
def part_f():
    time_step = 0.01         
    N = 250   
    time = np.array([i * time_step for i in range(N)])

    mu_0 = np.array([0.0, -50.0, 0.0, 40.0]).reshape((4,1))
    sigma_0 = np.eye(4)

    sig_pos = 0.01
    sig_vel = 0.1
    R_proc = np.diag([sig_pos**2, sig_pos**2, sig_vel**2, sig_vel**2])

    A_t = np.array([
        [1, 0, time_step, 0],
        [0, 1, 0, time_step],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


    B1_pos = np.array([-32, 50]).reshape((2,1))
    B2_pos = np.array([-32, -50]).reshape((2,1))
    B3_pos = np.array([-32, 0]).reshape((2,1))
    B4_pos = np.array([32, 0]).reshape((2,1))

    sig_BS = 0.1
    Q_single = np.array([[sig_BS**2]])
    Q_combined = np.diag([sig_BS**2, sig_BS**2])



    # def initial_state_2d(mu_0, sigma_0):
    #     state_0 = multivariate_normal(mu_0.reshape(4), sigma_0).reshape((4,1))
    #     return state_0

    # def motion_model_2d(mu_prev, R=R_proc):
    #     return A_t @ mu_prev + multivariate_normal(np.zeros(4), R).reshape((4,1))



    def Ht(state, BS):
        d = np.linalg.norm(BS - state[0:2]).item()
        return np.array([d]).reshape((1,1))

    def JacHt(state, BS):
        d = np.linalg.norm(BS - state[0:2]).item()
        if d < 1e-8:
            d = 1e-8
        J = np.zeros((1,4))
        J[0,0] = (state[0,0] - BS[0,0]) / d
        J[0,1] = (state[1,0] - BS[1,0]) / d
        return J

    def Ht_combined(state, BS1, BS2):
        m1 = Ht(state, BS1)
        m2 = Ht(state, BS2)
        return np.concatenate((m1, m2), axis=0)

    def JacHt_combined(state, BS1, BS2):
        J1 = JacHt(state, BS1)
        J2 = JacHt(state, BS2)
        return np.concatenate((J1, J2), axis=0)


    def extended_kalman_filter(mu_prev, sigma_prev, z_t=None, R=R_proc, C_t=None, Q=None, 
                                extended=False, BS=None, BS2=None):
        mu_t = A_t @ mu_prev
        sigma_t= A_t @ sigma_prev @ A_t.T + R

    
        K_t = sigma_t @ C_t.T @ np.linalg.inv(C_t @ sigma_t @ C_t.T + Q)
        
        if z_t is not None:
            if not extended:
                estimate_zt = C_t @ mu_t
            else:
                if BS2 is None:
                    estimate_zt = Ht(mu_t, BS)
                else:
                    estimate_zt = Ht_combined(mu_t, BS, BS2)
            mu_t = mu_t + K_t @ (z_t - estimate_zt)
            sigma_upd = (np.eye(4) - K_t @ C_t) @ sigma_t @ (np.eye(4) - K_t @ C_t).T + K_t @ Q @ K_t.T
        else:
            mu_t = mu_t
            sigma_upd = sigma_t
            
        return mu_t, sigma_upd, K_t


    def gen_ground_truth(mu_0, A_t, R_proc, N):
        state = mu_0.copy()
        state_t = [state]
        for i in range(1, N):
            noise = multivariate_normal(np.zeros(4), R_proc).reshape((4,1))
            state = A_t @ state + noise
            state_t.append(state)
        return state_t


    def gen_measurements(state_t, Q, BS=None, combined=False, BS2=None):
        if not combined:
            z_t = [Ht(state, BS) + multivariate_normal(np.zeros(1), Q).reshape((1,1))
                for state in state_t]
        else:
            z_t = [Ht_combined(state, BS, BS2) + multivariate_normal(np.zeros(2), Q).reshape((2,1))
                for state in state_t]
        return z_t


    def get_estimates_vs_time(z_t, R=R_proc, Q=Q_single, extended=True, BS=None, combined=False, BS2=None):
        K_t = []
        mu_t = [mu_0]
        simga_t = [sigma_0]
        for i in range(1, N):
            if not combined:
                C_t_local = JacHt(mu_t[i-1], BS)
                mu_upd, sigma_upd, K = extended_kalman_filter(mu_t[i-1], simga_t[i-1],
                                                            z_t[i], R, C_t_local, Q,
                                                            extended=extended, BS=BS)
            else:
                C_t_local = JacHt_combined(mu_t[i-1], BS, BS2)
                mu_upd, sigma_t_upd, K = extended_kalman_filter(mu_t[i-1], simga_t[i-1],
                                                                z_t[i], R, C_t_local, Q,
                                                                extended=extended, BS=BS, BS2=BS2)
                sigma_upd = sigma_t_upd  
            mu_t.append(mu_upd)
            simga_t.append(sigma_upd)
            K_t.append(K)
        return mu_t, simga_t, K_t

    def extract_estimates(est, state_t, z_t):
        simulation = {'mu_t': [], 'sigma_t': [], 'K_t': [], 'state_t': [], 'z_t': []}
        mu_t, simga_t, K_t = est
        simulation['mu_t'] = [mu_t[i].reshape(4) for i in range(len(mu_t))]
        simulation['sigma_t'] = [simga_t[i].reshape((4,4)) for i in range(len(simga_t))]
        simulation['K_t'] = [K_t[i] for i in range(len(K_t))]
        simulation['state_t'] = [state_t[i].reshape(4) for i in range(len(state_t))]
        simulation['z_t'] = [z_t[i].reshape(-1) for i in range(len(z_t))]
        return simulation


    cases = {
        "B1": {"h_func": lambda state: Ht(state, B1_pos),
            "H_func": lambda state: JacHt(state, B1_pos),
            "Q": Q_single,
            "BS_list": [B1_pos]},
        "B2": {"h_func": lambda state: Ht(state, B2_pos),
            "H_func": lambda state: JacHt(state, B2_pos),
            "Q": Q_single,
            "BS_list": [B2_pos]},
        "B3": {"h_func": lambda state: Ht(state, B3_pos),
            "H_func": lambda state: JacHt(state, B3_pos),
            "Q": Q_single,
            "BS_list": [B3_pos]},
        "B3B4": {"h_func": lambda state: Ht_combined(state, B3_pos, B4_pos),
                "H_func": lambda state: JacHt_combined(state, B3_pos, B4_pos),
                "Q": Q_combined,
                "BS_list": [B3_pos, B4_pos]}
    }


    state_t = gen_ground_truth(mu_0, A_t, R_proc, N)

    results = {}
    for key, meas_data in cases.items():
        if key != "B3B4":
            z_t = gen_measurements(state_t, meas_data["Q"], BS=meas_data["BS_list"][0], combined=False)
            mu_t, simga_t, K_t = get_estimates_vs_time(z_t, R=R_proc, Q=meas_data["Q"],
                                                                    extended=True, BS=meas_data["BS_list"][0],
                                                                    combined=False)
        else:
            z_t = gen_measurements(state_t, meas_data["Q"], BS=meas_data["BS_list"][0], combined=True,
                                    BS2=meas_data["BS_list"][1])
            mu_t, simga_t, K_t = get_estimates_vs_time(z_t, R=R_proc, Q=meas_data["Q"],
                                                                    extended=True, BS=meas_data["BS_list"][0],
                                                                    combined=True, BS2=meas_data["BS_list"][1])
        results[key] = {"estimates": (mu_t, simga_t, K_t), "state_t": state_t, "z_t": z_t}


    def get_ellipse_params(Sigma_pos, nsig=1):
        eigvals, eigvecs = np.linalg.eig(Sigma_pos)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        a = nsig * np.sqrt(eigvals[0])
        b = nsig * np.sqrt(eigvals[1])
        angle = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))
        return a, b, angle

    def ellipse_points(cx, cy, a, b, angle_deg, num_points=100):
        t = np.linspace(0, 2*np.pi, num_points)
        angle_rad = np.deg2rad(angle_deg)
        x = cx + a * np.cos(t) * np.cos(angle_rad) - b * np.sin(t) * np.sin(angle_rad)
        y = cy + a * np.cos(t) * np.sin(angle_rad) + b * np.sin(t) * np.cos(angle_rad)
        return x, y

    for key, meas_data in cases.items():
        est = results[key]["estimates"]
        state_t = results[key]["state_t"]
        z_t = results[key]["z_t"]

        truth_xy = np.array([s[0:2,0] for s in state_t])
        mu_t = est[0]
        est_xy = np.array([s[0:2,0] for s in mu_t])
        
        major_axis = []
        minor_axis = []
        
        fig = sp.make_subplots(rows=1, cols=2,
                            subplot_titles=(f"Case {key}: XY Trajectory with Uncertainty Ellipses",
                                            f"Case {key}: Evolution of Ellipse Axes Lengths"))
        
        fig.add_trace(go.Scatter(x=truth_xy[:,0],
                                y=truth_xy[:,1],
                                mode='lines',
                                name='Truth',
                                line=dict(color='blue')),
                    row=1, col=1)
        
        fig.add_trace(go.Scatter(x=est_xy[:,0],
                                y=est_xy[:,1],
                                mode='lines',
                                name='EKF Estimate',
                                line=dict(color='green', dash='dash')),
                    row=1, col=1)
        
        for idx, BS in enumerate(meas_data["BS_list"]):
            BS_vec = BS.reshape(-1)
            name = "BS" if idx == 0 else ""
            fig.add_trace(go.Scatter(x=[BS_vec[0]],
                                    y=[BS_vec[1]],
                                    mode='markers',
                                    marker=dict(color='red', size=10),
                                    name=name),
                        row=1, col=1)
        
        for i in range(0, N, 20):
            pos = est_xy[i]
            for BS in meas_data["BS_list"]:
                BS_vec = BS.reshape(-1)
                fig.add_trace(go.Scatter(x=[BS_vec[0], pos[0]],
                                        y=[BS_vec[1], pos[1]],
                                        mode='lines',
                                        line=dict(color='black', dash='dot', width=1),
                                        showlegend=False),
                            row=1, col=1)
        
        for i in range(0, N, 10):
            Sigma_pos = est[1][i][0:2,0:2]
            a_len, b_len, angle = get_ellipse_params(Sigma_pos, nsig=1)
            major_axis.append(2 * a_len)
            minor_axis.append(2 * b_len)
            cx, cy = est_xy[i,0], est_xy[i,1]
            t_vals = np.linspace(0, 2*np.pi, 100)
            angle_rad = np.deg2rad(angle)
            x_ellipse = cx + a_len * np.cos(t_vals) * np.cos(angle_rad) - b_len * np.sin(t_vals) * np.sin(angle_rad)
            y_ellipse = cy + a_len * np.cos(t_vals) * np.sin(angle_rad) + b_len * np.sin(t_vals) * np.cos(angle_rad)
            fig.add_trace(go.Scatter(x=x_ellipse,
                                    y=y_ellipse,
                                    mode='lines',
                                    line=dict(color='magenta', width=1.5),
                                    showlegend=False),
                        row=1, col=1)
        
        fig.update_xaxes(title_text="X", row=1, col=1)
        fig.update_yaxes(title_text="Y", row=1, col=1)
        
        time_axis = time[:len(major_axis)]
        fig.add_trace(go.Scatter(x=time_axis,
                                y=major_axis,
                                mode='lines',
                                name='Major Axis',
                                line=dict(color='red')),
                    row=1, col=2)
        fig.add_trace(go.Scatter(x=time_axis,
                                y=minor_axis,
                                mode='lines',
                                name='Minor Axis',
                                line=dict(color='blue')),
                    row=1, col=2)
        
        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Axis length (2-sigma)", row=1, col=2)
        
        fig.update_layout(title_text=f"Case {key}",
                        legend=dict(x=0.02, y=0.98),
                        width=1000, height=500)
        
        # fig.show()
        fig.write_html(f"{plot_dir}part_f/case_{key}.html")
        fig.write_image(f"{plot_dir}part_f/case_{key}.png")
        
part_f()
