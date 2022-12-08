import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse

def sum_variance_sindy(x_train, x_sim, time_steps, dim):
    variance = x_train[:, dim] - x_sim[:, dim]
    variance = np.abs(variance)
    return sum(variance)

def sum_variance_reservoir(x_train, x_sim_reservoir_dim, time_steps, dim):
    variance = x_train[:, dim] - x_sim_reservoir_dim
    variance = np.abs(variance)
    return sum(variance)

#1.
rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0
dt = 0.01

def f(state, t):
    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

state0 = [1.0, 1.0, 1.0]
time_steps = np.arange(0.0, 40.0, dt)

x_train = odeint(f, state0, time_steps)

#2.
radius = 0.6
sparsity = 0.01
input_dim = 3
reservoir_size = 1000
n_steps_prerun = 10
regularization = 1e-2
sequence = x_train

#3
def initialize_hidden(reservoir_size, n_steps_prerun, sequence):
    hidden = np.zeros((reservoir_size, 1))
    for t in range(n_steps_prerun):
        input = sequence[t].reshape(-1, 1)
        hidden = np.tanh(weights_hidden @ hidden + weights_input @ input)
    return hidden

def augment_hidden(hidden):
    h_aug = hidden.copy()
    h_aug[::2] = pow(h_aug[::2], 2.0)
    return h_aug

def predict(sequence, n_steps_predict):
    hidden = initialize_hidden(reservoir_size, n_steps_prerun, sequence)
    input = sequence[n_steps_prerun].reshape((-1, 1))
    outputs = []

    for t in range(n_steps_prerun, n_steps_prerun + n_steps_predict):
        hidden = np.tanh(weights_hidden @ hidden + weights_input @ input)
        hidden = augment_hidden(hidden)
        output = weights_output @ hidden
        input = output
        outputs.append(output)
    return np.array(outputs)

#4
n = 10
differences_all_s = [0]*n
differences_all_r = [0]*n

differences_x_s = [0]*n
differences_y_s = [0]*n
differences_z_s = [0]*n

differences_x_r = [0]*n
differences_y_r = [0]*n
differences_z_r = [0]*n

times = [0]*n

SINDy_better_all = 0
reservoir_better_all = 0
identically_all = 0

SINDy_better_x = 0
reservoir_better_x = 0
identically_x = 0

SINDy_better_y = 0
reservoir_better_y = 0
identically_y = 0

SINDy_better_z = 0
reservoir_better_z = 0
identically_z = 0

for i in range(n):
    #SINDy
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=0.05),
        feature_library=ps.PolynomialLibrary(degree=2),
    )

    model.fit(x_train, t=dt)

    x_sim = model.simulate(x_train[0], time_steps)

    print("По мнению sindy:")
    model.print()

    #reservoir
    weights_hidden = sparse.random(reservoir_size, reservoir_size, density=sparsity)
    eigenvalues, _ = sparse.linalg.eigs(weights_hidden)
    weights_hidden = weights_hidden / np.max(np.abs(eigenvalues)) * radius

    weights_input = np.zeros((reservoir_size, input_dim))
    q = int(reservoir_size / input_dim)
    for j in range(0, input_dim):
        weights_input[j * q:(j + 1) * q, j] = 2 * np.random.rand(q) - 1

    weights_output = np.zeros((input_dim, reservoir_size))

    hidden = initialize_hidden(reservoir_size, n_steps_prerun, sequence)
    hidden_states = []
    targets = []

    for t in range(n_steps_prerun, len(sequence) - 1):
        input = np.reshape(sequence[t], (-1, 1))
        target = np.reshape(sequence[t + 1], (-1, 1))
        hidden = np.tanh(weights_hidden @ hidden + weights_input @ input)
        hidden = augment_hidden(hidden)
        hidden_states.append(hidden)
        targets.append(target)

    targets = np.squeeze(np.array(targets))
    hidden_states = np.squeeze(np.array(hidden_states))

    weights_output = (np.linalg.inv(hidden_states.T@hidden_states + regularization * np.eye(reservoir_size)) @ hidden_states.T@targets).T
    
    x_sim_reservoir = predict(sequence, 4000)

    #estimation
    x_sim_reservoir_X = np.arange(0.0, 40.0, dt)
    for k in range(4000):
        x_sim_reservoir_X[k] = x_sim_reservoir[:, 0][k][0]

    x_sim_reservoir_Y = np.arange(0.0, 40.0, dt)
    for s in range(4000):
        x_sim_reservoir_Y[s] = x_sim_reservoir[:, 1][s][0]

    x_sim_reservoir_Z = np.arange(0.0, 40.0, dt)
    for c in range(4000):
        x_sim_reservoir_Z[c] = x_sim_reservoir[:, 2][c][0]    
    
    xs = sum_variance_sindy(x_train, x_sim, time_steps, 0)
    ys = sum_variance_sindy(x_train, x_sim, time_steps, 1)
    zs = sum_variance_sindy(x_train, x_sim, time_steps, 2)

    xr = sum_variance_reservoir(x_train, x_sim_reservoir_X, time_steps, 0)
    yr = sum_variance_reservoir(x_train, x_sim_reservoir_Y, time_steps, 1)
    zr = sum_variance_reservoir(x_train, x_sim_reservoir_Z, time_steps, 2)

    differences_all_s[i] = xs + ys + zs
    differences_all_r[i] = xr + yr + zr
    
    differences_x_s[i] = xs
    differences_y_s[i] = ys
    differences_z_s[i] = zs
    
    differences_x_r[i] = xr
    differences_y_r[i] = yr
    differences_z_r[i] = zr
    
    times[i] = i

    print()
    print("Тест ", i)
    print("Сумма модулей отклонений по X")
    print("SINDy: ", xs)
    print("reservoir: ", xr)
    print()
    print("Сумма модулей отклонений по Y")
    print("SINDy: ", ys)
    print("reservoir: ", yr)
    print()
    print("Сумма модулей отклонений по Z")
    print("SINDy: ", zs)
    print("reservoir: ", zr)
    print()
    print("Сумма модулей отклонений по XYZ")
    print("SINDy: ", xs + ys + zs)
    print("reservoir: ", xr + yr + zr)
    print()

    if (xs + ys + zs < xr + yr + zr):
        SINDy_better_all = SINDy_better_all + 1
    elif (xs + ys + zs > xr + yr + zr):
        reservoir_better_all = reservoir_better_all + 1
    else:
        identically_all = identically_all + 1

    if (xs < xr):
        SINDy_better_x = SINDy_better_x + 1
    elif (xs > xr):
        reservoir_better_x = reservoir_better_x + 1
    else:
        identically_x = identically_x + 1

    if (ys < yr):
        SINDy_better_y = SINDy_better_y + 1
    elif (ys > yr):
        reservoir_better_y = reservoir_better_y + 1
    else:
        identically_y = identically_y + 1

    if (zs < zr):
        SINDy_better_z = SINDy_better_z + 1
    elif (zs > zr):
        reservoir_better_z = reservoir_better_z + 1
    else:
        identically_z = identically_z + 1  
    
#Строим графики   
fig = plt.figure(figsize=(9,2))
ax = fig.gca()
ax.plot(times, differences_all_s, label = "SINDy", color = 'blue')
ax.plot(times, differences_all_r, label = "reservoir", color = 'red')
plt.title("Cумма модулей отклонений по XYZ")
plt.xlabel("разы")
plt.ylabel("сумма модулей отклонений")
plt.draw()
plt.legend()
plt.show()

fig = plt.figure(figsize=(9,2))
ax = fig.gca()
ax.plot(times, differences_x_s, label = "SINDy", color = 'blue')
ax.plot(times, differences_x_r, label = "reservoir", color = 'red')
plt.title("Cумма модулей отклонений по X")
plt.xlabel("разы")
plt.ylabel("сумма модулей отклонений")
plt.draw()
plt.legend()
plt.show()

fig = plt.figure(figsize=(9,2))
ax = fig.gca()
ax.plot(times, differences_y_s, label = "SINDy", color = 'blue')
ax.plot(times, differences_y_r, label = "reservoir", color = 'red')
plt.title("Cумма модулей отклонений по Y")
plt.xlabel("разы")
plt.ylabel("сумма модулей отклонений")
plt.draw()
plt.legend()
plt.show()

fig = plt.figure(figsize=(9,2))
ax = fig.gca()
ax.plot(times, differences_z_s, label = "SINDy", color = 'blue')
ax.plot(times, differences_z_r, label = "reservoir", color = 'red')
plt.title("Cумма модулей отклонений по Z")
plt.xlabel("разы")
plt.ylabel("сумма модулей отклонений")
plt.draw()
plt.legend()
plt.show()

#Статистика
print()
print("SINDy лучше по XYZ: ", SINDy_better_all, " из ", n)
print("reservoir лучше по XYZ: ", reservoir_better_all, " из ", n)
print("одинаково по XYZ: ", identically_all, " из ", n)
print()
print("SINDy лучше по X: ", SINDy_better_x, " из ", n)
print("reservoir лучше по X: ", reservoir_better_x, " из ", n)
print("одинаково по X: ", identically_x, " из ", n)
print()
print("SINDy лучше по Y: ", SINDy_better_y, " из ", n)
print("reservoir лучше по Y: ", reservoir_better_y, " из ", n)
print("одинаково по Y: ", identically_y, " из ", n)
print()
print("SINDy лучше по Z: ", SINDy_better_z, " из ", n + 1)
print("reservoir лучше по Z: ", reservoir_better_z, " из ", n)
print("одинаково по Z: ", identically_z, " из ", n)










    
    
