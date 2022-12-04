import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse

#1. Создаём систему 
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

#2. SINDy

model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=0.05),
    feature_library=ps.PolynomialLibrary(degree=2),
)

model.fit(x_train, t=dt)

x_sim = model.simulate(x_train[0], time_steps)

model.print()

def plot_dimension(dim, name, x_sim):
    fig = plt.figure(figsize=(9,2))
    ax = fig.gca()
    ax.plot(time_steps, x_train[:, dim])
    ax.plot(time_steps, x_sim[:, dim], "--")
    plt.xlabel("time")
    plt.ylabel(name)
    plt.draw()
    plt.show()

plot_dimension(0, 'x', x_sim)
plot_dimension(1, 'y', x_sim)
plot_dimension(2, 'z', x_sim)

#3. reservoir

radius = 0.6
sparsity = 0.01
input_dim = 3
reservoir_size = 1000
n_steps_prerun = 10
regularization = 1e-2
sequence = x_train

weights_hidden = sparse.random(reservoir_size, reservoir_size, density=sparsity)
eigenvalues, _ = sparse.linalg.eigs(weights_hidden)
weights_hidden = weights_hidden / np.max(np.abs(eigenvalues)) * radius

weights_input = np.zeros((reservoir_size, input_dim))
q = int(reservoir_size / input_dim)
for i in range(0, input_dim):
    weights_input[i * q:(i + 1) * q, i] = 2 * np.random.rand(q) - 1

weights_output = np.zeros((input_dim, reservoir_size))

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

x_sim_reservoir = predict(sequence, 4000)

plot_dimension(0, 'x', x_sim_reservoir)
plot_dimension(1, 'y', x_sim_reservoir)
plot_dimension(2, 'z', x_sim_reservoir)

print (x_sim_reservoir[0])
print (x_sim_reservoir[0][0])
print (x_sim_reservoir[1][0])

print (x_sim_reservoir[:, 0][0])
print (x_sim_reservoir[:, 0][0][0])
print (x_sim_reservoir[:, 0][1][0])

x_sim_reservoir_X = np.arange(0.0, 40.0, dt)

for i in range(4000):
    x_sim_reservoir_X[i] = x_sim_reservoir[:, 0][i][0]

print ("x_sim_reservoir_X")
print (x_sim_reservoir_X)

x_sim_reservoir_Y = np.arange(0.0, 40.0, dt)

for i in range(4000):
    x_sim_reservoir_Y[i] = x_sim_reservoir[:, 1][i][0]

print ("x_sim_reservoir_Y")
print (x_sim_reservoir_Y)

x_sim_reservoir_Z = np.arange(0.0, 40.0, dt)

for i in range(4000):
    x_sim_reservoir_Z[i] = x_sim_reservoir[:, 2][i][0]

print ("x_sim_reservoir_Z")
print (x_sim_reservoir_Z)

print("x_sim")
print(x_sim)
print("x_sim[:, 0]")
print(x_sim[:, 0])
print("x_sim[:, 1]")
print(x_sim[:, 1])
print("x_sim[:, 2]")
print(x_sim[:, 2])

print("x_sim_reservoir")
print(x_sim_reservoir)
print("x_sim[:, 0]")
print(x_sim_reservoir[:, 0])
print("x_sim[:, 1]")
print(x_sim_reservoir[:, 1])
print("x_sim[:, 2]")
print(x_sim_reservoir[:, 2])



#4. comparison

def plot_variance0(x_train, x_sim, time_steps, dim, name):
    variance = x_train[:, dim] - x_sim[:, dim]
    
    fig = plt.figure(figsize=(9,2))
    ax = fig.gca()
    ax.plot(time_steps, variance)
   
    plt.xlabel("time")
    plt.ylabel(name)
    plt.draw()
    plt.show()

    variance = np.abs(variance)
    print("Сумма модулей отклонений ", name, ':', sum(variance))

plot_variance0(x_train, x_sim, time_steps, 0, "SINDy, x")
plot_variance0(x_train, x_sim, time_steps, 1, "SINDy, y")
plot_variance0(x_train, x_sim, time_steps, 2, "SINDy, z")

def plot_variance1(x_train, x_sim_reservoir_dim, dim, time_steps, name):
    variance = x_train[:, dim] - x_sim_reservoir_dim
    
    fig = plt.figure(figsize=(9,2))
    ax = fig.gca()
    ax.plot(time_steps, variance)
   
    plt.xlabel("time")
    plt.ylabel(name)
    plt.draw()
    plt.show()

    variance = np.abs(variance)
    print("Сумма модулей отклонений ", name, ':', sum(variance))

plot_variance1(x_train, x_sim_reservoir_X, 0, time_steps, "reservoir, x")
plot_variance1(x_train, x_sim_reservoir_Y, 1, time_steps, "reservoir, y")
plot_variance1(x_train, x_sim_reservoir_Z, 2, time_steps, "reservoir, z")








