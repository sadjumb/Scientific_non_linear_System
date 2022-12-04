import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse

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

# что тут началось?
#1. какие-то параметры 
radius = 0.6                    #радиус
sparsity = 0.01                 #разреженность
input_dim = 3                   #размерность 
reservoir_size = 1000           #размер резервуара
n_steps_prerun = 10             #n, шаг, предварительный запуск 
regularization = 1e-2           #регуляризация
sequence = x_train              #последовательность 

# 2.
# а здесь что происходит?
# Разбираем:
# СЛУЧАЙНОЕ ЗАДАНИЕ МАТРИЦЫ СКРЫТЫХ ВЕСОВ
weights_hidden = sparse.random(reservoir_size, reservoir_size, density=sparsity) #?
#Создаёт разреженную матрицу заданной формы и плотности со случайно распределенными значениями.
#(m , n , плотность = 0,01)

eigenvalues, _ = sparse.linalg.eigs(weights_hidden) #?
#Найдёт k собственных значений и собственных векторов квадратной матрицы A.

weights_hidden = weights_hidden / np.max(np.abs(eigenvalues)) * radius #интересно 

weights_input = np.zeros((reservoir_size, input_dim)) 
#Сделает нулевой input_dim-мерный массив на reservoir_size значений 
q = int(reservoir_size / input_dim)
for i in range(0, input_dim):
    weights_input[i * q:(i + 1) * q, i] = 2 * np.random.rand(q) - 1 
    #Что это за ':'? Что тут происходит? 

weights_output = np.zeros((input_dim, reservoir_size))

def initialize_hidden(reservoir_size, n_steps_prerun, sequence):
    hidden = np.zeros((reservoir_size, 1))
    for t in range(n_steps_prerun):
        input = sequence[t].reshape(-1, 1) # Не понимаю, что здесь.
        hidden = np.tanh(weights_hidden @ hidden + weights_input @ input) 
        #hyperbolic tangent
        # Что за собаки?
    return hidden

def augment_hidden(hidden):
    h_aug = hidden.copy()
    h_aug[::2] = pow(h_aug[::2], 2.0)
    return h_aug
# тут тоже что-то

hidden = initialize_hidden(reservoir_size, n_steps_prerun, sequence)
hidden_states = []
targets = []

# что это?
for t in range(n_steps_prerun, len(sequence) - 1):
    input = np.reshape(sequence[t], (-1, 1))
    target = np.reshape(sequence[t + 1], (-1, 1))
    hidden = np.tanh(weights_hidden @ hidden + weights_input @ input)
    hidden = augment_hidden(hidden)
    hidden_states.append(hidden)
    targets.append(target)

targets = np.squeeze(np.array(targets)) #?
hidden_states = np.squeeze(np.array(hidden_states)) 

#?
weights_output = (np.linalg.inv(hidden_states.T@hidden_states + regularization * np.eye(reservoir_size)) @ hidden_states.T@targets).T
# np.linalg.inv возвращает обратную матрицу, np.eye единичная матрица

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

x_sim = predict(sequence, 4000)

# дальше графики

plt.figure(figsize=(6, 4))
plt.plot(x_train[:4000, 0], x_train[:4000, 2], label="Ground truth")
plt.plot(x_sim[:, 0], x_sim[:, 2],'--', label="Reservoir computing estimate")
plt.plot(x_train[0, 0], x_train[0, 2], "ko", label="Initial condition", markersize=8)

plt.legend()
plt.show()

def plot_dimension(dim, name):
    fig = plt.figure(figsize=(9,2))
    ax = fig.gca()
    ax.plot(time_steps, x_train[:, dim])
    ax.plot(time_steps, x_sim[:, dim], "--")
    plt.xlabel("time")
    plt.ylabel(name)
    plt.draw()
    plt.show()

plot_dimension(0, 'x')
plot_dimension(1, 'y')
plot_dimension(2, 'z')

# Короче, странная конструкция, надо ещё разбираться.





