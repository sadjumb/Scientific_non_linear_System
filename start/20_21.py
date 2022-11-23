import numpy as np
import pysindy as ps #SINDy
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 100) #100 значений от 0 до 1
x = 3 * np.exp(-2 * t)
y = 0.5 * np.exp(t)
X = np.stack((x, y), axis=-1) #?

model = ps.SINDy(
    differentiation_method=ps.FiniteDifference(order=2), #?
    feature_library=ps.FourierLibrary(),
    optimizer=ps.STLSQ(threshold=0.2), #?
    feature_names=["x", "y"]
)

model.fit(X, t=t)

def plot_simulation(model, x0, y0):
    t_test = np.linspace(0, 1, 100)
    x_test = x0 * np.exp(-2 * t_test)
    y_test = y0 * np.exp(t_test)

    sim = model.simulate([x0, y0], t=t_test)
    #Что за метод?

    plt.figure(figsize=(6, 4))
    plt.plot(x_test, y_test, label="Ground truth", linewidth=4)
    plt.plot(sim[:, 0], sim[:, 1], "--", label="SINDy estimate", linewidth=3)
    plt.plot(x0, y0, "ko", label="Initial condition", markersize=8)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

x0 = 6
y0 = -0.1

plot_simulation(model, x0,y0)
