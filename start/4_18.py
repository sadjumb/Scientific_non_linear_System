import numpy as np
import pysindy as ps #SINDy

t = np.linspace(0, 1, 100) #100 значений от 0 до 1
x = 3 * np.exp(-2 * t)
y = 0.5 * np.exp(t)
X = np.stack((x, y), axis=-1) #?странный метод

model = ps.SINDy(
    #Понять, что за класс
    differentiation_method=ps.FiniteDifference(order=2), #что за метод
    feature_library=ps.FourierLibrary(),
    optimizer=ps.STLSQ(threshold=0.2), #что за метод
    feature_names=["x", "y"]
)
model.fit(X, t=t) #обучение сети, не нашёл нормального описания

model.print()

# Здесь и далее до примера с Л-ом мы угадываем СДУ по ряду x и y от t, полученному из решений СДУ и начальных условий (t = 0)
