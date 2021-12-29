import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math
from scipy.integrate import odeint

# Постоянные параметры
OA = 1
OB = 1
a = 0.4
l1 = OA
l2 = OB
m1 = 1
m2 = 0.6
g = 9.81

def odesys(y, t, fom2):
    y1,y2 = y
    dydt = [y2, fom2(y1, y2)]
    return dydt

t = sp.Symbol('t')

# Условие, что phi1(t) тождественно равен нулю
Phi1 = 0

Phi2 = sp.Function('phi2')(t)

# Момент инерции стержня OA
M1 = m1 * (l1 ** 2) / 3

# Момент инерции стержня OB
M2 = m2 * (l2 ** 2) / 3

# Угловая скорость OA
w1 = sp.diff(Phi1, t)

# Угловая скорость OB
w2 = sp.diff(Phi2, t)

# Вращательная энергия OA
W1 = (M1 * (w1 ** 2)) / 2

# Вращательная энергия OB
W2 = (M2 * (w2 ** 2)) / 2

# Потенциальная энергия стержней
E1 = m1 * g * l1 / 2
E2 = m2 * g * l2 / 2

l0 = 0.3 # длина недеформированной пружины
c = 200 # коэффициент жёсткости

# угол между стержнями
phi0 = Phi2 - Phi1

# длина дефоромированной пружины
l = (2 * (a ** 2) * (1 - sp.cos(phi0))) ** 0.5

# растяжение пружины
delta_l = l - l0

# Потенциальная энергия-энергия деформированной пружины
Wpr = c * (delta_l ** 2) / 2

Wk = W1 + W2
Wp = E1 + E2 + Wpr

# Функция Лагранжа
L = Wk - Wp

# Уравнение Лагранжа для второго стержня
ur2 = sp.diff(sp.diff(L, w2), t) - sp.diff(L, Phi2)

a22 = ur2.coeff(sp.diff(w2, t), 1)
b2 = -ur2.coeff(sp.diff(w2,t),0).subs(sp.diff(Phi1,t), w2);

Dphi2Dt = b2 / a22

CountOfFrames = 1000

T = np.linspace(0, 10, CountOfFrames)
fom2 = sp.lambdify([Phi2, w2], Dphi2Dt, "numpy")

# Начальные значения
Phi20 = 0.7
w20 = 0
y0 = [Phi20, w20]

# численное интегрирование
sol = odeint(odesys, y0, T, args=(fom2,))

# построение графиков
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.set(xlim=[-2, 2], ylim=[-2, 2])


phi1 = 0
phi2 = sol[:, 0]

# Координаты точки A
X_A = l1 * np.cos(phi1)
Y_A = l1 * np.sin(phi1)

# Координаты точки B
X_B = l2 * np.cos(phi2)
Y_B = l2 * np.sin(phi2)

# Точка вращения стержней
X_O = 0
Y_O = 0

# Создание точки вращения, точек концов стержней и точек, находящихся на расстоянии a от оси вращения и соединённых пружиной
Point_O = ax.plot(X_O, Y_O, color='black', marker='o')[0]
Line_phi1 = ax.plot([X_O, X_A], [Y_O, Y_A], color='black')[0]
Line_phi2 = ax.plot([X_O, X_B[0]], [Y_O, Y_B[0]], color='black')[0]
Point_A = ax.plot(X_A + X_O, Y_A, color='black', marker='o')[0]
Point_B = ax.plot(X_B + X_O, Y_B, color='black', marker='o')[0]
Point_E = ax.plot(X_B[0] * a, Y_B[0] * a, color='black', marker='o')[0]
Point_D = ax.plot(X_A * a, Y_A * a, color='black', marker='o')[0]
Spring = ax.plot([X_B[0] * a, X_A * a], [Y_B[0] * a, Y_A * a], color='black')[0]

# вторая обобщённая координата
SecondCoordinate = fig.add_subplot(4, 2, 2)
SecondCoordinate.plot(T, phi2)
SecondCoordinate.set(xlim=[0, 5])
SecondCoordinate.set(ylim=[0, 5])
plt.title('phi2(t) coordinate')
plt.xlabel('t')
plt.ylabel('phi2(t)')

# угловая скорость точки B
AngleSpeedOfB = fig.add_subplot(4, 2, 4)
AngleSpeedOfB.plot(T, sol[:,1])
plt.title('Angle speed of point B')
plt.xlabel('t')
plt.ylabel('W')

plt.subplots_adjust(wspace=0.3, hspace=0.7)

# анимация
def Anima(i):
    Point_B.set_data(X_B[i], Y_B[i])
    Point_E.set_data(X_B[i] * a, Y_B[i] * a)
    Line_phi2.set_data([X_O, X_B[i]], [Y_O, Y_B[i]])
    Spring.set_data([X_B[i] * a, X_A * a], [Y_B[i] * a, Y_A * a])
    return [Point_B, Point_E, Line_phi2,  Spring]

anim = FuncAnimation(fig, Anima, frames=CountOfFrames, interval=10)
plt.show()