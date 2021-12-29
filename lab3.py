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

def odesys(y, t, fom1, fom2):
    y1, y2, y3, y4 = y
    dydt = [y3, y4, fom1(y1, y2, y3, y4), fom2(y1, y2, y3, y4)]
    return dydt

t = sp.Symbol('t')
Phi1 = sp.Function('phi1')(t)
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

# Уравнения Лагранжа
ur1 = sp.diff(sp.diff(L, w1), t) - sp.diff(L, Phi1)
ur2 = sp.diff(sp.diff(L, w2), t) - sp.diff(L, Phi2)

a11 = ur1.coeff(sp.diff(w1, t), 1)
a12 = ur1.coeff(sp.diff(w2, t), 1)
a21 = ur2.coeff(sp.diff(w1, t), 1)
a22 = ur2.coeff(sp.diff(w2, t), 1)

b1 = -(ur1.coeff(sp.diff(w1,t),0)).coeff(sp.diff(w2,t),0).subs([(sp.diff(Phi1,t), w1), (sp.diff(Phi2,t), w2)])
b2 = -(ur2.coeff(sp.diff(w1,t),0)).coeff(sp.diff(w2,t),0).subs([(sp.diff(Phi1,t), w1), (sp.diff(Phi2,t), w2)])

detA = a11*a22-a12*a21
detA1 = b1*a22-b2*a21
detA2 = a11*b2-b1*a21

Dphi1Dt = detA1 / detA
Dphi2Dt = detA2 / detA

CountOfFrames = 1000

T = np.linspace(0, 10, CountOfFrames)
fom1 = sp.lambdify([Phi1, Phi2, w1, w2], Dphi1Dt, "numpy")
fom2 = sp.lambdify([Phi1, Phi2, w1, w2], Dphi2Dt, "numpy")

# Начальные значения
Phi10 = 0
Phi20 = 0.7
w10 = 10
w20 = 0
y0 = [Phi10, Phi20, w10, w20]

# численное интегрирование
sol = odeint(odesys, y0, T, args=(fom1, fom2))

# построение графиков
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.set(xlim=[-2, 2], ylim=[-2, 2])
phi1 = sol[:, 0]
phi2 = sol[:, 1]

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
Line_phi1 = ax.plot([X_O, X_A[0]], [Y_O, Y_A[0]], color='black')[0]
Line_phi2 = ax.plot([X_O, X_B[0]], [Y_O, Y_B[0]], color='black')[0]
Point_A = ax.plot(X_A[0] + X_O, Y_A[0], color='black', marker='o')[0]
Point_B = ax.plot(X_B[0] + X_O, Y_B[0], color='black', marker='o')[0]
Point_E = ax.plot(X_B[0] * a, Y_B[0] * a, color='black', marker='o')[0]
Point_D = ax.plot(X_A[0] * a, Y_A[0] * a, color='black', marker='o')[0]
Spring = ax.plot([X_B[0] * a, X_A[0] * a], [Y_B[0] * a, Y_A[0] * a], color='black')[0]

# первая обобщённая координата
FirstCoordinate = fig.add_subplot(4, 2, 2)
FirstCoordinate.plot(T, phi1)
FirstCoordinate.set(xlim=[0, 5])
FirstCoordinate.set(ylim=[0, 5])
plt.title('phi1(t) coordinate')
plt.xlabel('t')
plt.ylabel('phi1(t)')

# вторая обобщённая координата
SecondCoordinate = fig.add_subplot(4, 2, 4)
SecondCoordinate.plot(T, phi2)
SecondCoordinate.set(xlim=[0, 5])
SecondCoordinate.set(ylim=[0, 5])
plt.title('phi2(t) coordinate')
plt.xlabel('t')
plt.ylabel('phi2(t)')

# угловая скорость точки A
AngleSpeedOfA = fig.add_subplot(4, 2, 6)
AngleSpeedOfA.plot(T, sol[:,2])
plt.title('Angle spped of point A')
plt.xlabel('t')
plt.ylabel('W')

# угловая скорость точки B
AngleSpeedOfB = fig.add_subplot(4, 2, 8)
AngleSpeedOfB.plot(T, sol[:,3])
plt.title('Angle speed of point B')
plt.xlabel('t')
plt.ylabel('W')

plt.subplots_adjust(wspace=0.3, hspace=0.7)

# анимация
def Anima(i):
    Point_A.set_data(X_A[i], Y_A[i])
    Point_B.set_data(X_B[i], Y_B[i])
    Point_E.set_data(X_B[i] * a, Y_B[i] * a)
    Point_D.set_data(X_A[i] * a, Y_A[i] * a)
    Line_phi1.set_data([X_O, X_A[i]], [Y_O, Y_A[i]])
    Line_phi2.set_data([X_O, X_B[i]], [Y_O, Y_B[i]])
    Spring.set_data([X_B[i] * a, X_A[i] * a], [Y_B[i] * a, Y_A[i] * a])
    return [Point_A, Point_B, Line_phi1, Line_phi2,  Spring]

anim = FuncAnimation(fig, Anima, frames=CountOfFrames, interval=50)
plt.show()