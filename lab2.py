import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math

# обобщённые координаты - phi1(t) и phi2(t)

Steps = 1001
t = np.linspace(0, 10, Steps)
x = np.sin(t)

phi1 = np.sin(t)
phi2 = np.cos(t + 1)

X_A = 1.5 * np.cos(phi1)
Y_A = 1.5 * np.sin(phi1)

X_B = 1 * np.sin(phi2 / 2)
Y_B = 1 * np.cos(phi2 / 2)

X_O = 0
Y_O = 0

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.set(xlim=[-2, 2], ylim=[-2, 2])
ax.axis('equal')

Point_O = ax.plot(X_O, Y_O, color='r', marker='o')[0]
Line_phi1 = ax.plot([X_O, X_A[0]], [Y_O, Y_A[0]], color='r')[0]
Line_phi2 = ax.plot([X_O, X_B[0]], [Y_O, Y_B[0]], color='r')[0]
Point_A = ax.plot(X_A[0] + X_O, Y_A[0], color='r', marker='o')[0]  # plotting point A
Point_B = ax.plot(X_B[0] + X_O, Y_B[0], color='r', marker='o')[0]  # plotting point B
Point_C = ax.plot(X_B[0] * (2 / 3), Y_B[0] * (2 / 3), color='r', marker='o')[0]
Point_D = ax.plot(X_A[0] * (1 / 2), Y_A[0] * (1 / 2), color='r', marker='o')[0]
Spring_len = ax.plot([X_B[0] * (2 / 3), X_A[0] * (1 / 2)], [Y_B[0] * (2 / 3), Y_A[0] * (1 / 2)], color='r')[0]
L_Spring_len = math.sqrt((X_B[0] * (2 / 3) - X_A[0] * (1 / 2)) ** 2 + (Y_B[0] * (2 / 3) - Y_A[0] * (1 / 2)) ** 2)
# for i in range(8):
#     Point_N = ax.plot(X_B[0] * (2 / 3) + (i + 1) * (L_Spring_len / 9), Y_B[0] * (2 / 3) - (i + 1) * (L_Spring_len / 8), color='r', marker='o')
# Spring = ax.plot(X_Spr*L_Spr[0], Y_Spr)[0]
#Spring = ax.plot([phiA1[0], phiB1[0]], [phiA2 / 2 , phiB2 / 2])[0]

def Kino(i):
    Point_A.set_data(X_A[i], Y_A[i])
    Point_B.set_data(X_B[i], Y_B[i])
    Point_C.set_data(X_B[i]*(2/3), Y_B[i]*(2/3))
    Point_D.set_data(X_A[i]*(1/2), Y_A[i]*(1/2))
    Line_phi1.set_data([X_O, X_A[i]], [Y_O, Y_A[i]])
    Line_phi2.set_data([X_O, X_B[i]], [Y_O, Y_B[i]])
    Spring_len.set_data([X_B[i]*(2/3), X_A[i]*(1/2)], [Y_B[i]*(2/3), Y_A[i]*(1/2)])
    #Spring.set_data([phiA1[i], phiB1[i]], [phiA2 / 2, phiB2 / 2])
    return [Point_A, Point_B, Line_phi1, Line_phi2,  Spring_len]

time = sp.Symbol('t')

phi11 = sp.sin(time)
phi22 = sp.cos(time + 1) / 2

A_speed = 1.5 * sp.diff(phi11, time)
A_acceleration = sp.diff(A_speed, time)

B_speed = sp.diff(phi22, time)
B_acceleration = sp.diff(B_speed, time)

T = np.linspace(0, 2 * math.pi, 1000)
A_Speed = np.zeros_like(T)
A_Acceleration = np.zeros_like(T)
B_Speed = np.zeros_like(T)
B_Acceleration = np.zeros_like(T)

for i in np.arange(len(T)):
    A_Speed[i] = sp.Subs(A_speed, time, T[i])
    A_Acceleration[i] = sp.Subs(A_acceleration, time, T[i])
    B_Speed[i] = sp.Subs(A_speed, time, T[i])
    B_Acceleration[i] = sp.Subs(B_acceleration, time, T[i])

# скорость точки A
ax_av = fig.add_subplot(4, 2, 2)
ax_av.plot(T, A_Speed)
plt.title('speed of the point A')
plt.xlabel('T')
plt.ylabel('v of point A')

# ускорение точки A
ax_aw = fig.add_subplot(4, 2, 4)
ax_aw.plot(T, A_Acceleration)
plt.title('acceleration of the point A')
plt.xlabel('T')
plt.ylabel('w of point A')

# скорость точки B
ax_bv = fig.add_subplot(4, 2, 6)
ax_bv.plot(T, B_Speed)
plt.title('speed of the Point B')
plt.xlabel('T')
plt.ylabel('v of point B')

# ускорение точки B
ax_bw = fig.add_subplot(4, 2, 8)
ax_bw.plot(T, B_Acceleration)
plt.title('acceleration of the Point B')
plt.xlabel('T')
plt.ylabel('w of point B')

plt.subplots_adjust(wspace=0.3, hspace=0.7)

anima = FuncAnimation(fig, Kino, frames=Steps, interval=1)
plt.show()
