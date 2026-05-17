import numpy as np
import matplotlib.pyplot as plt

# 1. Define signal parameters
U = [3, 6, 0, -4]
T = [3, 5]
fd = 8
Td = 1 / fd

# 2. Discretize the signal
t1 = np.arange(0, T[0], Td)          # from 0 to T1 inclusive
t2 = np.arange(T[0], T[1] + Td, Td)       # from T1 to T2 inclusive

a1 = U[1] - U[0]
b1 = U[0]
u1 = a1*t1 + b1                         # first linear segment

a2 = U[3] - U[2]
b2 = U[2]
u2 = a2*t2 + b2                         # second linear segment

# Concatenate (duplicate at T[0] is kept, as in MATLAB)
discrete_t = np.concatenate((t1, t2))
u = np.concatenate((u1, u2))

# 3. Compute numerical parameters
N = len(discrete_t)                       # number of samples
U0 = np.sum(u)                            # sum of all samples
# Alternating sum: u1 - u2 + u3 - u4 + ...
Upi = sum(u[i] if i%2 else -u[i] for i in range(len(u)))
E = np.sum(u ** 2)                        # signal energy

print(f"N   = {N}")
print(f"U0  = {U0}")
print(f"Upi = {Upi}")
print(f"E   = {E}")

# 4. Reconstruct analog signal using Kotelnikov (Whittaker–Shannon) interpolation
steps_around = 5
start_val = -steps_around * Td + discrete_t[0]
end_val = discrete_t[-1] + steps_around * Td
step_val = Td / 10
n_points = int(np.round((end_val - start_val) / step_val)) + 1
more_discrete_t = np.linspace(start_val, end_val, n_points)

s = np.zeros_like(more_discrete_t)
for i, t_val in enumerate(more_discrete_t):
    for k in range(N):
        s[i] += u[k] * np.sinc((t_val - k * Td) / Td)



jump_time = T[0]

# ---- Reconstructed signal: find min and max near the jump (within ±0.5 sec) ----
window = 0.5  # seconds around jump_time
mask = np.abs(more_discrete_t - jump_time) <= window
s_near_jump = s[mask]
t_near_jump = more_discrete_t[mask]

s_min = np.min(s_near_jump)
s_max = np.max(s_near_jump)
idx_min = np.argmin(s_near_jump)
idx_max = np.argmax(s_near_jump)
t_min = t_near_jump[idx_min]
t_max = t_near_jump[idx_max]

print(f"""
Локальный минимум = {s_min}, при t = {t_min}
Локальный максимум = {s_max}, при t = {t_max}""")

plt.figure('Дискретизированный сигнал, восстановленный аналоговый и его экстремумы')
plt.stem(discrete_t, u, basefmt=" ", linefmt='b-', markerfmt='bo', label='Дискретизированный сигнал')
plt.plot(more_discrete_t, s, 'r', label='Восстановленный аналоговый сигнал')

plt.axline((t_min, 0), (t_min, s_min), c= 'm',ls='--', label = 'Локальный минимум')
plt.axline((0, s_min), (t_min, s_min), c= 'm',ls='--')
plt.axline((t_max, 0), (t_max, s_max), c= 'y',ls='--', label = 'Локальный максимум')
plt.axline((0, s_max), (t_max, s_max), c= 'y',ls='--')

plt.legend(loc='lower left')
plt.grid(True)
plt.show()