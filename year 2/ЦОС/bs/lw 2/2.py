import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 1. Загрузка данных
x1 = np.loadtxt('lab2_data1.txt')

#plt.figure('Зашумлённый сигнал 1')
#plt.plot(x1)
#plt.grid(True)

# 2. Параметры фильтра
w0 = .75                    # частота настройки в рад/отсчёт
a2 = .75                    # коэффициент, определяющий полосу
a1 = -(a2 + 1) * np.cos(w0)
b0 = .5*(1 - a2)
b1 = .0
b2 = .5*(1 - a2)

# 3. Реализация дискретной фильтрации (разностное уравнение)
y1 = np.zeros_like(x1)
for k in range(2, len(x1)):
    y1[k] = b0*x1[k] + b1*x1[k-1] + b2*x1[k-2] - a1*y1[k-1] - a2*y1[k-2]

#plt.figure(f'Отфильтрованный сигнал 1 (w0={w0}, a2={a2})')
#plt.plot(y1)
#plt.grid(True)

# 4. Определение параметров Морзе для первого сигнала
N1 = 7            # цифра, переданная в первом сигнале (определяется по графику)
M = 50            # длительность элементарного такта в отсчётах




# 8: Характеристики фильтра
num = [b0, b1, b2]
den = [1, a1, a2]   # a0 = 1
fs = 1.0            # частота дискретизации

# 8.1-8.3 Передаточная характеристика (АЧХ и ФЧХ)
w, h = signal.freqz(num, den, worN=1024)
freq = w / np.pi   # нормированная частота (0..1)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(freq, 20 * np.log10(abs(h)))
plt.title('АЧХ фильтра')
plt.ylabel('Амплитуда (дБ)')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(freq, np.angle(h, deg=True))
plt.title('ФЧХ фильтра')
plt.xlabel('Нормированная частота (×π рад/отсчёт)')
plt.ylabel('Фаза (градусы)')
plt.grid(True)
plt.tight_layout()

# 8.5 Переходная характеристика (step response)
# Подаём на вход единичный скачок
step_input = np.ones(100)
step_output = signal.lfilter(num, den, step_input)

plt.figure('Переходная характеристика фильтра')
plt.plot(step_output)
plt.grid(True)

# 8.6 Импульсная характеристика (impulse response)
impulse_input = np.zeros(100)
impulse_input[0] = 1.0
impulse_output = signal.lfilter(num, den, impulse_input)

plt.figure('Импульсная характеристика фильтра')
plt.plot(np.arange(len(impulse_output)), impulse_output)
plt.grid(True)
plt.show()

print(f"""N1 = {N1}
a2 = {a2}
M  = {M}""")
plt.show()
