import numpy as np
import matplotlib.pyplot as plt

show_fig = [1,1,1,1,1]
common_fig_size = (14, 8)

font = {'fontsize': 12}
print_precision = 6

array_size = 10000

variant = 5 # ∈ {1,...,8}
norm_cutoff_freq = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.05][variant - 1] # ω_гп = F_analog / F_sampling, [rad/sample = (rad/sec) / (samples/sec)]

time_constant = 1.0 # [s]
angular_cutoff_freq = 1.0 / time_constant # [rad/s]
gamma = 1.0 / np.tan(np.pi * norm_cutoff_freq) # [-]

w_analog = np.logspace(-2, 2, array_size) # Аналоговая угловая частота [rad/s]
#                               1
# Аналоговый прототип H(s) = -------
#                            T*s + 1
# s := j*ω  ==>
#                         1
# ==> H(s) = H(j*ω) = ---------
#                     T*j*ω + 1
H_analog = 1.0 / (time_constant * 1j * w_analog + 1.0) # АФЧХ аналогового фильтра, [-]
A_analog = np.abs(H_analog) # АЧХ аналогового фильтра, [-]

# Билинейное преобразование:
# H(z) = 1 / (T*s + 1)  ∧  s = γ*(1 - z^-1) / (1 + z^-1)  ==>
#
#                           1                           1 + z^-1
# ==> H(z) = ------------------------------- = --------------------------- =
#            T*γ*(1 - z^-1) / (1 + z^-1) + 1   T*γ*(1 - z^-1) + (1 + z^-1)
#
#           1 + z^-1             b0 + b1*z^-1
# = -------------------------- = ------------
#   (1 + T*γ) + (1 - T*γ)*z^-1   a0 + a1*z^-1
b = np.array([1.0, 1.0])
a = np.array([1.0 + time_constant * gamma, 1.0 - time_constant * gamma])
b_norm = b / a[0]
a_norm = a / a[0]
def z(w):
  return np.exp(1j * w)
additional_half_periods = 5
discrete_periods = 0.5 + additional_half_periods/2
w_discrete = np.linspace(0, np.pi*(1 + additional_half_periods), array_size) # [cycles/sample]
H_discrete =   (b_norm[0] + b_norm[1] * z(w_discrete) ** -1)\
             / (a_norm[0] + a_norm[1] * z(w_discrete) ** -1) # АФЧХ цифрового фильтра, [-]
A_discrete = np.abs(H_discrete) # АЧХ цифрового фильтра, [-]
w_discrete_norm = w_discrete / (2 * np.pi) # Нормированная частота ∈ [0..0.5], [rad/sample = (cycles/sample) / (rad/cycle)]

if show_fig[0]:
  plt.figure(figsize=common_fig_size)
  plt.subplot(1, 2, 1)
  plt.semilogx(w_analog, 20 * np.log10(A_analog))
  plt.grid(True)
  plt.title('АЧХ аналогового прототипа', font)
  plt.xlabel('Угловая аналоговая частота, рад/с', font)
  plt.ylabel('Амплитуда, дБ', font)
  
  plt.subplot(1, 2, 2)
  plt.plot(w_discrete_norm, 20 * np.log10(A_discrete))
  plt.grid(True)
  plt.title(f'АЧХ цифрового фильтра (ω_гп={norm_cutoff_freq})', font)
  plt.xlabel('Нормированная цифровая частота, циклов/отсчёт', font)

  plt.tight_layout()
  plt.show()

if show_fig[1]:
  plt.figure(figsize=(common_fig_size[0]/2**.5, common_fig_size[1]/2**.5))

  plt.subplot(1, 1, 1)
  samples_per_period = int(array_size/discrete_periods)
  # Обратное билинейное преобразование
  plt.plot(gamma*np.tan(.5*w_discrete[:samples_per_period//2]), A_discrete[:samples_per_period//2], '.', label='АЧХ цифрового фильтра')
  plt.plot(w_analog, A_analog, label='АЧХ аналогового фильтра-прототипа')
  plt.xlim(w_analog[0], w_analog[-1])
  plt.grid(True)
  plt.ylabel('Амплитуда, дБ', font)
  plt.xlabel('Частота, рад/с', font)
  plt.title('Сравнение АЧХ', font)
  plt.legend()

  plt.tight_layout()
  plt.show()


# Разностное уравнение цифрового БИХ-фильтра 1-го порядка
# y(n) = b0_norm * x(n) + b1_norm * x(n-1) - a1_norm * y(n-1)
print(f"""Разностное уравнение:
y(n) = {b_norm[0]:.{print_precision}}*x(n) + {b_norm[1]:.{print_precision}}*x(n-1) - {a_norm[1]:.{print_precision}}*y(n-1)\n""")

samples_from_signal_start = 21
samples_till_signal_start = 1 
samples_to_model = samples_from_signal_start + samples_till_signal_start
def filter_discrete(in_signal, numerator_poly, denominator_poly, signal_start= samples_till_signal_start):
  out_signal = np.zeros_like(in_signal)
# Начало сигнала на k-ом отсчёте <==> ∀i < k: in_signal[k] == 0
# ‖
# V
# out_signal[k] = (    numerator_poly[0]* in_signal[k  ] 
#                  +   numerator_poly[1]* in_signal[k-1] 
#                  - denominator_poly[1]*out_signal[k-1]
#                 ) / denominator_poly[0]
# ‖
# V
# out_signal[k] = (    numerator_poly[0]*in_signal[k] 
#                  +   numerator_poly[1]*0
#                  - denominator_poly[1]*0
#                 ) / denominator_poly[0]
# ‖
# V
  out_signal[signal_start] = numerator_poly[0] * in_signal[signal_start] / denominator_poly[0]
  for i in range(signal_start + 1, len(in_signal)):
    out_signal[i] = (    numerator_poly[0]* in_signal[i  ] 
                     +   numerator_poly[1]* in_signal[i-1] 
                     - denominator_poly[1]*out_signal[i-1]
                    ) / denominator_poly[0]
  return out_signal

def adjust_axis_to_farthest_of(axes0, axes1):
  x_min0, x_max0 = axes0.get_xlim()
  y_min0, y_max0 = axes0.get_ylim()
  x_min1, x_max1 = axes1.get_xlim()
  y_min1, y_max1 = axes1.get_ylim()
  x_farthest = (min(x_min0, x_min1), max(x_max0, x_max1))
  y_farthest = (min(y_min0, y_min1), max(y_max0, y_max1))
  axes0.set_xlim(x_farthest)
  axes0.set_ylim(y_farthest)
  axes1.set_xlim(x_farthest)
  axes1.set_ylim(y_farthest)

impulse_signal = np.zeros(samples_to_model)
impulse_signal[samples_till_signal_start] = 1.0
impulse_response = filter_discrete(impulse_signal, b_norm, a_norm)
if show_fig[2]:
  plt.figure(figsize=common_fig_size)
  axes_in = plt.subplot(2, 1, 1)
  plt.stem(range(-samples_till_signal_start, samples_from_signal_start), impulse_signal, basefmt=' ')
  plt.grid(True)
  plt.title('Сигнал x(n) = δ(n)', font)
  plt.ylabel('x(n)', font, labelpad= 0)

  axes_out = plt.subplot(2, 1, 2)
  plt.stem(range(-samples_till_signal_start, samples_from_signal_start), impulse_response, basefmt=' ')
  plt.ylim()
  plt.grid(True)
  plt.title('Импульсная характеристика y(n) = h(n)', font)
  plt.ylabel('y(n)', font, labelpad= 0)

  plt.xlabel('Номер отсчёта', font)

  adjust_axis_to_farthest_of(axes_in, axes_out)
  plt.tight_layout()
  plt.show()

step_signal = np.ones(samples_to_model)
step_signal[:samples_till_signal_start] = 0
step_response = filter_discrete(step_signal, b_norm, a_norm)
if show_fig[3]:
  plt.figure(figsize=common_fig_size)
  axes_in = plt.subplot(2, 1, 1)
  plt.stem(range(-samples_till_signal_start, samples_from_signal_start), step_signal, basefmt=' ')
  plt.grid(True)
  plt.title('Сигнал x(n) = u(n)', font)
  plt.ylabel('x(n)', font)

  axes_out = plt.subplot(2, 1, 2)
  plt.stem(range(-samples_till_signal_start, samples_from_signal_start), step_response, basefmt=' ')
  plt.grid(True)
  plt.title('Переходная характеристика y(n)', font)
  plt.ylabel('y(n)', font)

  plt.xlabel('Номер отсчёта', font)

  adjust_axis_to_farthest_of(axes_in, axes_out)
  plt.tight_layout()
  plt.show()


cycles_to_show = 4 # [cycles]

sin_norm_freq0 = 0.125 # [cycles/sample]
sin_norm_freq1 = 0.25  # [cycles/sample]
sin_sum_norm_period = np.lcm(int(1/sin_norm_freq0), int(1/sin_norm_freq1)) # [samples/cycle]
sin_sum_samples = sin_sum_norm_period * cycles_to_show # [samples = (samples/cycle) * cycles]
sin_sum_signal =   np.sin(2 * np.pi * sin_norm_freq0 * np.arange(sin_sum_samples))\
                 + np.sin(2 * np.pi * sin_norm_freq1 * np.arange(sin_sum_samples))
sin_sum_response = filter_discrete(sin_sum_signal, b_norm, a_norm, signal_start=0)
sample_axis = np.arange(sin_sum_samples)

rect_wave_period = 8 # [samples/cycle]
rect_wave_samples = cycles_to_show * rect_wave_period # [samples = (samples/cycle) * cycles]
rect_wave_signal = np.tile( # tiling of periods
                     np.repeat([0., 1.], repeats= rect_wave_period//2), # an array of 1 period
                     reps= cycles_to_show
                   )
rect_wave_response = filter_discrete(rect_wave_signal, b_norm, a_norm, signal_start=0)
input_common_stem_fmt  = {'linefmt':'r-' , 'markerfmt':'r', 'basefmt':' '}
output_common_stem_fmt = {'linefmt':'b--', 'markerfmt':'b', 'basefmt':' '}
if show_fig[4]:
  plt.figure(figsize=common_fig_size)
  
  axes_in = plt.subplot(2, 1, 1)
  plt.stem(sample_axis, sin_sum_signal, **input_common_stem_fmt, label='Вход x(n)')
  plt.stem(sample_axis, sin_sum_response, **output_common_stem_fmt, label='Выход y(n)')
  plt.grid(True)
  plt.legend()
  plt.title('Реакция на сигнал x(n) = sin(2*π*0.125*n) + sin(2*π*0.25*n)', font)
  plt.ylabel('Амплитуда', font)
  
  axes_out = plt.subplot(2, 1, 2)
  plt.stem(sample_axis, rect_wave_signal, **input_common_stem_fmt, label='Вход x(n)')
  plt.stem(sample_axis, rect_wave_response, **output_common_stem_fmt, label='Выход y(n)')
  plt.grid(True)
  plt.legend()
  plt.title('Реакция на прямоугольную волну (меандр)', font)
  plt.ylabel('Амплитуда', font)

  plt.xlabel('Номер отсчёта', font)

  adjust_axis_to_farthest_of(axes_in, axes_out)
  plt.tight_layout()
  plt.show()

# Таблицы характеристик (первые 20 отсчетов)
print('\n' + '=' * 60)
print('Таблица импульсной характеристики h(n) (первые 20 отсчетов)')
print('=' * 60)
print(' n\t h(n)')
for i in range(min(20, samples_from_signal_start)):
  print(f'{i:2d}\t{impulse_response[i]:.6f}')

print('\n' + '=' * 60)
print('Таблица переходной характеристики (первые 20 отсчетов)')
print('=' * 60)
print(' n\t y(n)')
for i in range(min(20, samples_from_signal_start)):
  print(f'{i:2d}\t{step_response[i]:.6f}')

print('\n' + '=' * 60)
print('Таблица входного и выходного сигналов (прямоугольные импульсы, 1 период)')
print('=' * 60)
print(' n\t x(n)\t y(n)')
for i in range(8):
  print(f'{i:2d}\t{rect_wave_signal[i]:.0f}\t{rect_wave_response[i]:.6f}')
