import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import warnings
warnings.filterwarnings('ignore', category= UserWarning)

dynamic_link_count = 8
analysis_succeeded = [[1]*dynamic_link_count for _ in range(3)]
dynamic_link_names = [
  'Безынерционное (пропорциональное)',
  'Апериодическое 1-го порядка',
  'Апериодическое 2-го порядка',
  'Колебательное 2-го порядка',
  'Идеальное интегрирующее',
  'Реальное интегрирующее',
  'Идеальное дифференцирующее',
  'Реальное дифференцирующее',
]
k  = 5
T  = [0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.3]
T1 = [0.0, 0.0, 1.2, 0.8, 0.0, 0.0, 0.0, 0.0]
T2 = [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
denominators = [
  [       0,     0, 1],
  [       0, T [1], 1],
  [T2[2]**2, T1[2], 1],
  [T2[3]**2, T1[3], 1],
  [       0,     1, 0],
  [T [5]   ,     1, 0],
  [       0,     0, 1],
  [       0, T [7], 1],
]
numerators = [
  [0, k],
  [0, k],
  [0, k],
  [0, k],
  [0, k],
  [0, k],
  [k, 0],
  [k, 0],
]
                   #1,2,3,4,5,6,7,8
is_inertial_link = [0,1,1,1,0,1,0,1]
inertial_link_time_constants = [
  [
    time_constant for time_constant in (T[i], T1[i], T2[i])
    if is_inertial_link[i] and time_constant != 0
  ]
  for i in range(dynamic_link_count)
]

def exception_handler(e):
  print(e)
  plt.text(0.5, 0.5, e, ha= 'center', va= 'center')
  plt.axis('off')

for i in range(dynamic_link_count):
  num = numerators[i]
  den = denominators[i]
  tf = ctrl.tf(num, den)
  # 1. Передаточные функции
  tf_lines = str(tf).split('\n')
  tf_frac_lines = [tf_line.lstrip() for tf_line in tf_lines[4:]]
  frac_len = len(tf_frac_lines[1])
  print(f'''
{i+1}. {dynamic_link_names[i]}
Передаточная функция:
       {' '*((frac_len - len(tf_frac_lines[0]))//2)}{tf_frac_lines[0]}
W(s) = {tf_frac_lines[1]}
       {' '*((frac_len - len(tf_frac_lines[2]))//2)}{tf_frac_lines[2]}''')

  # 2. Вывод операторных уравнений
  a2, a1, a0 = den
  b1, b0 = num
  print(f"Операторное уравнение: {a0}*y + {a1}*y' + {a2}*y'' = {b0}*x + {b1}*x'")

  # 3. Анализ характеристик во временной и частотной областях.
  plt.figure(dynamic_link_names[i], figsize= (15, 9))

  ax_sr = plt.subplot(3, 2, 1)
  plt.title('Переходная характеристика')
  try:
    sr = ctrl.step_response(tf, squeeze= True)
    ax_sr.plot(sr.time, sr.outputs, label= 'K  = 5')
    ax_sr.set_xlabel('Время (с)')
    ax_sr.set_ylabel('Амплитуда')
    ax_sr.grid(True)
    ax_sr.legend()
  except Exception as e:
    exception_handler(e)

  plt.subplot(3, 2, 2)
  plt.title('Импульсная характеристика')
  try:
    ir = ctrl.impulse_response(tf, squeeze= True)
    plt.plot(ir.time, ir.outputs)
    plt.xlabel('Время (с)')
    plt.grid(True)
  except Exception as e:
    exception_handler(e)

  ax_frequency_response = plt.subplot(2, 3, 4)
  plt.title('АФХ')
  try:
    ctrl.nyquist(tf, ax= ax_frequency_response)
    plt.grid(True)
  except Exception as e:
    exception_handler(e)

  plt.subplot(2, 3, 5)
  plt.title('ЛАЧХ')
  omega = np.logspace(-3, 3, 1000)
  fr = ctrl.frequency_response(tf, omega, squeeze= True)
  try:
    plt.semilogx(omega, 20 * np.log10(fr.magnitude))
    plt.xlabel('Частота (рад/с)')
    plt.ylabel('Амплитуда (дБ)')
    plt.grid(True)
  except Exception as e:
    exception_handler(e)

  plt.subplot(2, 3, 6)
  plt.title('ЛФЧХ')
  try:
    plt.semilogx(omega, np.degrees(fr.phase))
    plt.xlabel('Частота (рад/с)')
    plt.ylabel('Фаза (°)')
    plt.grid(True)
  except Exception as e:
    exception_handler(e)

  # 4. Частоты сопряжения и среза для инерционных звеньев
  if is_inertial_link[i]:
    mag_db = 20 * np.log10(np.squeeze(fr.magnitude))
    idx_cut = np.where(mag_db <= 0)[0]
    cutoff_freq = omega[idx_cut[0]] if len(idx_cut) > 0 else None
    if cutoff_freq:
      print(f'Частота среза: {cutoff_freq} рад/с')
    break_freqs = [1/p for p in inertial_link_time_constants[i]]
    if len(break_freqs) > 0:
      print(f'Частоты сопряжения: {break_freqs} рад/с')

  # 5. Полюсы и нули
  print(f'Звено: {dynamic_link_names[i]}')
  print(f'Полюсы: {ctrl.poles(tf)}')
  print(f'Нули: {ctrl.zeros(tf)}')

  # 6. Влияние K на переходный процесс
  num2 = [x*2 for x in num]

  tf2 = ctrl.tf(num2, den)
  try:
    sr2 = ctrl.step_response(tf2)
    ax_sr.plot(sr2.time, np.squeeze(sr2.outputs), '--', label="K' = 10")
    ax_sr.set_xlabel('Время (с)')
    ax_sr.set_ylabel('Амплитуда')
    ax_sr.grid(True)
    ax_sr.legend()
  except Exception as e:
    exception_handler(e)

  plt.tight_layout(); plt.show()

# 7. Влияние T на переходный процесс (вариант 3)
var3_idx = 4 - 1
var3_n = 5
var3_denominator = denominators[var3_idx][:] # deep copy
var3_denominator[1] = T1[var3_idx] * var3_n

orig_tf = ctrl.tf(numerators[var3_idx], denominators[var3_idx])
var3_tf = ctrl.tf(numerators[var3_idx], var3_denominator)

sr_orig = ctrl.step_response(orig_tf, squeeze= True)
sr_var3 = ctrl.step_response(var3_tf, squeeze= True)

plt.figure()#figsize= (8, 5))
plt.plot(sr_orig.time, sr_orig.outputs,       label= f'T1  = {T1[var3_idx]}')
plt.plot(sr_var3.time, sr_var3.outputs, '--', label= f"T1' = {T1[var3_idx] * var3_n}")
plt.title(f'Влияние T на переходный процесс')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True); plt.legend(); plt.show()
