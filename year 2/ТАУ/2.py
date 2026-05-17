import numpy as np
from matplotlib import (
  pyplot as plt,
  ticker as tck
)
import control as ctrl
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

#closed_loop: 1
#  open_loop:   1
calc_tf =    [1,1]
#checked:     1
#closed_loop: 1 1 1|
#  open_loop:      |1 1 1 1
#             0 1 2|3 4 5 6
show_fig   = [1,1,1,1,1,1,1]

tolerance = 1e-10
plt_print_precision = 2

degrees_ticks = np.arange(-180,180,30)

motor_gain = 2.4                # K_д,            [рад/(В·с)]
motor_time_constant = 5e-3      # T = √(T_м·T_э), [с]
motor_damping = 0.125           # ξ

regulator_gain = 1              # K_р,            [рад/(В·с)]
regulator_time_constant = 0.002 # T_1,            [с]

motor_tf     = ctrl.tf([motor_gain], # W_д(s)
                       [motor_time_constant**2, 2*motor_damping*motor_time_constant, 1])
regulator_tf = ctrl.tf([regulator_gain], # W_р(s)
                       [regulator_time_constant, 1])

closed_loop_tf = ctrl.feedback(regulator_tf * motor_tf, 1)
open_loop_tf = regulator_tf * motor_tf

def first_x_at(y, xs_sampled, ys_sampled, poly_degree=4):
  global tolerance
  poly_approximation = np.poly1d(np.polyfit(xs_sampled, ys_sampled, poly_degree))
  xs_at_y = (poly_approximation - y).roots
  xs_at_y = xs_at_y[abs(xs_at_y.imag) < tolerance].real
  return xs_at_y[np.argmin(xs_at_y)]

def measure_frequency_response( # "Измерение" АФХ на выходе ctrl.forced_response
  tf, max_time_constant, freq_to_model_from=0.5, points_to_model_per_decade=1e2,
):
  # Бинарный поиск частоты ω, до которой будет происходить моделирование,
  # по условию A(ω) >= A(ω_п) = 0.1*A(0)
  global tolerance
  freq_low = freq_to_model_from
  freq_high = freq_to_model_from * 1e10
  while freq_high - freq_low > tolerance:
    freq_mid = (freq_low + freq_high) / 2
    if np.abs(ctrl.evalfr(tf, 1j*freq_mid)) > 0.1 * ctrl.dcgain(tf):
      freq_low = freq_mid
    else:
      freq_high = freq_mid
  freq_to_model_till = freq_high # ω_п, [рад/с]

  points_to_model = int(points_to_model_per_decade
                        * (np.log10(freq_to_model_till) - np.log10(freq_to_model_from)))
  freqs_to_model = np.logspace( # ω, [рад/с]
    np.log10(freq_to_model_from),
    np.log10(freq_to_model_till),
    points_to_model
  ) 
  mag_response = np.empty(points_to_model) # A(ω), [-]
  phs_response = np.empty(points_to_model) # φ(ω), [°]

  # Моделирование АФХ
  def clip_min(arr, min_val):
    return np.clip(arr, a_min=min_val, a_max=None)
  for i, freq_i in enumerate(freqs_to_model):
    # Подобранные параметры точности/скорости расчётов
    periods_to_model_after_settling = np.floor(max(3e1*np.log10(freq_i), 5e0))
    time_constants_till_settled     =          max(4e1*np.log10(freq_i), 1e1)
    time_steps_per_period           =          max(5e1*np.log10(freq_i), 1e2)

    # Временные точки
    settling_time = time_constants_till_settled * max_time_constant
    input_sin_period = 2 * np.pi / freq_i # T(ω) = 2*π/ω
    time_to_model = settling_time + periods_to_model_after_settling * input_sin_period
    time_step = input_sin_period / time_steps_per_period
    timepoints_limit = 10_000
    if time_to_model / time_step > timepoints_limit:
      time_step = time_to_model / timepoints_limit
    timepoints = np.arange(0, time_to_model, time_step)
    settling_timepoint = np.argmin(np.abs(timepoints - settling_time))

    # Входной и выходной сигналы
    #input_sin = input_sin_mag * np.sin(freq_i * timepoints) # x(t) = x_m * sin(ω*t) (1)
    input_sin = np.sin(freq_i * timepoints) # x(t) = sin(ω*t)   <---   (x_m = 1) ∧ (x(t) = x_m * sin(ω*t))
    _, output_sin = ctrl.forced_response(tf, timepoints, input_sin) # output_sin[settling_timepoint:] ≈ A*sin(ω*t + φ)

    # Амплитуда
    output_sin_mag = np.sqrt(2) * np.std(output_sin, ddof=0) # y_m(ω)
    #mag_response[i] = output_sin_mag / input_sin_mag # A(ω) = y_m(ω) / x_m
    mag_response[i] = output_sin_mag # A(ω) = y_m(ω)   <---   (x_m = 1) ∧ (A(ω) = y_m(ω) / x_m)

    # Запаздывание и фаза
    corr = np.correlate( input_sin[settling_timepoint:] - np.mean( input_sin[settling_timepoint:]),
                        output_sin[settling_timepoint:] - np.mean(output_sin[settling_timepoint:]), mode='same')
    time_shift_samples = np.argmax(corr) - len(corr)//2
    time_shift = time_shift_samples * time_step

    phs_response[i] = 360 * time_shift / input_sin_period # φ(ω) = 360°*Δt(ω)/T(ω)  (10)

  phs_response = np.unwrap(phs_response + 180, period=360) - 180
  return freqs_to_model, mag_response, phs_response

def hline_named_var(ax, x, x_name, x_dim='', **kwargs):
  global plt_print_precision
  ax.axhline(x, label=f'{x_name} = {x:.{plt_print_precision}f}{x_dim}', **kwargs)
def vline_named_var(ax, y, y_name, y_dim='', **kwargs):
  global plt_print_precision
  ax.axvline(y, label=f'{y_name} = {y:.{plt_print_precision}f}{y_dim}', **kwargs)


# Замкнутая САУ 
if calc_tf[0]:
  closed_loop_poles = ctrl.poles(closed_loop_tf)
  closed_loop_is_stable = np.all(np.real(closed_loop_poles) < tolerance)
  if closed_loop_is_stable:
    # 1. Моделирование АЧХ и ФЧХ
    cl_freqs,\
    cl_mag_response,\
    cl_phs_response = measure_frequency_response(closed_loop_tf, 1 / min(abs(closed_loop_poles)))

    # 2. Характеристики M, ω_р, ω_0, ω_п
    cl_loop_DC_gain = ctrl.dcgain(closed_loop_tf) # A(0)

    cl_resonance_idx = np.argmax(cl_mag_response)
    cl_resonance_mag = cl_mag_response[cl_resonance_idx] # A(ω_р)
    cl_resonance_factor = cl_resonance_mag / cl_loop_DC_gain # M
    cl_resonance_freq = cl_freqs[cl_resonance_idx] # ω_р

    cl_cutoff_mag = cl_loop_DC_gain / np.sqrt(2) # A(0)/√2
    cl_cutoff_freq = first_x_at(cl_cutoff_mag, cl_freqs, cl_mag_response) # ω_0

    cl_pass_mag = cl_mag_response[-1] # A(ω_п) = 0.1*A(0) 
    cl_pass_freq = cl_freqs[-1] # ω_п

    if show_fig[0]:
      _, (ax_cl_mag_response, ax_cl_phs_response) =\
        plt.subplots(2, 1, figsize=(10, 8), sharex=True, label='АЧХ и ФЧХ замкнутой САУ')
      ax_cl_mag_response.set_title('АЧХ')
      ax_cl_mag_response.set_ylabel('A(ω) [-]')
      ax_cl_mag_response.semilogx(cl_freqs, cl_mag_response, 'b.-')
      hline_named_var(ax_cl_mag_response, cl_loop_DC_gain,   'A(0)',            c='b', ls='--')
      vline_named_var(ax_cl_mag_response, cl_resonance_freq, 'ω_р', 'рад/с',    c='y', ls='-.')
      hline_named_var(ax_cl_mag_response, cl_resonance_mag,  'A(ω_р)',          c='y', ls='-.')
      vline_named_var(ax_cl_mag_response, cl_cutoff_freq,    'ω_0', 'рад/с',    c='r', ls=':' )
      hline_named_var(ax_cl_mag_response, cl_cutoff_mag,     'A(ω_0)=A(0)/√2',  c='r', ls=':' )
      vline_named_var(ax_cl_mag_response, cl_pass_freq,      'ω_п', 'рад/с',    c='g', ls='-.')
      hline_named_var(ax_cl_mag_response, cl_pass_mag,       'A(ω_п)=0.1*A(0)', c='g', ls='-.')
      ax_cl_mag_response.grid(True)
      ax_cl_mag_response.legend()

      ax_cl_phs_response.set_title('ФЧХ')
      ax_cl_phs_response.set_ylabel('φ(ω) [°]')
      ax_cl_phs_response.set_xlabel('ω [рад/с]')
      ax_cl_phs_response.semilogx(cl_freqs, cl_phs_response, 'b.-')
      ax_cl_phs_response.yaxis.set_major_locator(tck.MultipleLocator(45))
      ax_cl_phs_response.grid(True)
      ax_cl_phs_response.legend()
      plt.tight_layout()
      plt.show()

    # 7. Сравнение характеристик измеренных самостоятельно и библиотекой control
    if show_fig[1]:
      cl_mag_response_ctrl,\
      cl_phs_response_ctrl, _ = ctrl.bode(closed_loop_tf, cl_freqs, plot=False)
      cl_phs_response_ctrl = np.degrees(cl_phs_response_ctrl)

      _, (ax_cl_mag_compare, ax_cl_phs_compare) =\
        plt.subplots(2, 1, figsize=(10, 8), label='Сравнение АЧХ и ФЧХ замкнутой САУ', sharex=True)
      ax_cl_mag_compare.set_title('АЧХ')
      ax_cl_mag_compare.set_ylabel('A(ω) [-]')
      ax_cl_mag_compare.semilogx(cl_freqs, cl_mag_response,      'b.-', label='своя')
      ax_cl_mag_compare.semilogx(cl_freqs, cl_mag_response_ctrl, 'r-',  label='control')
      ax_cl_mag_compare.legend()
      ax_cl_mag_compare.grid(True)

      ax_cl_phs_compare.set_title('ФЧХ')
      ax_cl_phs_compare.set_ylabel('φ(ω) [°]')
      ax_cl_phs_compare.semilogx(cl_freqs, cl_phs_response,      'b.-', label='своя')
      ax_cl_phs_compare.semilogx(cl_freqs, cl_phs_response_ctrl, 'r-',  label='control')
      ax_cl_phs_compare.set_xlabel('ω [рад/с]')
      ax_cl_phs_compare.yaxis.set_major_locator(tck.MultipleLocator(45))
      ax_cl_phs_compare.legend()
      ax_cl_phs_compare.grid(True)
      plt.tight_layout()
      plt.show()

    if show_fig[2]:
      frequency_response = cl_mag_response * np.exp(1j * np.radians(cl_phs_response))
      frequency_response_ctrl = ctrl.evalfr(closed_loop_tf, 1j * cl_freqs)

      plt.figure(figsize=(10, 8), label='Сравнение АФХ замкнутой САУ')
      plt.plot(frequency_response     .real, frequency_response     .imag, 'b.-', label='своя')
      plt.plot(frequency_response_ctrl.real, frequency_response_ctrl.imag, 'r-',  label='control')
      
      plt.grid()
      plt.show()

# Разомкнутая САУ 
if calc_tf[1]:
  ol_poles = ctrl.poles(open_loop_tf)
  ol_is_stable = np.any(np.real(ol_poles) < tolerance)
  if ~ol_is_stable:
    print("Разомкнутая САУ неустойчива, анализ невозможен")
  else:
    # 3. Моделирование АЧХ и ФЧХ
    ol_freqs,\
    ol_mag_response,\
    ol_phs_response = measure_frequency_response(open_loop_tf, 1 / min(abs(ol_poles)))

    # 4. Запасы устойчивости по модулю ∆A и по фазе ∆φ
    ol_phs_after_180 = np.where(np.fmod(ol_phs_response + 180, 360) <= 0)[0]
    if len(ol_phs_after_180) == 0:
      print("Фаза φ <= -180° не достигнута в измеренном диапазоне.")
    else:
      ol_idx_after_phs_180 = ol_phs_after_180[0]
      if ol_idx_after_phs_180 == 0:
        ol_gain_at_phs_180 = ol_mag_response[0]
        ol_freq_at_phs_180 = ol_freqs[0]
      else:
        ol_gain_at_phs_180 = np.interp(180,
                                       ol_phs_response[ol_idx_after_phs_180-1:ol_idx_after_phs_180+1],
                                       ol_mag_response[ol_idx_after_phs_180-1:ol_idx_after_phs_180+1])
        ol_gain_at_phs_180_dB = 20 * np.log10(ol_gain_at_phs_180)
        ol_freq_at_phs_180 = np.interp(180,
                                       ol_phs_response[ol_idx_after_phs_180-1:ol_idx_after_phs_180+1],
                                       ol_freqs       [ol_idx_after_phs_180-1:ol_idx_after_phs_180+1])
      ol_gain_margin = 1 - ol_gain_at_phs_180 # ∆A
      ol_gain_margin_dB = 20 * np.log10(ol_gain_at_phs_180)

    ol_mag_after_1 = np.where(ol_mag_response <= 1)[0]
    if len(ol_mag_after_1) == 0:
      print("Амплитуда A <= 1 не достигнута в измеренном диапазоне.")
    else:
      ol_idx_after_mag_1 = ol_mag_after_1[0]
      if ol_idx_after_mag_1 == 0:
        ol_freq_at_mag_1 = ol_freqs[0]
        ol_phs_at_mag_1 = ol_phs_response[0]
      else:
        ol_freq_at_mag_1 = np.interp(1,
                                     ol_mag_response[ol_idx_after_mag_1-1:ol_idx_after_mag_1+1],
                                     ol_freqs       [ol_idx_after_mag_1-1:ol_idx_after_mag_1+1])
        ol_phs_at_mag_1  = np.interp(1,
                                     ol_mag_response[ol_idx_after_mag_1-1:ol_idx_after_mag_1+1],
                                     ol_phs_response[ol_idx_after_mag_1-1:ol_idx_after_mag_1+1])
      ol_phs_margin = 180 + ol_phs_at_mag_1 # ∆φ

      print(f"""Замкнутая САУ {'' if ol_gain_margin > 1 and ol_phs_margin > 0 else 'не'}устойчива
Запас по модулю ∆A = {ol_gain_margin
:.{plt_print_precision}f} на частоте {ol_freq_at_phs_180:.{plt_print_precision}f}
Запас по фазе Δφ = {ol_phs_margin
:.{plt_print_precision}f}° на частоте среза ω_ср = {ol_freq_at_mag_1:.{plt_print_precision}f} рад/с""")

    # 5. АЧХ и ФЧХ разомкнутой САУ
    if show_fig[3]:
      _, (ax_ol_log_mag_resp, ax_ol_phs_resp) =\
        plt.subplots(2, 1, figsize=(10, 8), label='ЛАЧХ и ФЧХ разомкнутой САУ', sharex=True)
      ax_ol_log_mag_resp.semilogx(ol_freqs, 20 * np.log10(ol_mag_response), 'b.-')
      if ol_freq_at_phs_180 is not None and ol_freq_at_mag_1 is not None:
        ax_ol_log_mag_resp.axhline(0, label='0 дБ',                                       c='r', ls='--')
        hline_named_var(ax_ol_log_mag_resp, ol_gain_at_phs_180_dB, 'L(ω(φ=-180°))', 'дБ', c='r', ls='--')
        vline_named_var(ax_ol_log_mag_resp, ol_freq_at_phs_180,    'ω(φ=-180°)', 'рад/с', c='g', ls=':')
        vline_named_var(ax_ol_log_mag_resp, ol_freq_at_mag_1,      'ω_ср',       'рад/с', c='g', ls=':')
      ax_ol_log_mag_resp.set_ylabel('L(ω) [дБ]')
      ax_ol_log_mag_resp.set_title('ЛАЧХ')
      ax_ol_log_mag_resp.legend()
      ax_ol_log_mag_resp.grid(True)

      ax_ol_phs_resp.semilogx(ol_freqs, ol_phs_response, 'b.-')
      if ol_freq_at_phs_180 is not None and ol_freq_at_mag_1 is not None:
        ax_ol_phs_resp.axhline(-180, label='-180°',                                 c='r', ls='--')
        hline_named_var(ax_ol_phs_resp, ol_phs_at_mag_1,     'φ(ω_ср)',        '°', c='r', ls='--')
        vline_named_var(ax_ol_phs_resp, ol_freq_at_phs_180,  'ω(φ=-180°)', 'рад/с', c='g', ls=':')
        vline_named_var(ax_ol_phs_resp, ol_freq_at_mag_1,    'ω_ср',       'рад/с', c='g', ls=':')
      ax_ol_phs_resp.set_xlabel('ω [рад/с]')
      ax_ol_phs_resp.set_ylabel('φ(ω) [°]')
      ax_ol_phs_resp.set_title('ФЧХ')
      ax_ol_phs_resp.legend()
      ax_ol_phs_resp.grid(True)
      plt.tight_layout()
      plt.show()

    # 6. АФХ разомкнутой САУ, анализ устойчивости
    ol_freq_response = ol_mag_response * np.exp(1j * np.radians(ol_phs_response))
    if show_fig[4]:
      _, ax_nyq = plt.subplots(figsize=(8, 8), label='АФХ разомкнутой САУ')
      ax_nyq.plot(np.real(ol_freq_response), np.imag(ol_freq_response), 'b.-')
      ax_nyq.plot(-1, 0, 'rx', markersize=10, label='(-1, j0)')
      ax_nyq.axhline(0, c='gray', ls=':')
      ax_nyq.axvline(0, c='gray', ls=':')
      ax_nyq.set_xlabel('Re')
      ax_nyq.set_ylabel('Im')
      ax_nyq.axis('equal')
      ax_nyq.legend()
      ax_nyq.grid(True)
      plt.show()

    # 7. Сравнение характеристик вычисленных самостоятельно и библиотекой control
    if show_fig[5]:
      ol_mag_response_ctrl,\
      ol_phs_response_ctrl, _ = ctrl.bode(open_loop_tf, ol_freqs, plot=False)
      ol_phs_response_ctrl = np.degrees(ol_phs_response_ctrl)

      _, (ax_ol_mag_compare, ax_ol_phs_compare) =\
        plt.subplots(2, 1, figsize=(10, 8), label='Сравнение АЧХ и ФЧХ разомкнутой САУ', sharex=True)
      ax_ol_mag_compare.semilogx(ol_freqs, ol_mag_response,      'b.-', label='своя')
      ax_ol_mag_compare.semilogx(ol_freqs, ol_mag_response_ctrl, 'r-',  label='control')
      ax_ol_mag_compare.set_ylabel('A(ω) [-]')
      ax_ol_mag_compare.set_title('АЧХ')
      ax_ol_mag_compare.legend()
      ax_ol_mag_compare.grid(True)

      ax_ol_phs_compare.semilogx(ol_freqs, ol_phs_response,      'b.-', label='своя')
      ax_ol_phs_compare.semilogx(ol_freqs, ol_phs_response_ctrl, 'r-',  label='control')
      ax_ol_phs_compare.set_xlabel('ω [рад/с]')
      ax_ol_phs_compare.set_ylabel('φ(ω) [°]')
      ax_ol_phs_compare.set_title('ФЧХ')
      ax_ol_phs_compare.legend()
      ax_ol_phs_compare.grid(True)
      plt.tight_layout()
      plt.show()

    if show_fig[6]:
      ol_freq_response_ctrl = ctrl.evalfr(open_loop_tf, 1j * ol_freqs)
      _, ax_nyq_cmp = plt.subplots(figsize=(8, 8), label='Сравнение АФХ разомкнутой САУ')
      ax_nyq_cmp.plot(ol_freq_response     .real, ol_freq_response     .imag, 'b.-', label='своя')
      ax_nyq_cmp.plot(ol_freq_response_ctrl.real, ol_freq_response_ctrl.imag, 'r-',  label='control')
      ax_nyq_cmp.plot(-1, 0, 'rx', markersize=10, label='(-1, j0)')
      ax_nyq_cmp.axhline(0, c='gray', ls=':')
      ax_nyq_cmp.axvline(0, c='gray', ls=':')
      ax_nyq_cmp.set_xlabel('Re')
      ax_nyq_cmp.set_ylabel('Im')
      ax_nyq_cmp.axis('equal')
      ax_nyq_cmp.legend()
      ax_nyq_cmp.grid(True)
      plt.show()

