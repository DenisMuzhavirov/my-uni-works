close all; clear all;
clc;
show_fig = [1,1,1,1];
pulses_to_show = 1;

%% 1. Загрузка данных
file_id = fopen('lab4_data1.txt', 'r');
file_content = fread(file_id, '*char').';
fclose(file_id);
% delete utf-8 BOM
if file_content(1:3) == [239 187 191]
  file_content = file_content(4:end);
end % if

lines = strsplit(strtrim(file_content));
signal = zeros(size(lines));
for i = 1:length(lines)
  if lines{i}(end) == 'e'
    signal(i) = str2double(lines{i}(1:end-2));
  elseif ~isnan(str2double(lines{i}))
    signal(i) = str2double(lines{i}         );
  else
    disp(i)
    disp(lines{i})
  end
end % for i

if show_fig(1)
  figure('Name','Зашумлённый сигнал',...
         'NumberTitle','off');
  plot(signal);
end % if show_fig(1)

%% 2. Определение частоты дискретизации сигнала
DFT = fft(signal .^ 2); % [1]*e^[cycles]
% (*) remove redundant info by cutting of negative mirror-half (fft returns [0 + -] ~ [[0,π)∪[-π,0)])
% and accounting for it by doubling magnitudes of positive frequencies
mag_response = 2 * abs(DFT(1:ceil(length(DFT)/2))); % [1]
mag_response(1) = .5 * mag_response(1);
if show_fig(2)
  figure('Name','АЧХ зашумлённого сигнала',...
         'NumberTitle','off');
  plot(mag_response);
end % if show_fig(2)

[~, pulse_cycles] = max(mag_response(2:end)); % [1]
pulse_duration_seconds = 45 * 1e-3; % [s]
pulse_cycle_seconds    = 85 * 1e-3; % [s]
pulse_samples_per_cycle  = length(signal) / pulse_cycles; % [samples = (samples) / (1)]
pulse_samples_per_second = pulse_samples_per_cycle / pulse_cycle_seconds; % [samples/s = (samples) / (s)]

%% 3. Разделение сигнала на фрагменты
pulses = reshape(signal, pulse_samples_per_cycle, []);
pulse_duration_samples = pulse_duration_seconds * pulse_samples_per_second; % [samples = (s) * (samples/s)]
assert(pulse_duration_samples - round(pulse_duration_samples) < .5);
pulse_duration_samples = round(pulse_duration_samples);
pulses_active   = pulses(                       1:pulse_duration_samples,:); % information + noise
pulses_inactive = pulses(pulse_duration_samples+1:end                   ,:); % pure noise (white noise assumed)
white_noise_variation = var(pulses_inactive(:));
one_freq_of_noise_variation = pulse_duration_samples * white_noise_variation;

%% 4. Вычисление ДПФ-спектров фрагментов сигнала
pulses_DFT = fft(pulses_active); % [1]*e^[cycles]

positive_half_of_bins = 1:ceil(pulse_duration_samples/2); % see (*)
mags_per_pulse = 2 * abs(pulses_DFT(positive_half_of_bins, :)); % [1]
mags_per_pulse(1) = .5 * mags_per_pulse(1);
freq_resolution = pulse_samples_per_second / pulse_duration_samples; % [Hz = (samples/s) / (samples)]
freqs = (positive_half_of_bins - 1) * freq_resolution; % [Hz]
if show_fig(3)
  for i = 1:min(pulses_to_show,pulse_cycles)
    figure('Name', cstrcat('АЧХ ', num2str(i), '-го импульса'),...
           'NumberTitle','off');
    plot(freqs, mags_per_pulse(:, i));
  end
end % if show_fig(3)

%% 5. Выделение нужных частот
DTMF_row_freqs = [ 697, 770, 852, 941];    % [Hz]
DTMF_col_freqs = [1209,1336,1477];%,1633]; % [Hz]

row_bins = 1 + round(DTMF_row_freqs * pulse_duration_seconds);
col_bins = 1 + round(DTMF_col_freqs * pulse_duration_seconds);

%pulse_DTMF_rows = goertzel(pulses_active, row_bins);
%pulse_DTMF_cols = goertzel(pulses_active, col_bins);
pulse_DTMF_rows = pulses_DFT(row_bins,:);
pulse_DTMF_cols = pulses_DFT(col_bins,:);

%% 6. Определение передаваемых частот
DTMF_row_mags = abs(pulse_DTMF_rows); % [1]
[~, most_probable_row] = max(DTMF_row_mags);
DTMF_col_mags = abs(pulse_DTMF_cols);
[~, most_probable_col] = max(DTMF_col_mags);
symbols = ['1' '2' '3' ;% 'A';
           '4' '5' '6' ;% 'B';
           '7' '8' '9' ;% 'C';
           '*' '0' '#'];% 'D'];
decoded_msg = symbols(sub2ind(size(symbols), most_probable_row, most_probable_col));

if show_fig(4)
  for i = 1:min(pulses_to_show, pulse_cycles)
    DTMF_row_nrgs = DTMF_row_mags(:,i) .^ 2;
    DTMF_col_nrgs = DTMF_col_mags(:,i) .^ 2;
    DTMF_button_nrg = DTMF_row_nrgs + DTMF_col_nrgs.';
    DTMF_total_nrg = sum(DTMF_row_nrgs) + sum(DTMF_col_nrgs);
    button_probability = exp(  -(DTMF_total_nrg - DTMF_button_nrg)
                             /  (2 * one_freq_of_noise_variation));
    button_probability = button_probability / sum(button_probability(:));

    figure('Name', sprintf('DTMF сетка импульса %d', i), 'NumberTitle', 'off');
    imagesc(button_probability, [0 1]); colorbar;
    for r = 1:length(DTMF_row_freqs)
      for c = 1:length(DTMF_col_freqs)
	      if r == most_probable_row(i) && c == most_probable_col(i)
          txtColor = 'k';
        else
          txtColor = 'w';
        end
        text(c, r, symbols(r,c),...
             'HorizontalAlignment', 'center',...
             'VerticalAlignment', 'middle',...
             'Color', txtColor,...
             'FontWeight', 'bold',...
             'FontSize', 14);
      end % for c
    end % for r
    set(gca,...
        'YTick', 1:length(DTMF_row_freqs),...
        'YTickLabel', num2str(DTMF_row_freqs(:)),...
        'XTick', 1:length(DTMF_col_freqs),...
        'XTickLabel', num2str(DTMF_col_freqs(:)));
    xlabel('Столбцовые частоты (Гц)');
    ylabel('Строковые частоты (Гц)');
  end % for i
end % if show_fig(4)

%% Итог
fprintf(['Период следования DTMF сигналов = %g отсчётов\n\n'...
         'Частота дискретизации сигнала = %g Гц\n\n'...
         'Номера спектральных отсчётов наиболее\n'...
         'близких к частотам DTMF = %s\n\n'...
         'Строка переданных в сигнале символов = %s\n\n']...
         , pulse_samples_per_cycle...
         , pulse_samples_per_second...
         , mat2str([row_bins col_bins])...
         , decoded_msg);
