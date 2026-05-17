clear all; close all;

%% 1. �������� ���������� ��� ���������� �������
U = [3 6 0 -4];
T = [3 5];
fd = 8;
Td = 1/fd;

%% 2. ������������� �������
t1 =    0:Td:T(1);
t2 = T(1):Td:T(2);

a1 = U(2) - U(1);
b1 = U(1);
u1 = a1*t1 + b1; % ������ �������� �������� ������������������� �������

a2 = U(4) - U(3);
b2 = U(3);
u2 = a2*t2 + b2; % ������ �������� �������� ������������������� �������

discrete_t = [t1 t2];
u = [u1 u2]; % ���� ������������������ ������

%% 3. ������ �������� ���������� ��� ��������
N   = length(discrete_t) % ����� �������� ����������� �������
U0  = sum(u) % ����� ���� �������� �������
Upi = sum(u(1:2:end) - u(2:2:end)) % ��������������� ����� �������� �������
E   = sum(u.^2) % ������� �������

%% 4. �������������� ����������� ������� �� ������� ������������
steps_around = 5;
more_discrete_t = -steps_around*Td + discrete_t(1): Td/10 : discrete_t(N) + steps_around*Td;
s = zeros(size(more_discrete_t));
for i = 1:length(more_discrete_t)
%   unsorted_summands = zeros(length(discrete_t));
%   for k = 1:length(discrete_t)
%     unsorted_summands(k) = u(k)*sinc((more_discrete_t(i) - (k - 1)*Td) / Td);
%   end
%   [~,idx] = sort(abs(unsorted_summands));
%   sorted_summands = unsorted_summands(idx);
%   for k = 1:length(discrete_t)
%     s(i) = s(i) + sorted_summands(k);
%   end
  for k = 1:length(discrete_t)
    s(i) = s(i) + u(k)*sinc((more_discrete_t(i) - (k - 1)*Td) / Td);
  end
end
figure('Name', '������������������ ������, �������������� ���������� � ��� ����������', 'NumberTitle','off');
stem(discrete_t, u);
hold on;
plot(more_discrete_t, s, 'r');

%% 5. ��������� �������� ���������� ��� ��������
y_min = inf;
y_max = -inf;
x_min = 0;
x_max = 0;
hold off;