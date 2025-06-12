close all; clear all; clc;

%% ============================ Parameter =================================
FsDAC = 256* 10^9;
FsADC = 80* 10^9;
Baud = 56* 10^9;
M = 4;
amplitudes = [-3, -1, 3, 1]; % Define PAM4 amplitude levels
sample_per_symbol = FsDAC / Baud;
RefLen = 1000;
% load("TestingBERSNR.mat"); % Using in testing SNR

fig_bottom = 1;
% Transmitter
%% =========================== Random Sequence ============================

tx1 = randi([0, 1], 1, 40000);

%% ==================== Group Bits into Symbols ===========================

tx1_grouped = reshape(tx1, 2, []); % Reshape into 2 rows (each column is a 2-bit group)
tx2 = tx1_grouped(1, :) * 2 + tx1_grouped(2, :); % Convert binary to decimal

%% ===================Map Symbols to PAM4 Amplitudes ======================

tx3 = amplitudes(tx2 + 1); % Map symbols to amplitudes

%% ================== Plot Constellation Diagram ==========================
if fig_bottom == 1
    figure;
    scatter(real(tx3), imag(tx3), 20, 'MarkerEdgeColor', [0 0.4470 0.7410], 'MarkerFaceColor', [0 0.4470 0.7410]);
    
    % Plot constellation points
    hold on;
    
    % Add binary labels
    for k = 1:M
        % Binary encoding for each level
        binary_label = dec2bin(k-1, 2); % Convert decimal to 3-bit binary
        text(real(amplitudes(k)), 0.2, binary_label,'HorizontalAlignment', 'center', 'Color', 'red'); % Binary label
        text(real(amplitudes(k)), -0.2, num2str(k-1), 'HorizontalAlignment', 'center', 'Color', 'red'); % Decimal value
    end
    hold off;
    % Adjust plot aesthetics
    xlabel('In-Phase');
    ylabel('Quadrature');
    title('Scatter plot');
    axis([-M M -2 2]); % Adjust axis limits
    grid on;
    
    figure;
    stem(tx3(:, 1:100), 'LineWidth', 2, 'MarkerSize', 4, 'Color', [0 0.4470 0.7410]);
    xlabel('Samples (n)');
    ylabel('Amplitude');
    title('Discrete Time-Domain PAM4 Signal');
    grid on;
end

%% ============= Upsample (Insert Zeros Between Symbols) ==================
tx_upsampled = upsample(tx3, round(sample_per_symbol)); % interpolation

%% ======== Apply Pulse Shaping Filter (e.g., Root-Raised Cosine) =========
span = 10; % Filter span in symbols
roll_off = 0.25; % Roll-off factor
rrc_filter = rcosdesign(roll_off, span, round(sample_per_symbol), 'sqrt'); % RRC filter design
% fvtool(rrc_filter, 'Analysis', 'impulse'); % check impulse response

% Convolve the upsampled signal with the RRC filter
tx4 = conv(tx_upsampled, rrc_filter, 'same'); % Apply pulse shaping

%% ====================== Generate Time Vector ============================
tx5 = [zeros(1, RefLen) tx4 zeros(1, RefLen)];
t = (0:length(tx5)-1) / FsDAC; % Time vector based on DAC sampling rate

% ============================ Plot Figures ============================= %
if fig_bottom == 1
    figure;
    
    % Discrete signal after upsampling
    subplot(2, 1, 1);
    stem(t(1:100) * 1e9, tx_upsampled(1:100), 'LineWidth', 2, 'MarkerSize', 4, 'Color', [0 0.4470 0.7410]); % Show the first 100 samples
    xlabel('Time (ns)');
    ylabel('Amplitude');
    title('Upsampled PAM4 Signal');
    grid on;
    
    % Filtered signal in time domain
    subplot(2, 1, 2);
    plot(t(1:100) * 1e9, tx4(1:100), 'LineWidth', 1, 'MarkerSize', 4, 'Color', [0 0.4470 0.7410]);
    xlabel('Time (ns)');
    ylabel('Amplitude');
    title('Filtered PAM4 Signal');
    grid on;
    
    figure;
    plot(tx5);
    xlabel('Time (ns)');
    ylabel('Amplitude');
    title('PAM4 Signal');
    xlim([0 length(tx5)]);
    
    % Eye Diagram
    eyediagram(tx4, 2* round(sample_per_symbol));
    grid on;
    
    % Autocorrelation
    maxlag = 100;
    [c,lags] = xcorr(tx4, maxlag, 'normalized');
    figure;
    stem(lags,c)
    xlabel('Lags');
    ylabel('Correlation (Normalized)');
    title('Autocorrelation of Signal');
    grid on;
end
%% ============================ Channel ===================================
tx6 = resample(tx5, FsADC, FsDAC);
SNR = 20;
rx1_simu = awgn(tx6, SNR, 'measured');

% ============================ Plot Figures ============================= %
if fig_bottom == 1
    maxlag = 100;
    [c,lags] = xcorr(rx1_simu, maxlag, 'normalized');
    figure;
    stem(lags,c)
    xlabel('Lags');
    ylabel('Correlation (Normalized)');
    title('Autocorrelation of Receive Signal');
    grid on;
end

% Receiver
%% ========================== Match Filter ================================
rx1 = resample(rx1_simu, FsDAC, FsADC);
match_filter = fliplr(rrc_filter);
rx2 = conv(rx1, match_filter, 'same');
t = (0:length(rx1)-1) / FsDAC; % 產生時間向量

% 移除Group Delay
gd = (length(match_filter)-1)/2;
if length(rx2) > 2*gd
    rx2 = rx2(gd+1 : end-gd);
end
tt = (0:length(rx2)-1) / FsDAC;

% 繪製結果
if fig_bottom ==1
    figure;
    subplot(2,1,1);
    plot(t * 1e9, rx1, 'b');
    xlabel('Time (ns)');
    ylabel('Amplitude');
    xlim([0 max(t* 1e9)])
    title('Before Match Filter');
    grid on;
    
    subplot(2,1,2);
    plot(tt * 1e9, rx2, 'b');
    xlabel('Time (ns)');
    ylabel('Amplitude');
    xlim([0 max(tt* 1e9)])
    title('After Match Filter');
    grid on;
end
%% ========================= Synchronization ==============================
syncLv = [0.1 0.9];
for corr = linspace(max(syncLv) , min(syncLv), 10)
    [syncIndex, syncValue] = winsync(rx2, tx4, RefLen, corr);
    disp(['... located', num2str(corr), 'at', num2str(syncIndex)]);
    if syncIndex < length(rx2) - length(tx4)    % Located
        rx3 = reshape(rx2(syncIndex : syncIndex + length(tx4) - 1), 1, []); 
        break
    elseif corr == min(syncLv)  % Intertupted
        disp('ERROR : Failed in synchronization!!');
        return
    end
end
% Plot Figure
if fig_bottom == 1
    figure;
    subplot(2,1,1);
    plot(tt *1e9, rx2, 'b'); title('Signal before synchronization');
    xlabel('Time (ns)');
    ylabel('Amplitude');
    xlim([0 max(tt* 1e9)])
    subplot(2,1,2);
    plot(rx3, 'r'); title('Signal after synchronization');
    xlabel('Time (ns)');
    ylabel('Amplitude');
    xlim([0 max(length(rx3))])

    grid on;
    % Eye Diagram
    eyediagram(rx3, 2* round(sample_per_symbol));
    grid on;
end
%% ============================= Downsample ===============================
rx4 = downsample(rx3, round(sample_per_symbol));

if fig_bottom == 1
    figure;
    scatter(real(rx4), imag(rx4), 20, 'MarkerEdgeColor', [0 0.4470 0.7410], 'MarkerFaceColor', [0 0.4470 0.7410]);
    % Plot constellation points
    hold on;
    
    % Add binary labels
    for k = 1:M
        % Binary encoding for each level
        binary_label = dec2bin(k-1, 2); % Convert decimal to 3-bit binary
        text(real(amplitudes(k)), 0.2, binary_label,'HorizontalAlignment', 'center', 'Color', 'red'); % Binary label
        text(real(amplitudes(k)), -0.2, num2str(k-1), 'HorizontalAlignment', 'center', 'Color', 'red'); % Decimal value
    end
    hold off;
    % Adjust plot aesthetics
    xlabel('In-Phase');
    ylabel('Quadrature');
    title('Scatter plot');
    axis([-M M -2 2]); % Adjust axis limits
    grid on;
end

%% =========================== Equalizer ==================================

L = 3;
a = 0.0001;
training_length = 10000;
x = rx4(1 : training_length);
d = tx3(1 : training_length);
y = LMS_Equalizer(x, rx4, d, L, a);
rx5 = y(training_length + 1 : end);

%% =========================== Demodulation ===============================
rx6 = zeros(1, length(rx5));
for i = 1:length(rx5)
    if rx5(i) > 2
        rx6(i) = 3;
    elseif (rx5(i) > 0) && (rx5(i) <= 2)
        rx6(i) = 1;
    elseif (rx5(i) > -2) && (rx5(i) <= 0)
        rx6(i) = -1;
    else
        rx6(i) = -3;
    end
end

%% ============================ Demapping =================================
rx7 = zeros(1, 2*length(rx6));  % 最終 bit array
for i = 1:length(rx6)
    if rx6(i) == -3
        rx7(2*i-1 : 2*i) = [0 0];
    elseif rx6(i) == -1
        rx7(2*i-1 : 2*i) = [0 1];
    elseif rx6(i) ==  1
        rx7(2*i-1 : 2*i) = [1 1];
    elseif rx6(i) ==  3
        rx7(2*i-1 : 2*i) = [1 0];
    end
end

%% =========================== Evaluation =================================

tx_short = tx1(2*training_length + 1 : end);
rx_short = rx7(1:end);

num_error = sum(tx_short ~= rx_short);
BER = num_error / length(rx_short);
BERMatrix = [];
BERMatrix = BER;
fprintf('Bit Error Rate (BER) = %g\n', BER);
