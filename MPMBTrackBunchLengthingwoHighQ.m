% 多粒子多束团纵向追踪-STABLE
% 研究谐波腔致束团拉伸
% GPU-accelerated
% author： Tianlong He
% modified: Peizhi Fang; Add BbB feedback module & detune frequency calculate iteration

clc;clear;

%% beam parameters
% MAX-IV Params
cspeed = 299792458; 
sigma_t0 = 100e-12;      % s initial rms bunch length
sigma_e0 = 7.69e-4;     %  rms energy spread
alpha_c  = 3.06e-4;      %  momentum compaction factor
tau_s    = 25e-3;       %  radiation damping time
tau_z    = 25e-3;       %  radiation damping time
I0     = 600*1e-3;        %  beam current
E0     = 3e9;         %  beam energy
U0     = 363.8e3;         %  energy loss per turn
V_mc   = 1e6;         %  main cavity voltage
h      = 176;           % harmonic number
n_hc   = 3;             % harmonic order of HHC
Q_hc   = 20800;           % HHC loaded quality factor
R_hc   = 2.75e6*2;       % HHC loaded shunt impedance
C      = 528;        % Circumference of the ring
% fre_shift = detune_HC_calc(I0,n_hc,C,h,U0,V_mc,R_hc,Q_hc);% in near-optimum lengthening condition
[fre_shift, HC_sc] = detune_HC_iter_calc(I0,n_hc,C,h,U0,V_mc,R_hc,Q_hc,sigma_t0,sigma_e0,alpha_c,E0);
% fre_shift = 145.642e3;
R_mc = 0.32e6*4;
Q_mc = 3688;
fre_shift_mc = 0;

% fill pattern
pattern  = ones(1,h);

fillrate = length(find(pattern==1))/h;
HALF = machine(C,I0,U0,E0,tau_s,tau_z,sigma_t0,sigma_e0,alpha_c,h,V_mc,n_hc,R_hc,Q_hc,fillrate,fre_shift,Q_mc,R_mc,fre_shift_mc);
HALF.ShortRange_on = 0; % 0 - neglecting short range effect, 1 considering.

% 
% HOMs_m0;  % add HOMs  see the codes of HOM_m0.m
% PI_Set;   % add PI    see the codes of PI_Set.m

%% bunch generation
Par_num = 1e4; Bun_num = length(find(pattern==1));

%% charge pattern

% HALF
charge = ones(1,h).*pattern; 

charge = charge/sum(charge)*Bun_num;

% generation of initial distribution

q1 = TruncatedGaussian(1, [-3,3], [Par_num,1]);
p1 = TruncatedGaussian(1, [-3,3], [Par_num,1]);
q  = repmat(q1,1,Bun_num);
p  = repmat(p1,1,Bun_num);%
% CPU to GPU
Q=gpuArray(single(q)); P=gpuArray(single(p));     % single type

index_add = 1:Bun_num;
index_add = gpuArray(single(index_add-1));

Dq = 0.05;
%% wake data (intrabunch motion)
tau_q = (0:Dq:200)'*sigma_t0;
Wake_inter_hc = -HALF.wr_hc *  R_hc /Q_hc*exp(-tau_q*HALF.wr_hc/2/Q_hc) .*(...
    cos(tau_q*HALF.wr_hc*HALF.rot_coef_hc)-HALF.VbImagFactor_hc*sin(tau_q*HALF.wr_hc*HALF.rot_coef_hc));
Wake_inter_hc(1) = Wake_inter_hc(1)/2;

Wake_inter_mc = -HALF.wr_mc *  R_mc /Q_mc*exp(-tau_q*HALF.wr_mc/2/Q_mc) .*(...
    cos(tau_q*HALF.wr_mc*HALF.rot_coef_mc)-HALF.VbImagFactor_mc*sin(tau_q*HALF.wr_mc*HALF.rot_coef_mc));
Wake_inter_mc(1) = Wake_inter_mc(1)/2;

Wake_inter = Wake_inter_hc + Wake_inter_mc;

bin_tau = Dq*sigma_t0;
%% Longitudinal RW wake
% load('wakez_rw.mat');
% Wake_rw = wake_rw_calc(t,bin_tau,tau_q,wakelong);
% % plot(t,wakelong,'r','LineWidth',1.5);hold on;
% % plot(tau_q,Wake_rw,'b','LineWidth',1.5);xlabel('t [s]');ylabel('V/C');grid minor;title('long.rw wake');
% Wake_inter = Wake_inter - Wake_rw;  % add longitudinal rw wake

%% Longitudinal Geometry wake
% load('wakez_geo.mat');
% Wake_geo = interp1(t,wakelong,tau_q(2:end)); Wake_geo=[wakelong(1)/2;Wake_geo];
% %plot(tau_q,Wake_geo,'b','LineWidth',1.5);xlabel('t [s]');ylabel('wake V/C');grid minor;title('long.geo. wake');
% Wake_inter = Wake_inter - Wake_geo;
%% BBR WAKE
% fr = 30e9; Rs = 2e3;
% Wake_BBR = 2*pi*fr*Rs*exp(-tau_q*2*pi*fr/2).*(cos(tau_q*2*pi*fr*sqrt(0.75))-sin(tau_q*2*pi*fr*sqrt(0.75))/2/sqrt(0.75));
% Wake_BBR(1) = Wake_BBR(1)/2;
% % plot(tau_q,Wake_BBR);
% Wake_inter = Wake_inter - Wake_BBR;

% wr_bb = 11.549e9*2*pi;   % Hz
% Rs_bb = 5.730e3;         % Ohm
% Q_bb  =6;
% az = wr_bb/(2*Q_bb);wr1 = sqrt(wr_bb^2-az^2);
% Wake_bb = -2*az*Rs_bb*exp(-az*tau_q).*(cos(-wr1*tau_q)+az/wr1*sin(-wr1*tau_q));
% Wake_bb(1) = Wake_bb(1)/2;
% Wake_inter = Wake_inter + Wake_bb;
%%
Wake_inter    = gpuArray(single(Wake_inter));
% plot(tau_q,Wake_inter);

%% start tracking Track_num = 1e3
% charge per macro-particle   : HALF.qc
HALF.qc   = charge.*pattern * HALF.qc / Par_num;             
% induced voltage per macro-particle  : HALF.V_b
HALF.Vb_hc  = HALF.qc * HALF.wr_hc * HALF.R_hc / HALF.Q_hc; 
HALF.Vb_mc  = HALF.qc * HALF.wr_mc * HALF.R_mc / HALF.Q_mc; 

% HALF.V_b  = HALF.qc * HALF.w_r * HALF.R_hc / HALF.Q_hc *(1+1i*HALF.VbImagFactor); 
% initial loaded voltage
V_hc_load_0_real = real(HALF.V_hc_load_0);
V_hc_load_0_imag = imag(HALF.V_hc_load_0);
V_hc_load_0      = V_hc_load_0_real+1i*V_hc_load_0_imag;
% V_hc_load_0      =0;

V_mc_load_0_real = real(HALF.V_mc_load_0);
V_mc_load_0_imag = imag(HALF.V_mc_load_0);
V_mc_load_0      = V_mc_load_0_real+1i*V_mc_load_0_imag;
% V_mc_load_0=0; 


rot_decay_coef_hc = 1i * HALF.rot_coef_hc - 1 / (2 * HALF.Q_hc);  % rotation+decay
TbAng_coef_hc     = exp(rot_decay_coef_hc * HALF.angle_hc);       % 
exp_ang_coef_hc   = -rot_decay_coef_hc * HALF.wr_hc * sigma_t0;

rot_decay_coef_mc = 1i * HALF.rot_coef_mc - 1 / (2 * HALF.Q_mc);  % rotation+decay
TbAng_coef_mc     = exp(rot_decay_coef_mc * HALF.angle_mc);       % 
exp_ang_coef_mc   = -rot_decay_coef_mc * HALF.wr_mc * sigma_t0;

wake_kick_coef = HALF.qc * HALF.kick_coef;

Vg_mc = abs(HALF.Vg_mc_init);
[Vg_angle]=round(Vb_angle_calc(real(HALF.Vg_mc_init),imag(HALF.Vg_mc_init))*1e12)/1e12;
HALF.Vg_mc_track = HALF.Vg_mc_init;
HALF.rfcoef1_track     = HALF.rfcoef1 / HALF.V_mc * Vg_mc;
fai_s_track            = pi/2-Vg_angle;                    % 发射机电压矢量的同步相位

Track_num  = 50e4;   % set tracking turns
% record parameters 
Recor_step = 10;
HALF.Recor_step=Recor_step;
Recor_num  = Track_num / Recor_step;
% Vg_mc_track_record = zeros(1,Track_num*h/10);
Vb_hc_track_record = zeros(1,Track_num);
record_Q_mean = zeros(Recor_num,Bun_num);record_Q_std = zeros(Recor_num,Bun_num);
record_P_mean = zeros(Recor_num,Bun_num);record_P_std = zeros(Recor_num,Bun_num);


%% BbB module setting

record_th = 0;
effective_kfb_vec = [];
centroid_pos = [];
fb_turn = 1000;                        % Turn at which feedback is turned on
G = 3e5;                                % Total feedback gain
Nd = 0;                                 % Integer turn delay (turns)
% ------------------------------------------------------
% FIR filter design (based on downsampled slow clock)
% ------------------------------------------------------
D = 1;                                  % Downsampling factor (sample/compute every D turns)
Fs = (cspeed / C) / D;                  % Equivalent downsampled sampling rate (~17.7 kHz)
h_fir = [1];
N_taps = length(h_fir);
% ------------------------------------------------------
% [Mode B Core Modification] Downsampled ring buffer configuration
% ------------------------------------------------------
Nd_dec = ceil(Nd / D);                  % Turn delay in downsampled turns
fb_turn_dec = ceil(fb_turn / D);        % Activation turn in downsampled turns
% Downsampled buffer size: only needs to hold (N_taps + Nd_dec) downsampled points
buf_size_dec = Nd_dec + N_taps + 2; 
ring_buffer_dec = zeros(buf_size_dec, h); % Buffer for downsampled history data only
% Zero-Order Hold (ZOH) latch register and PA state initialization
V_dsp_held = zeros(1, h);
V_pa_state = zeros(1, h);               % Ensure PA state variable is initialized
enable_clipping = false;                % Enable/disable voltage clipping
V_max = 500;                            % Maximum voltage limit
% Power amplifier analog bandwidth parameters (bucket-to-bucket continuous)
BW_PA = 10000e6;                        % 50 MHz
T_bucket = C / cspeed / h;              % Bucket spacing
alpha_PA = exp(-2.0 * pi * BW_PA * T_bucket);

%%
gd = gpuDevice(); 
tic;

for i =1:Track_num
    % drift
    Q = Q + P * HALF.drift_coef;   
    Q_min = min(Q);    
    Q_new = round((Q - Q_min) *(1/Dq));
        
%% Harmonic cavity     
    % beam induced voltage at nominal bucket position HHC
    exp_angle  = exp(exp_ang_coef_hc * Q);
    exp_angle_sum= gather(sum(exp_angle));       % 耗时 0.007s sum()函数较慢    
    V_load_cpu = double(exp_angle_sum.*HALF.Vb_hc(pattern==1)); % *HALF.Vb_hc  
    [V_load,V_hc_load_0]=VoltageLoadCalc_matlab(V_hc_load_0,V_load_cpu,TbAng_coef_hc,pattern); 
    Vb_hc_track_record(i)=mean(V_load);

    if mod(i,1000)==0
        figure(13)
        subplot(2,1,1)
        plot(abs(V_load)/1e3);
        title('Harmonic Cavity');ylabel('Amplitude [kV]');
        subplot(2,1,2)
        plot(imag(V_load)/1e3);ylabel('Phase [deg]');xlabel('Bucket ID');
    end

    % Vc_hc = 2*I0*R_hc*cos(atan(2*Q_hc*fre_shift/300e6))*exp(1i*atan(2*Q_hc*fre_shift/300e6))*ones(1,h);
    Vc_hc = V_load;
    V_load_cpu = Vc_hc(pattern==1)*HALF.kick_coef;   % 约化V_load;
    V_load_gpu     = gpuArray(single(V_load_cpu));    
    % intrabunch kick    - V_load_kick    real part
    V_hc_load_kick = V_load_gpu./exp_angle; 

% _________________________________________________________________________    
%% Main cavity       
    % beam induced voltage at nominal bucket position MC
    exp_angle  = exp(exp_ang_coef_mc * Q);
    exp_angle_sum= gather(sum(exp_angle));       % 耗时 0.007s sum()函数较慢    
    V_load_cpu_mc = double(exp_angle_sum.*HALF.Vb_mc(pattern==1)); % *HALF.Vb_mc
    [V_load_mc,V_mc_load_0]=VoltageLoadCalc_matlab(V_mc_load_0,V_load_cpu_mc,TbAng_coef_mc,pattern);

    % Vc_mc = -V_mc*exp(1i*acos(U0/V_mc))*ones(1,h);
    Vc_mc = ones(1,h)*HALF.Vrf_ideal; % Force MC volatge to ideal setting
    % [Vc_mc,Vg_mc_track,HALF.Vg_mc_track_0,V_load,HALF.V_mc_load_0,PI]=PI_Control(PI,HALF.Vrf_ideal,HALF.Vg_mc_track_0,...
    % HALF.V_mc_load_0,V_load_cpu,TbAng_coef_mc,pattern); % every 5120 buckets to do PI

    V_mc_kick = gpuArray(single(Vc_mc(pattern==1)*HALF.kick_coef))./exp_angle;
%% BbB feedback
    % ------------------------------------------------------
    % Step 1: Digital downsampling and FIR filtering (triggered every D turns)
    % ------------------------------------------------------
    if mod(i, D) == 0
        % Calculate downsampled cycle counter (Sample Index)
        sample_idx = i / D;
        % Calculate write pointer for downsampled buffer
        curr_ptr_dec = mod(sample_idx - 1, buf_size_dec) + 1;
        % Extract centroid positions for all bunches (1 x h) and store into buffer
        current_centroids = gather(mean(Q)); 
        ring_buffer_dec(curr_ptr_dec, :) = current_centroids;
        % Check if buffer has accumulated sufficient history and passed activation turn
        if sample_idx > (Nd_dec + N_taps + fb_turn_dec)
            fir_inputs = zeros(N_taps, h); 
            for k = 1:N_taps
                % Directly index downsampled history (no multiplication by D required)
                target_sample = sample_idx - Nd_dec - (k - 1);
                ptr = mod(target_sample - 1, buf_size_dec) + 1;
                fir_inputs(k, :) = ring_buffer_dec(ptr, :);
            end
            % Compute FIR digital output voltage
            V_dsp_held = G * (h_fir * fir_inputs) * sigma_t0 * cspeed; 
        end
    end
    % ------------------------------------------------------
    % Step 2: Zero-Order Hold (ZOH)
    % ------------------------------------------------------
    % During non-sampling turns (mod(i, D) ~= 0), hold previous V_dsp_held
    V_dsp = V_dsp_held; 
    % ======================================================
    % Step 3: Bucket-to-bucket analog continuous response (Crosstalk)
    % ======================================================
    if i > (Nd + (N_taps - 1) * D + fb_turn)
        V_fb_final = zeros(1, h);
        
        % V_pa_state carries residual state from bucket (h) of the previous turn
        for idx = 1:h
            V_clipped = V_dsp(idx);
            
            if enable_clipping
                if V_clipped > V_max,  V_clipped = V_max;  end
                if V_clipped < -V_max, V_clipped = -V_max; end
            end
            
            % Single-pole low-pass model: decay residual voltage and add new input
            V_pa_state = alpha_PA * V_pa_state + (1.0 - alpha_PA) * V_clipped;
            
            V_fb_final(idx) = V_pa_state;
        end
        
        V_fb = V_fb_final;
    else
        V_fb = zeros(1, h);
        V_pa_state = 0.0; % Reset state while feedback is disabled
    end
    exp_angle_fb = exp(0 * Q);
    V_fb_kick    = gpuArray(single(V_fb(pattern==1)*HALF.kick_coef))./exp_angle_fb;
    % V_fb_kick = 0;
%% short-range wake kick   
    % count bins
    if HALF.ShortRange_on ==1                 % modified in 2022/11/14
        binnum=max(max(Q_new))+1; binnum=gather(binnum);
        bin_num_q=sum(BinNumCalZ(binnum,Q_new));   % double type
        bin_num_q=reshape(bin_num_q,binnum,Bun_num);    
        kick_conv = conv2(bin_num_q,Wake_inter(1:binnum));    
        kick_conv = kick_conv(1:binnum,:) .* wake_kick_coef(pattern==1)*min(i/5000,1);
    % wake kick
        Q_new = Q_new + (1 + index_add * binnum);  % modified in 2020/09/22
        wake_kick = kick_conv(Q_new);
    else
        wake_kick = 0;
    end
    % radiation damping and quantum excitation term  + wake_kick
    rad_quan_kick = -HALF.radampcoef * P + HALF.quanexcoef *...
        gpuArray.randn(Par_num,Bun_num,'single');

    P = P + rad_quan_kick - HALF.ploss;

    P = P - real(V_hc_load_kick) + imag(V_hc_load_kick) * HALF.VbImagFactor_hc...
        - real(V_mc_kick) + imag(V_mc_kick) * HALF.VbImagFactor_mc+ wake_kick -real(V_fb_kick);    
    
    if mod(i,2000)==0
        Centroid_std=std(record_Q_mean(record_th,:))*HALF.sigma_t0*1e12;
        disp(['tracking turn = ',num2str(i),'; Centroid_std = ',num2str(Centroid_std),'ps']);
        toc;
    end
    % output data
    if mod(i,Recor_step)==0
        record_th = record_th +1;
        record_Q_mean(record_th,:)=gather(mean(Q));
        record_Q_std(record_th,:)=gather(std(Q));
        record_P_mean(record_th,:)=gather(mean(P));
        record_P_std(record_th,:)=gather(std(P));
    end
end
wait(gd);
toc;
%% savefile
filename=['HALF_100percent_I0',num2str(I0*1e3),'mA','_RLfp',num2str(R_hc),...
    '_QLfp',num2str(Q_hc),'_detune',num2str(fre_shift),'_fb_',num2str(G / (sigma_t0 * cspeed)),'.mat'];
save(filename,'record_Q_mean','record_Q_std','record_P_mean','record_P_std','Q','Track_num','HALF','Bun_num','Vb_hc_track_record');


%% plot
figure(1);
Recor_step=HALF.Recor_step;
Nturns = (1:Track_num/Recor_step)*Recor_step;
for i=1:10:h
    subplot(2,2,1)
    plot(Nturns,record_Q_mean(:,i)*HALF.sigma_t0*1e12); hold on;
    subplot(2,2,2)
    plot(Nturns,record_Q_std(:,i)*HALF.sigma_t0*1e12); hold on;
    subplot(2,2,3)
    plot(Nturns,record_P_mean(:,i)*HALF.sigma_e0); hold on;
    subplot(2,2,4)
    plot(Nturns,record_P_std(:,i)*HALF.sigma_e0); hold on; 
end
subplot(2,2,1);ylabel('<\tau>  [ps]');xlabel('turns');xlim([1,Track_num]);grid on;
set(gca,'FontName','Times New Roman','FontSize',12);
subplot(2,2,2);ylabel('\sigma_{\tau}  [ps]');xlabel('turns');xlim([1,Track_num]);grid on;
set(gca,'FontName','Times New Roman','FontSize',12);
subplot(2,2,3);ylabel('<\delta> ');xlabel('turns');xlim([1,Track_num]);grid on;
set(gca,'FontName','Times New Roman','FontSize',12);
subplot(2,2,4);ylabel('\sigma_{\delta} ');xlabel('turns');xlim([1,Track_num]);grid on;
set(gca,'FontName','Times New Roman','FontSize',12);
% %%
figure(2);
for i=40000:500:50000
subplot(1,2,2);plot(record_Q_mean(end-i,:)*HALF.sigma_t0*1e12,'.');hold on;
ylabel('<\tau>  [ps]');xlabel('bunch number');
subplot(1,2,1);plot(record_Q_std(end-i,:)*HALF.sigma_t0*1e12,'.');hold on;
ylabel('\sigma_{\tau}  [ps]');xlabel('bunch number');
end
mean(record_Q_std(end-i,:)*HALF.sigma_t0*1e12)
subplot(1,2,2);
% ylim([-15,15]);
% grid minor;
set(gca,'FontName','Times New Roman','FontSize',12);xlim([1,Bun_num]);
subplot(1,2,1);
% grid minor;
set(gca,'FontName','Times New Roman','FontSize',12);xlim([1,Bun_num]);


figure(3);
% for i=10
% subplot(2,1,2);plot(mean(record_Q_mean(end-i:end,:))*HALF.sigma_t0*1e12,'.');hold on;
% ylabel('<\tau>  [ps]');xlabel('bunch number');
% subplot(2,1,1);plot(mean(record_Q_std(end-i:end,:))*HALF.sigma_t0*1e12,'.');hold on;
% ylabel('\sigma_{\tau}  [ps]');xlabel('bunch number');
% end
for i=40000:100:44000
subplot(1,2,2);plot(abs(fft(record_Q_mean(end-i,:)*HALF.sigma_t0*1e12)),'.');hold on;
ylabel('<\tau>  FFT Magnitude');xlabel('Coupled-Bunch Mode Number');
subplot(1,2,1);plot(abs(fft(record_Q_std(end-i,:)*HALF.sigma_t0*1e12)),'.');hold on;
ylabel('\sigma_{\tau}  [ps]');xlabel('bunch number');
end
mean(record_Q_std(end-i,:)*HALF.sigma_t0*1e12)
subplot(1,2,2);
% ylim([-15,15]);
% grid minor;
set(gca,'FontName','Times New Roman','FontSize',12);xlim([1,Bun_num]);
subplot(1,2,1);
% grid minor;
set(gca,'FontName','Times New Roman','FontSize',12);xlim([1,Bun_num]);
%%  统计作密度分布图  Dq = 0.4;
Dq = 0.5;
Q_min = min(Q); 
tau_min = gather(Q_min)*HALF.sigma_t0;
Q_new = round((Q - Q_min) *(1/Dq));
binnum=max(max(Q_new))+1; binnum=gather(binnum);
bin_num_q=sum(BinNumCalZ(binnum,Q_new));
bin_num_q=reshape(bin_num_q,binnum,Bun_num);
figure(4);
bin_i = [1:10:h];
colorset =[1 0 0;0 1 0;0 0 1;0 0 0.5;1 0.5 0.5; 0.5 1 0.5;0.5 0.5 1;1 0 1;0 1 1;1 1 0];
for i =1:length(bin_i)
    bin_range = (1:binnum)*Dq*HALF.sigma_t0+tau_min(bin_i(i));
    plot((bin_range'-Dq*HALF.sigma_t0)*1e12,bin_num_q(:,bin_i(i))/1e4/5,'-','Color',colorset(i,:),'Linewidth',2);hold on; % rs
end
ylabel('norm.density ');xlabel('\tau [ps]'); 
xlim([-150,150]);
set(gca,'FontName','Times New Roman','FontSize',14);
% %% 画出Vg电压
% figure(656)
% plot(abs(Vb_hc_track_record)/1e3/mean(abs(Vb_hc_track_record(1:50000))/1e3));title('Harmonic Cavity');hold on;
% plot(angle(Vb_hc_track_record)/pi*180/mean(angle(Vb_hc_track_record(1:50000))/pi*180));hold on;
% plot(Nturns,record_P_mean(:,1)/7+1);hold on;
% legend('Amplitude','Phase','<\delta>');
% xlabel('Turns');ylabel('Norm.Amp. [a.u.]');xlim([0,10e4]);
% %% 主腔发射机功率  PI.Ig_track 
% % 发射机电流
% figure(666)
% subplot(2,1,1);
% plot(abs(PI.Ig_track));hold on
% subplot(2,1,2);
% plot(angle(PI.Ig_track));hold on;
% %% 发射机功率
% Q_mc_0 = 5e8;R_mc_0 = Q_mc_0*44.5; betacoupling = Q_mc_0/HALF.Q_mc-1;% main cavity param.
% figure(667)
% Pg_mc = 1/8*HALF.Ig_track.^2*R_mc_0/betacoupling*4; % *4 due to similar to Ib
% plot(abs(Pg_mc)/1e3);hold on;ylabel('P_g  [kW]');

%% FFT分析振荡频率 Q
% for i=1:1
% mean_q = record_Q_mean(5000*(i)+1:5000*(i+1),1)'; % 1 first bunch
% mean_q = mean_q-mean(mean_q); % 去DC
% n_turns= length(mean_q);
% % 统计质心的振荡频率
% % 注意此处是每10圈记录一次数据
% freqs = 0.00001:1/n_turns:0.5;amp = abs(fft(mean_q));
% figure(12)
% % plot(freqs/10,amp(1:length(freqs))/max(amp(1:length(freqs)))); %/10 表示每10圈记录一次数据
% % xlim([0,0.5]);
% plot((freqs/2)*(299792458/HALF.C),amp(1:length(freqs))/max(amp(1:length(freqs)))); %/10 表示每10圈记录一次数据
% % xlim([25e3,35e3]);
% hold on;
% pause(1)
% end
