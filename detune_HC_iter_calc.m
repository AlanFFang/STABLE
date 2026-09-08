function [fre_detRQ, S] = detune_HC_iter_calc(Ib,n_hc,C,h,U0,V_mc,R_hc,Q_hc,sigma_t0,sigma_e0,alpha_c,E0,nr_equi_iters)
%DETUNE_HC_ITER_CALC  Self-consistent passive-HHC detuning (iterated).
%
%   [fre_detRQ,S] = detune_HC_iter_calc(Ib,n_hc,C,h,U0,V_mc,R_hc,Q_hc, ...
%                       sigma_t0,sigma_e0,alpha_c,E0,nr_equi_iters)
%
%   MATLAB port of the Python scheme
%       for _ in range(nr_equi_iters):
%           detune_angle = equi.calc_detune_for_fixed_harmonic_voltage(
%               peak_harm_volt=voltage, Rs=...shunt_impedance)
%           equi.impedance_sources[-1].detune_angle = detune_angle
%           equi.calc_longitudinal_equilibrium(niter=10000,beta=0.1,tol=1e-8,m=3)
%   plus an outer convergence check: the loop runs until BOTH the detuning
%   and the rms bunch length stop changing (rel. tol 1e-4), not a fixed count.
%
%   The old detune_HC_calc assumed a bunch form factor of 1 at the harmonic,
%   i.e. harmonic beam current = 2*Ib. In reality the beam-loading voltage of
%   the passive harmonic cavity is
%       |V_hc| = 2*Ib*F(n_hc)*R_hc*cos(psi),
%   where F(n_hc) is the form factor of the LENGTHENED bunch. Since the bunch
%   shape depends on the HHC voltage and the HHC voltage depends on the bunch
%   shape, detuning and equilibrium distribution must be iterated:
%
%   outer loop (nr_equi_iters):
%     1) psi = detune_for_fixed_hc_voltage: choose the detuning angle so that
%        the passively induced voltage amplitude equals the TARGET harmonic
%        voltage (flat-potential requirement, cf. R_fp_f_fp_Calc.m)
%            V_hc_target = k_fp*V_mc, cos(psi) = V_hc_target/(2*Ib*F*R_hc)
%        using F of the CURRENT distribution;
%     2) longitudinal_equilibrium: re-solve the Haissinski equilibrium in the
%        total voltage (main RF + HHC beam loading with that psi) by a damped
%        fixed-point iteration (inner: niter/beta/tol).
%
%   Output fre_detRQ keeps the sign convention and rounding of
%   detune_HC_calc, so it drops into machine() unchanged.

if nargin < 13
    nr_equi_iters = 50;                 % max outer iterations (early exit on convergence)
end
tol_conv = 1e-4;                        % outer convergence tolerance (relative)

cspeed = 299792458;
T0     = C/cspeed;                      % revolution period [s] (beta ~ 1)
f_rf   = h/T0;
w_rf   = 2*pi*f_rf;
fre_hc = n_hc*f_rf;                     % harmonic frequency [Hz]

% ---------- flat-potential voltage requirement (target) --------------------
n2     = n_hc^2;
k_fp   = sqrt(1/n2 - 1/(n2-1)*(U0/V_mc)^2);   % V_hc target = k_fp*V_mc
V_hc_t = k_fp*V_mc;

fprintf('detune_HC_iter_calc: V_hc target = %.1f kV (k_fp = %.4f)\n',V_hc_t/1e3,k_fp);
fprintf('   reference, F=1 (old detune_HC_calc): %.1f kHz\n', ...
    tan(pi-acos(V_hc_t/(-2*Ib*R_hc)))/(2*Q_hc)*fre_hc/1e3);

% ---------- Haissinski grid over one RF bucket -----------------------------
Ntap = 2*4096+1;                        % odd -> tau = 0 lies on the grid
T_rf = 1/f_rf;
tau  = linspace(-0.499*T_rf,0.499*T_rf,Ntap)';   % [s]
D_h  = E0*alpha_c*T0*sigma_e0^2;        % Haissinski denominator [V*s]

% ---------- main cavity: fixed voltage, phase as in machine.m ---------------
% machine.m: fais_mc_whc = pi - asin((U0+2*I0*R_hc*cos(det_angle_hc)^2)/V_mc)
%   -> the MC synchronous phase compensates the HHC beam-loading energy loss
%      2*Ib*F*R_hc*cos(psi)^2 (here with the actual form factor F); the
%      pi-asin convention of machine.m maps to asin (Q1, rising flank) here.
% (fais_s is recomputed every outer iteration, see step 2 below)

% ---------- initial guess: natural Gaussian --------------------------------
sig_t = sigma_t0;
lam   = exp(-0.5*(tau/sig_t).^2);
lam   = lam/trapz(tau,lam);

psi = NaN;  v = [];  v_hc = [];  fre_hz = NaN;
fre_prev = Inf;  sig_prev = Inf;  conv = false;
for it_o = 1:nr_equi_iters
    % ---- 1) detune angle for fixed harmonic voltage (outer step) ----------
    Fn_c = form_factor(lam,tau,n_hc*w_rf);
    psi  = detune_for_fixed_hc_voltage(V_hc_t,R_hc,Ib,Fn_c);

    % ---- 2) main-cavity phase: compensate the HHC loss (machine.m style) --
    Vhc_loss = 2*Ib*abs(Fn_c)*R_hc*cos(psi)^2;    % HHC loss seen by the bunch
    fais_s   = asin(min(1,(U0+Vhc_loss)/V_mc));   % Q1 form of fais_mc_whc

    % ---- 3) longitudinal equilibrium with the updated detuning (inner) ----
    [lam,v,v_hc] = longitudinal_equilibrium(lam,psi, ...
        tau,D_h,w_rf,n_hc,V_mc,U0,fais_s,Ib,R_hc);

    sig_t = sqrt(rms_width(lam,tau));
    fre_hz = tan(psi)/(2*Q_hc)*fre_hc;
    fprintf('outer iter %2d: sigma_t = %6.1f ps, F(n) = %.4f, psi = %6.2f deg, detune = %8.1f kHz\n',...
        it_o, sig_t*1e12, abs(Fn_c), psi*180/pi, fre_hz/1e3);

    % ---- 4) outer convergence check ----------------------------------------
    if it_o > 1
        d_fre = abs(fre_hz-fre_prev)/max(abs(fre_hz),1);   % guards fre_hz = 0
        d_sig = abs(sig_t-sig_prev)/sig_t;
        if d_fre < tol_conv && d_sig < tol_conv
            conv = true;
            fprintf('   outer loop converged (d_fre = %.1e, d_sigma = %.1e)\n', d_fre, d_sig);
            break;
        end
    end
    fre_prev = fre_hz;  sig_prev = sig_t;
end
if ~conv
    warning('detune_HC_iter_calc: outer loop did not converge within %d iterations.', nr_equi_iters);
end

fre_detRQ = round(fre_hz);

% ---------- outputs ---------------------------------------------------------
S.converged = conv;
S.iter      = it_o;
S.sigma_t   = sig_t;        % rms bunch length [s]
S.Fn        = Fn_c;         % complex form factor at n_hc
S.psi       = psi;          % detuning angle [rad]
S.fre_hz    = fre_hz;       % detuning [Hz]
S.V_hc_t    = V_hc_t;       % target (and delivered) HHC voltage [V]
S.k_fp      = k_fp;
S.fais_s    = fais_s;       % main RF synchronous phase [rad]
S.tau       = tau;          % grid [s]
S.lambda    = lam;          % converged bunch distribution
S.v         = v;            % total voltage [V]
S.v_hc      = v_hc;

try % plotting may be unavailable in -batch / headless runs
% figure('Name','detune_HC_iter_calc','Color','w');
% subplot(2,1,1);
% plot(tau*1e12,v/1e3,'b','LineWidth',1.5); hold on;
% plot(tau*1e12,U0/1e3*ones(size(tau)),'k--');
% grid on; xlabel('\tau [ps]'); ylabel('V_{tot} [kV]');
% title(sprintf('self-consistent: #sigma_t = %.1f ps, F = %.3f, detune = %.1f kHz',...
%     sig_t*1e12,abs(Fn_c),fre_hz/1e3));
% subplot(2,1,2);
% plot(tau*1e12,lam*1e-12,'r','LineWidth',1.5); grid on;
% xlabel('\tau [ps]'); ylabel('\lambda(\tau) [1/s]');
% catch
end
end

% ===========================================================================
function psi = detune_for_fixed_hc_voltage(V_hc_peak,Rs,Ib,Fn_c)
%DETUNE_FOR_FIXED_HC_VOLTAGE  detuning angle so that the induced voltage of the
% passive HHC reaches the required peak harmonic voltage:
%   |V_hc| = 2*Ib*|F|*Rs*cos(psi) = V_hc_peak
% psi in [0,pi/2); convention of machine.m / detune_HC_calc:
% phih = pi - psi, fre_shift = tan(pi-phih)/(2Q)*fre_hc = tan(psi)/(2Q)*fre_hc.
carg = V_hc_peak/(2*Ib*abs(Fn_c)*Rs);
if carg > 1
    warning(['detune_HC_iter_calc: target HHC voltage %.1f kV exceeds the max ' ...
             'inducible 2*Ib*F*Rs = %.1f kV (F = %.3f). Clamping psi = 0 (on-resonance).'], ...
             V_hc_peak/1e3, 2*Ib*abs(Fn_c)*Rs/1e3, abs(Fn_c));
    psi = 0;
else
    psi = acos(min(carg,1));            % detuning angle, psi = pi - phih
end
end

% ===========================================================================
function [lam,v,v_hc] = longitudinal_equilibrium(lam,psi, ...
    tau,D_h,w_rf,n_hc,V_mc,U0,fais_s,Ib,R_hc)
%LONGITUDINAL_EQUILIBRIUM  damped fixed-point solution of the Haissinski
% equation. Main cavity fixed: v_mc = V_mc*sin(w_rf*tau+fais_s). The HHC
% beam-loading voltage is impedance-determined:
%   |V_hc| = 2*Ib*F(n_hc)*R_hc*cos(psi), phase follows the bunch's own
%   harmonic current (angle of F) and the detuning angle psi.
% Inner iteration: niter / beta / tol as in the Python reference.
niter = 10000;  beta = 0.1;  tol = 1e-8;
n_harm_w = n_hc*w_rf;
v = [];  v_hc = [];
for it_i = 1:niter
    Fn_c = form_factor(lam,tau,n_harm_w);
    Vamp = 2*Ib*abs(Fn_c)*R_hc*cos(psi);
    % overall minus: at the bunch v_hc ~ -Vamp*cos(psi) < 0 (energy loss to
    % the cavity) with positive curvature +(nw)^2*Vamp*cos(psi) and slope
    % -nw*Vamp*sin(psi), i.e. both the curvature and the linear focusing of
    % the main RF are reduced -> bunch lengthening.
    % The waveform phase is referenced to the bucket (theta = 0), NOT to the
    % moving centroid: a centroid-tracking lock is an undamped slow mode of
    % the static solve (drifts the bunch long) and disagrees with tracking.
    v_hc = -Vamp*cos(n_harm_w*tau - psi);
    v     = V_mc*sin(w_rf*tau+fais_s) + v_hc;

    W    = cumtrapz(tau,v-U0);  W = W - min(W);   % anchor at the bucket bottom
    lamN = exp(-W/D_h);
    lamN = lamN/trapz(tau,lamN);

    err  = max(abs(lamN-lam))/max(lamN);
    lam  = (1-beta)*lam + beta*lamN;               % damped update
    lam  = lam/trapz(tau,lam);
    if err < tol, break; end
end
end

% ===========================================================================
function Fn_c = form_factor(lam,tau,kw)
%FORM_FACTOR  complex bunch form factor at harmonic kw: F = <exp(i k tau)>
nrm = trapz(tau,lam);
Fn_c = trapz(tau,lam.*exp(1i*kw*tau))/nrm;
end

% =========================================================================**
function s = rms_width(lam,tau)
nrm = trapz(tau,lam);
tc  = trapz(tau,lam.*tau)/nrm;
s   = trapz(tau,lam.*(tau-tc).^2)/nrm;
end
