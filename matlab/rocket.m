clc;
clear all;
close all;


% Physical constants
R = 8.314462;       % [J/K/mol] Universal gas constant
T = 25 + 273.15;    % [K] Temperature
M_N2 = 28e-3;       % [kg/mole] Molar mass of N2
g = 9.81;           % [m*s^-2] gravity
p_0 = 101325;       % [Pa] atmospheric pressure
rho_0 = 1.293;      % [kg*m^-3] air density

% Rocket parameters
m_str = 0.390;      % [kg] Mass of structure
m_gas = 0.090;      % [kg] Mass of gas

mf = m_str + m_gas; % [kg] Full mass
me = m_str;         % [kg] Empty mass
V = 0.22e-3;        % [m^3] Volume
A = (1e-3)^2;       % [m^2] Nozzle area
Cd = 0.75;          % [] Drag coeff
Ar = 0.002;         % [m^2] Rocket attack surface



% Time span
t_0     = 0; 
t_end   = 10;
dt      = 0.005;
t       = linspace(t_0,t_end,(t_end-t_0)/dt+1);

% State vectors
m_t = zeros(size(t));       % mass of fuel
a_t = zeros(size(t));       % acceleration 
v_t = zeros(size(t));       % velocty   
h_t = zeros(size(t));       % position
p_t = zeros(size(t));       % pressure

% Flight status
s_t = zeros(size(t));       % flight status: 0:upwards, 1:downwards, 2:landed

% Initial conditions
m_t(1) = mf-me;
v_t(1) = 0;
h_t(1) = 0;
p_t(1) = m_t(1)*R*T / (M_N2*V);



for i = 1:length(t)-1
    switch s_t(i)
        case 0
            %% Upwards flight condition
            % Exhaust velocity
            v_ex = sqrt(2*(p_t(i)-p_0)*V/m_t(i));
            % Mass difference over dt 
            dm = m_t(i)/V * A * v_ex * dt;
            % Force equation  
            Fnet = (p_t(i)-p_0)*A + dm*v_ex - (m_t(i)+me)*g - sign(v_t(i))*0.5*Cd*rho_0*Ar*v_t(i)^2;
            a_t(i) = Fnet / (m_t(i)+me);

            % Future states
            m_t(i+1) = m_t(i) - dm;
            v_t(i+1) = v_t(i) + a_t(i)*dt;
            h_t(i+1) = h_t(i) + v_t(i)*dt + 0.5*a_t(i)*dt^2;
            p_t(i+1) = m_t(i+1)*R*T / (M_N2*V);
        
        case 1
            %% Downwards flight condition
            % Force equation  
            Fnet =  - (m_t(i)+me)*g - sign(v_t(i))*0.5*Cd*rho_0*Ar*v_t(i)^2; 
            a_t(i) = Fnet / (m_t(i)+me);

            % Future states
            m_t(i+1) = 0;
            v_t(i+1) = v_t(i) + a_t(i)*dt;
            h_t(i+1) = h_t(i) + v_t(i)*dt + 0.5*a_t(i)*dt^2;
            p_t(i+1) = p_0;
        case 2
            %% Landed
            h_t(i+1) = -0.1;
            p_t(i+1) = p_0;

    end

    if h_t(i) <= -0.1
        s_t(i+1) = 2; %Landed
    elseif p_t(i+1) <= p_0
        s_t(i+1) = 1; %Falling
    end

end


figure(1);
subplot(2,1,1)
hold on;
plot(t,h_t);
plot(t,v_t);
plot(t,a_t);
yline(0,'k');
xlabel('Time [s]')
ylabel('Amplitude []')
legend('Height [m]','Velocity [m/s]','Acceleration[m/s^2]')

subplot(2,1,2)
hold on;
plot(t,m_t);
plot(t,p_t*1e-5*1e-3);
plot(t,s_t/6);
yline(0,'k');
xlabel('Time [s]')
ylabel('Amplitude []')
legend('Mass [kg]','Pressure [kbar]','Flight Stage')