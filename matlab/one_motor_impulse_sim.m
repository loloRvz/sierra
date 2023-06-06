clc;
clear all;
close all;

% Max values
i_max   = 2;                % [A]
v_max   = 77 /60*2*pi;      % [rad/s]

% Motor perf constants
k1      = 1;                % [A/N/m]       formula: i(t) = k1 * T(t)    
k2      = - v_max/2.1;      % [rad/s/N/m]   forumla: v(t) = k2 * T(t) + v_max       

% Physical constants
Jm      = 278e-6 ;            % [kg*m^2]

% Time span
t_0     = 0; 
t_end   = 0.01;
dt      = 0.0001;
t       = linspace(t_0,t_end,(t_end-t_0)/dt+1);


% State vectors
pos = zeros(size(t));
vel = zeros(size(t));
acc = zeros(size(t));
curr = zeros(size(t));
torque = zeros(size(t));

% Initial conditions
pos(1) = 0;
vel(1) = 0;
acc(1) = 0;
curr(1) = i_max;
torque(1) = i_max / k1;

for i = 2:length(t)
    % Get acceleration from torque
    acc(i) = torque(i-1) / Jm;

    % Get velocity and position update
    vel(i) = vel(i-1) + acc(i) * dt;
    pos(i) = pos(i-1) + vel(i) * dt;

    % Get new current and torques from actual velocity
    torque(i) = (vel(i)-v_max)/k2;
    curr(i) = k1 * torque(i);

end


figure(1);
hold on;
plot(t,pos*10,'LineWidth',2)
plot(t,vel/10,'LineWidth',2)
plot(t,acc/1000,'LineWidth',2)
plot(t,curr,'LineWidth',2)
yline(0,'k')
xlabel('Time [s]')
ylabel('Amplitude []')
legend('Position [10rad]','Velocity [10rad/s]','Acceleration[1000rad/s^2]','Current [A]')