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
Jm      = 548e-6 ;            % [kg*m^2]

N_J = 10;
Jms     = linspace(100e-6, 1000e-6, N_J) % [kg*m^2]


% Time span
t_0     = 0; 
t_end   = 0.01;
dt      = 0.0001;
t       = linspace(t_0,t_end,(t_end-t_0)/dt+1);


% State vectors
pos = zeros(size(t,2),N_J);
vel = zeros(size(t,2),N_J);
acc = zeros(size(t,2),N_J);
curr = zeros(size(t,2),N_J);
torque = zeros(size(t,2),N_J);

% Initial conditions
pos(1,:) = 0 * ones(N_J,1);
vel(1,:) = 0 * ones(N_J,1);
acc(1,:) = 0 * ones(N_J,1);
curr(1,:) = i_max * ones(N_J,1);
torque(1,:) = i_max / k1 * ones(N_J,1);

for i = 2:length(t)
    % Get acceleration from torque
    acc(i,:) = torque(i-1,:) ./ Jms;

    % Get velocity and position update
    vel(i,:) = vel(i-1,:) + acc(i,:) * dt;
    pos(i,:) = pos(i-1,:) + vel(i,:) * dt;

    % Get new current and torques from actual velocity
    torque(i,:) = (vel(i,:)-v_max)/k2;
    curr(i,:) = k1 * torque(i,:);

end


figure(1);
hold on;
surf(Jms,t,pos)
% surf(Jms,t,vel/10)
% surf(Jms,t,acc/1000)
% surf(Jms,t,curr)
%yline(0,'k')
xlabel('Inertia [kg*m^2]')
ylabel('Time [s]')
zlabel('Amplitude []')
legend('Position [rad]','Velocity [10rad/s]','Acceleration[1000rad/s^2]','Current [A]')