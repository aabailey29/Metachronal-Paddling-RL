%Hotdog Tests 2 limbs reward table calc

%Set parameters
num_limbs = 2;
params.L1 = -1.5; %Attach limb set 1 at this point in x space
params.L2 = 1.5; %Attach limb set 2 at this point in x space
params.Lp1 = 3; %Length of paddle set 1
params.Lp2 = 3; %Length of paddle set 2
num_states = 121; %This is decided based on how we discretize the angles
num_paddle_states = sqrt(num_states);
params.dtheta = pi/2/(num_paddle_states-1); %theta distance between states
num_actions = 3^num_limbs - 1; %All possible combinations from {0,1,-1} x {0,1,-1}
integrator = "2pt-gq";

% Discretization of angles
% -3pi/4 < theta < -pi/4

% Split the range of theta into 7 states
% -9pi/12 -> 0
% -8pi/12 -> 1
% -7pi/12 -> 2
% -6pi/12 -> 3
% -5pi/12 -> 4
% -4pi/12 -> 5
% -3pi/12 -> 6


%Action list
action_list = [[1,0]; [0,1]; [1,1]; [-1,0]; [0,-1]; [-1,-1]; [1,-1]; [-1,1]];


%Initialize reward table for swim speed
R = zeros(num_states,num_actions);

% Initialize reward table for efficiency
% E = zeros(num_states,num_actions);

%initialize power table
% W = zeros(num_states,num_actions);


%Make a list of states where paddles are crossing
bad_states = [];
bad_state_arr = [];

for i = 0:num_states-1
    %Convert state number to starting configuration [state1, state2]
    %S = s1 + 11s2
    s1 = rem(i,num_paddle_states);
    s2 = (i-s1)/num_paddle_states;
    config_start = [s1 s2];
    d = params.L2 - params.L1;
    cross = iscrossing(d,params.Lp1,config_start,params.dtheta);
    if cross == 1 && ~(ismember(i, bad_states))
        bad_states = [bad_states; i]; %Save list of bad states where paddles cross
        bad_state_arr = [bad_state_arr; [s1 s2]];
    end
end


%Set reward to -999 for unallowable state-action pairs
for i = 0:num_states-1
    %Convert state number to starting configuration [state1, state2]
    %S = s1 + 11s2
    s1 = rem(i,num_paddle_states);
    s2 = (i-s1)/num_paddle_states;
    config_start = [s1 s2];
    for j = 1:num_actions
        %Pick an action for each paddle from action space a = [-1, 0, 1]
        % -1 -> move paddle left by pi/20
        % 0 -> does not move paddle
        % 1 -> move paddle right by pi/20
        action = action_list(j,:);

        %Get ending configuration
        config_end = config_start + action;

        %Block movement beyond endpoints
        if config_end(1) > num_paddle_states-1 || config_end(1) < 0 || config_end(2) > num_paddle_states-1 || config_end(2) < 0
            R(i+1,j) = -999;
            E(i+1,j) = -999;
        end

        % No crossing of paddles. Block entry into these states
        for m = 1:size(bad_state_arr,1)
            if (config_end(1) == bad_state_arr(m,1)) && (config_end(2) == bad_state_arr(m,2))
                R(i+1,j) = -999;
                E(i+1,j) = -999;
            end
        end

        %Block out entire row for bad states
        for n = 1:length(bad_states)
                R(bad_states(n)+1,:) = -999;
                E(bad_states(n)+1,:) = -999;
        end

    end
end

% Get indices of allowed state-action pairs in reward
P = R + 999;
K = find(P);
D = zeros(1,length(K));
Q = zeros(1,length(K));
% 
% calculate the swim speed for allowed states in PARALLEL OHHHH SHITTTT
% parfor k = 1:length(K) can't use this until I redownload parallel
% computing toolbox >:(
for k = 1:length(K)
    [i,j] = ind2sub([num_states,num_actions],K(k)); % get index of allowed state-action pair

    % Convert state number to starting configuration [state1, state2]
    %S = s1 + 11s2
    s1 = rem(i-1,num_paddle_states);
    s2 = (i-1-s1)/num_paddle_states;
    config_start = [s1 s2]; 

    % Convert action number to action
    action = action_list(j,:);
    
    % solve for swimming speed with a numerical integrator
    rhs=@(t,x)(paddler2limb3D(t,x,params,config_start,action));
    
    if integrator == "ode45"
        %ODE45 Solve
        x0 = [0 0];
        tspan = [0, 1];
        [t1,x1] = ode45(rhs, tspan, x0);
        D(k) = x1(end,1); %this is not right I don't think
    end
    if integrator == "2pt-gq"
        x0 = [0,0];
        t0 = 0.5*(1-(1/sqrt(3)));
        t1 = 0.5*(1+(1/sqrt(3)));
        [U1,P1] = rhs(t0,x0);
        [U2,P2] = rhs(t1,x0);
        D2 = 0.5*(U1+U2);
        W2 = 0.5*(P1+P2);
        D(k) = D2(3);
        Q(k) = W2;
    end
    if integrator == "3pt-gq"
       t0 = 0.5*(1 - (sqrt(3/5)));
        t1 = 0.5;
        t2 = 0.5*(1 + sqrt(3/5));

        [U1,P1] = rhs(t0,x0);
        [U2,P2] = rhs(t1,x0);
        [U3,P3] = rhs(t2,x0);

        D3 = 0.5*((5/9)*U1 + (8/9)*U2 + (5/9)*U3);
        W3 = 0.5*((5/9)*P1 + (8/9)*P2 + (5/9)*P3);
        D(k) = D3(3);
        Q(k) = W3;
    end
end

% Fill out the swim speed reward matrix
for m=1:length(K)
    [i,j] = ind2sub([num_states,num_actions],K(m)); 
    R(i,j) = D(m);
    % E(i,j) = D(m)*abs(D(m))/Q(m);
end

% % POWER TABLE CALCULATION TO CHECK IF POWER IS POSITIVE
% for l=1:length(K)
%     [i,j] = ind2sub([num_states,num_actions],K(l));
% 
%     % Convert state number to starting configuration [state1, state2]
%     %S = s1 + 11s2
%     s1 = rem(i-1,num_paddle_states);
%     s2 = (i-1-s1)/num_paddle_states;
%     config_start = [s1 s2]; 
% 
%     % Convert action number to action
%     action = action_list(j,:);
% 
%     config_end = config_start+action;
% 
%     cycle = [config_start; config_end];
% 
%     [eff,Pt] = stroke_efficiency(cycle,num_limbs,num_paddle_states,R(i,j),params.L1,params.L2,0,0,params.Lp1);
%     P(l) = sum(Pt);
% 
% end
% 
% % Fill out the power matrix
% for m=1:length(K)
%     [i,j] = ind2sub([num_states,num_actions],K(m)); 
%     W(i,j) = P(m);
% end

% writematrix(E,'Efficiency_Table_HD2_paddle11_5-6.csv')
writematrix(R,'Reward_Table_HD2_3D_-1.5-1.5.csv')
% writematrix(W,'Power_Table_HD2_paddle11_3-7.csv')



