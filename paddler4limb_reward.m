%Hotdog Tests 4 limbs

%Set parameters
params.L1 = 1.75; %Attach limb set 1 at this point in x space
params.L2 = 4; %Attach limb set 2 at this point in x space
params.L3 = 6.25; %Attach limb set 3 at this point in x space
params.L4 = 8.5;
params.Lp1 = 3; %Length of paddle set 1
params.Lp2 = 3; %Length of paddle set 2
params.Lp3 = 3; %Length of paddle set 3
params.Lp4 = 3;
num_states = 14641; %This is decided based on how we discretize the angles
num_paddle_states = 11;
params.dtheta = pi/2/(num_paddle_states-1); %theta distance between states
num_actions = 80; %All possible combinations from {0,1,-1} x {0,1,-1} X {0,1,-1} x {0,1,-1}
integrator = "2pt-gq";

% Paddle spacing
d = params.L2 - params.L1;

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
action_list = setprod([0 1 -1], [0 1 -1], [0 1 -1], [0 1 -1]);
action_list(41,:) = []; %delete row of all zeros


%Initialize reward table
R = zeros(num_states,num_actions);

% Initialize reward table for efficiency
E = zeros(num_states,num_actions);

%Make a list of states where paddles are crossing
bad_states = [];
bad_state_arr = [];

for i = 0:num_states-1
    %Convert state number to starting configuration [state1, state2, state3]
    %S = s1 + 7s2 + 49s3 + 343s4
    s4 = floor(i/(num_paddle_states^3));
    rr = rem(i,num_paddle_states^3);
    s3 = floor(rr/(num_paddle_states^2));
    r = rem(rr,num_paddle_states^2);
    s2 = floor(r/num_paddle_states);
    s1 = rem(r,num_paddle_states);
    paddle_pairs = [[s1 s2]; [s2 s3]; [s3 s4]];
    for m = 1:length(paddle_pairs)
    config_start = paddle_pairs(m,:);
        cross = iscrossing(d,params.Lp1,config_start,params.dtheta);
        if cross == 1 && ~(ismember(i, bad_states))
            bad_states = [bad_states; i]; %Save list of bad states where paddles cross
            bad_state_arr = [bad_state_arr; [s1 s2 s3 s4]];
        end
    end
end

%Set reward to -999 for unallowable state-action pairs
for i = 0:num_states-1
    %Convert state number to starting configuration [state1, state2, state3]
    %S = s1 + 7s2 + 49s3 + 343s4
    s4 = floor(i/(num_paddle_states^3));
    rr = rem(i,num_paddle_states^3);
    s3 = floor(rr/(num_paddle_states^2));
    r = rem(rr,num_paddle_states^2);
    s2 = floor(r/num_paddle_states);
    s1 = rem(r,num_paddle_states);
    config_start = [s1 s2 s3 s4];
    for j = 1:num_actions
        %Pick an action for each paddle from action space a = [-1, 0, 1]
        % -1 -> move paddle left by pi/20
        % 0 -> does not move paddle
        % 1 -> move paddle right by pi/20
        action = action_list(j,:);

        %Get ending configuration
        config_end = config_start + action;

        %Block movement beyond endpoints
        if config_end(1) > num_paddle_states-1 || config_end(1) < 0 || config_end(2) > num_paddle_states-1 || config_end(2) < 0 || config_end(3) > num_paddle_states-1 || config_end(3) < 0 || config_end(4) > num_paddle_states-1 || config_end(4) < 0
            R(i+1,j) = -999;
            E(i+1,j) = -999;
        end

        % No crossing of paddles. Block entry into these states
        for m = 1:size(bad_state_arr,1)
            if (config_end(1) == bad_state_arr(m,1)) && (config_end(2) == bad_state_arr(m,2)) && (config_end(3) == bad_state_arr(m,3)) && (config_end(4) == bad_state_arr(m,4))
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
% parfor k = 1:length(K)
for k = 1:length(K)
    [i,j] = ind2sub([num_states,num_actions],K(k)); % get index of allowed state-action pair

    % Convert state number to starting configuration [state1, state2, state3]
    %S = s1 + 11s2 + 121s3 + 1331s4
    s4 = floor((i-1)/(num_paddle_states^3));
    rr = rem((i-1),num_paddle_states^3);
    s3 = floor(rr/(num_paddle_states^2));
    r = rem(rr,num_paddle_states^2);
    s2 = floor(r/num_paddle_states);
    s1 = rem(r,num_paddle_states);
    config_start = [s1 s2 s3 s4];
    % Convert action number to action
    action = action_list(j,:);
    
    rhs=@(t,x)(paddler4limb(t,x,params,config_start,action));
    if integrator == "ode45"
        %ODE45 Solve
        x0 = [0 0];
        tspan = [0, 1];
        [t1,x1] = ode45(rhs, tspan, x0); 
        D(k) = x1(end,1);
    end
    if integrator == "2pt-gq"
        x0 = [0,0];
        t0 = 0.5*(1-(1/sqrt(3)));
        t1 = 0.5*(1+(1/sqrt(3)));
        [U1,P1] = rhs(t0,x0);
        [U2,P2] = rhs(t1,x0);
        D2 = 0.5*(U1+U2);
%         W2 = 0.5* (P1+P2);
        D(k) = D2(1);
%         Q(k) = W2;
    end
    if integrator == "3pt-gq"
        x0 = [0,0];
        t0 = 0.5*(1 - (sqrt(3/5)));
        t1 = 0.5;
        t2 = 0.5*(1 + sqrt(3/5));

        [U1,P1] = rhs(t0,x0);
        [U2,P2] = rhs(t1,x0);
        [U3,P3] = rhs(t2,x0);

        D3 = 0.5*((5/9)*U1 + (8/9)*U2 + (5/9)*U3);
        W3 = 0.5*((5/9)*P1 + (8/9)*P2 + (5/9)*P3);
        D(k) = D3(1);
        Q(k) = W3;
    end
end

for m=1:length(K)
    [i,j] = ind2sub([num_states,num_actions],K(m)); 
    R(i,j) = D(m);
%     E(i,j) = D(m)*abs(D(m))/Q(m);
end



writematrix(R,'Reward_Table_HD4_paddle11_1.75-4-6.25-8.5.csv')
% writematrix(E,'Efficiency_Table_HD4_paddle11_2-4-6-8.csv)