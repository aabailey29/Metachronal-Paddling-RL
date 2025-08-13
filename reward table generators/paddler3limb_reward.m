%Hotdog Tests 3 limbs

%Set parameters
params.L1 = 1.25; %Attach limb set 1 at this point in x space
params.L2 = 5; %Attach limb set 2 at this point in x space
params.L3 = 8.75; %Attach limb set 3 at this point in x space
params.Lp1 = 3; %Length of paddle set 1
params.Lp2 = 3; %Length of paddle set 2
params.Lp3 = 3; %Length of paddle set 3
num_states = 1331; %This is decided based on how we discretize the angles
num_paddle_states = 11;
params.dtheta = pi/2/(num_paddle_states-1); %theta distance between states
num_actions = 26; %All possible combinations from {0,1,-1} x {0,1,-1} X {0,1,-1} 
integrator = "3pt-gq";

% checking paddles 1&2, 2&3, then 1&3
% paddle_spc = [params.L2 - params.L1 params.L3 - params.L2 params.L3 - params.L1];
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
action_list = [[0,0,1]; [0,0,-1]; [0,1,0]; [0,1,1]; [0,1,-1]; [0,-1,0]; [0,-1,1]; [0,-1,-1]
    [1,0,0]; [1,0,1]; [1,0,-1]; [1,1,0]; [1,1,1]; [1,1,-1]; [1,-1,0]; [1,-1,1]; [1,-1,-1]
    [-1,0,0]; [-1,0,1]; [-1,0,-1]; [-1,1,0]; [-1,1,1]; [-1,1,-1]; [-1,-1,0]; [-1,-1,1]; [-1,-1,-1]];


%Initialize reward table for swim speed
R = zeros(num_states,num_actions);

% Initialize reward table for efficiency
E = zeros(num_states,num_actions);


%Make a list of states where paddles are crossing
bad_states = [];
bad_state_arr = [];

for i = 0:num_states-1
    %Convert state number to starting configuration [state1, state2, state3]
    %S = s1 + 11s2 + 121s3
    s3 = floor(i/(num_paddle_states^2));
    r = rem(i,num_paddle_states^2);
    s2 = floor(r/num_paddle_states);
    s1 = rem(r,num_paddle_states);
    paddle_pairs = [[s1 s2]; [s2 s3]];
    for m = 1:length(paddle_pairs)
    config_start = paddle_pairs(m,:);
        cross = iscrossing(d,params.Lp1,config_start,params.dtheta);
        if cross == 1 && ~(ismember(i, bad_states))
            bad_states = [bad_states; i]; %Save list of bad states where paddles cross
            bad_state_arr = [bad_state_arr; [s1 s2 s3]];
        end
    end
end



%Set reward to -999 for unallowable state-action pairs
for i = 0:num_states-1
    %Convert state number to starting configuration [state1, state2, state3]
    %S = s1 + 11s2 + 121s3
    s3 = floor(i/(num_paddle_states^2));
    r = rem(i,num_paddle_states^2);
    s2 = floor(r/num_paddle_states);
    s1 = rem(r,num_paddle_states);
    config_start = [s1 s2 s3];
    for j = 1:num_actions
        %Pick an action for each paddle from action space a = [-1, 0, 1]
        % -1 -> move paddle left by pi/20
        % 0 -> does not move paddle
        % 1 -> move paddle right by pi/20
        action = action_list(j,:);

        %Get ending configuration
        config_end = config_start + action;

        %Block movement beyond endpoints
        if config_end(1) > num_paddle_states-1 || config_end(1) < 0 || config_end(2) > num_paddle_states-1 || config_end(2) < 0 || config_end(3) > num_paddle_states-1 || config_end(3) < 0
            R(i+1,j) = -999;
            E(i+1,j) = -999;
        end

        % No crossing of paddles. Block entry into these states
        for m = 1:size(bad_state_arr,1)
            if (config_end(1) == bad_state_arr(m,1)) && (config_end(2) == bad_state_arr(m,2)) && (config_end(3) == bad_state_arr(m,3))
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
    %S = s1 + 11s2 + 121s3
    s3 = floor((i-1)/(num_paddle_states^2));
    r = rem(i-1,num_paddle_states^2);
    s2 = floor(r/num_paddle_states);
    s1 = rem(r,num_paddle_states);
    config_start = [s1 s2 s3];
    % Convert action number to action
    action = action_list(j,:);
    
    rhs=@(t,x)(paddler3limb(t,x,params,config_start,action));
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
        W2 = 0.5*(P1+P2);
        D(k) = D2(1);
        Q(k) = W2;
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

writematrix(R,'Reward_Table_HD3_paddle11_1.25-5-8.75.csv')
% writematrix(E,'Efficiency_Table_HD3_paddle11_4.25-5-5.75.csv')



%%%%%%%%%%%%%%%%%%%%%%%%% OLD CODE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Fill out reward matrix
% for i = 0:num_states-1
%     %Convert state number to starting configuration [state1, state2]
%     %S = s1 + 7s2 + 49s3
%     s3 = floor(i/(num_paddle_states^2));
%     r = rem(i,num_paddle_states^2);
%     s2 = floor(r/num_paddle_states);
%     s1 = rem(r,num_paddle_states);
%     config_start = [s1 s2 s3];
% 
%     for j = 1:num_actions
%         %Pick an action for each paddle from action space a = [-1, 0, 1]
%         % -1 -> move paddle left by pi/12
%         % 0 -> does not move paddle
%         % 1 -> move paddle right by pi/12
%         %Note that at -9pi/12, a != -1
%         %Note that at -3pi/12, a != 1
%         action = action_list(j,:);
% 
%         % Unallowed movements
%         if R(i+1,j) == -999
%             %Don't run ode45, leave as is.
%         else
%             %ODE45 Solve
%             x0 = [0 0];
%             tspan = [0, 1];
%             rhs=@(t,x)(paddler3limb(t,x,params,config_start,action));
%             [t1,x1] = ode45(rhs, tspan, x0); 
%             R(i+1,j) = x1(end,1);
%         end
% 
%     end
% end
% 
% 
% writematrix(R,'Reward_Table_HD3_paddle11_1-5-9.csv')






% for i = 0:num_states-1
%     %Convert state number to starting configuration [state1, state2, state3]
%     %S = s1 + 7s2 + 49s3
%     %S = s1 + 11s2 + 121s3
%     s3 = floor(i/(num_paddle_states^2));
%     r = rem(i,num_paddle_states^2);
%     s2 = floor(r/num_paddle_states);
%     s1 = rem(r,num_paddle_states);
%     config_start = [s1 s2 s3];
%     for k = 0:num_paddle_states
%         if ((s1 > (num_paddle_states - 1)/2) && (s2 < (num_paddle_states - 1)/2)) %Checking if paddles 1 and 2 are tilted toward each other
%             d = params.L2 - params.L1;
%             %Get reference angles for paddle 1 and 2
%             ref1 = pi/2 - (s1*params.dtheta - (num_paddle_states-1)*params.dtheta/2);
%             ref2 = s2*params.dtheta + (num_paddle_states-1)*params.dtheta/2;
% 
%             %Find x component of each paddle
%             x1 = params.Lp1 * cos(ref1);
%             x2 = params.Lp2 * cos(ref2);
%             if (x1 + x2 >= d) && ~(ismember(i, bad_states))
%                 bad_states = [bad_states; i]; %Save list of bad states where paddles cross
%                 bad_state_arr = [bad_state_arr; [s1 s2 s3]];
%             end
%         end
%         if  ((s2 > (num_paddle_states - 1)/2) && (s3 < (num_paddle_states - 1)/2))
%             d = params.L3 - params.L2;
%             %Get reference angles for paddle 2 and 3
%             ref2 = pi/2 - (s2*params.dtheta - (num_paddle_states-1)*params.dtheta/2);
%             ref3 = s3*params.dtheta + (num_paddle_states-1)*params.dtheta/2;
%             %Find x component of each paddle
%             x2 = params.Lp2 * cos(ref2);
%             x3 = params.Lp3 * cos(ref3);
%             if (x2 + x3 >= d) && ~(ismember(i, bad_states))
%                 bad_states = [bad_states; i]; %Save list of bad states where paddles cross
%                 bad_state_arr = [bad_state_arr; [s1 s2 s3]];
%             end
%         end
%         if ((s1 > (num_paddle_states - 1)/2) && (s3 < (num_paddle_states - 1)/2)) %Checking if paddles 1 and 3 are tilted toward each other
%             d = params.L3 - params.L1;
%             %Get reference angles for paddle 1 and 2
%             ref1 = pi/2 - (s1*params.dtheta - (num_paddle_states-1)*params.dtheta/2);
%             ref3 = s3*params.dtheta + (num_paddle_states-1)*params.dtheta/2;
% 
%             %Find x component of each paddle
%             x1 = params.Lp1 * cos(ref1);
%             x3 = params.Lp3 * cos(ref3);
%             if (x1 + x2 >= d) && ~(ismember(i, bad_states))
%                 bad_states = [bad_states; i]; %Save list of bad states where paddles cross
%                 bad_state_arr = [bad_state_arr; [s1 s2 s3]];
%             end
%         end
%     end
% end
