%2D HOTDOG SWIM RIGID PADDLES


function [Ut,Pt] = paddler4limb(t,x,params,config_start,action)
    

    %Get paddle positions based on config_start
    state1 = config_start(1);
    state2 = config_start(2);
    state3 = config_start(3);
    state4 = config_start(4);
    
    %Get actions
    action1 = action(1);
    action2 = action(2);
    action3 = action(3);
    action4 = action(4);

    %Hotdog body
    N = 101; %Number of points for straight parts
    r = 1; %radius of outer semicircles
    xU = linspace(0,10,N)'; %Make upper half of body
    xL = linspace(0,10,N)'; %Make lower half of body
    yU = 1; %y-coord of upper half
    yL = -1; %y-coord of lower half
    theta = linspace(pi/2,3*pi/2,N); %Variable for semicircles
    Lc = [r*cos(theta); r*sin(theta)]'; %Left semicircle
    Lc = Lc(2:end-1,:); %Delete endpoints to avoid overlap
    Rc = [-r*cos(theta)+10; r*sin(theta)]'; %Right semicircle
    Rc = Rc(2:end-1,:); %Delete endpoints to avoid overlap
    
    X = [xU 0*xU+yU; xL 0*xL+yL; Lc; Rc]; %Body
    
    %LEGS
    L1 = params.L1; %Attach limb set 1 at this point in x space
    L2 = params.L2; %Attach limb set 2 at this point in x space
    L3 = params.L3;
    L4 = params.L4;
    ds = 10/(N-1); %paddle point spacing
    epsilon = ds/2; %Reg stokes 
    mu=1;
    dtheta = params.dtheta;
    Lp1 = params.Lp1; %paddle length 1
    Lp2 = params.Lp2; %paddle length 2
    Lp3 = params.Lp3;
    Lp4 = params.Lp4;
    Nl1 = round(Lp1/ds); %Number of points on paddle set 1
    Nl2 = round(Lp2/ds); %Number of points of paddle set 2
    Nl3 = round(Lp3/ds);
    Nl4 = round(Lp4/ds);
    s1 = (1:Nl1)'*ds; %start s at 1 to avoid overlap points
    s2 = (1:Nl2)'*ds;
    s3 = (1:Nl3)'*ds;
    s4 = (1:Nl4)'*ds;

    %state*pi/12 - 9pi/12 = angle
    %Parametrize state-action pair as a function of time
    
    %Set 1 (leftmost paddles)
    if action1 == 1
        theta1 = @(t)((state1*dtheta - 12*pi/16) + (dtheta)*t); %Angle of first set of paddles as a function of t
        theta1_dot = @(t)(dtheta); %Time derivative of angle of paddle set 1
    elseif action1 == -1
        theta1 = @(t)((state1*dtheta - 12*pi/16) - (dtheta)*t); %Angle of first set of paddles as a function of t
        theta1_dot = @(t)(-dtheta); %Time derivative of angle of paddle set 1
    elseif action1 == 0
        theta1 = @(t)((state1*dtheta - 12*pi/16) + 0*t); %Angle of first set of paddles as a function of t
        theta1_dot = @(t)(0); %Time derivative of angle of paddle set 1
    else
        disp('Not an allowable action.')
        return
    end
    
    %Set 2 (middle paddles)
    if action2 == 1
        theta2 = @(t)((state2*dtheta - 12*pi/16) + (dtheta)*t); %Angle of second set of paddles as a function of t
        theta2_dot = @(t)(dtheta); %Time derivative of angle of paddle set 2
    elseif action2 == -1
        theta2 = @(t)((state2*dtheta - 12*pi/16) - (dtheta)*t); %Angle of second set of paddles as a function of t
        theta2_dot = @(t)(-dtheta); %Time derivative of angle of paddle set 2
    elseif action2 == 0
        theta2 = @(t)((state2*dtheta - 12*pi/16) + 0*t); %Angle of second set of paddles as a function of t
        theta2_dot = @(t)(0); %Time derivative of angle of paddle set 2
    else
        disp('Not an allowable action.')
        return
    end

    %Set 3 (middle paddles)
    if action3 == 1
        theta3 = @(t)((state3*dtheta - 12*pi/16) + (dtheta)*t); %Angle of second set of paddles as a function of t
        theta3_dot = @(t)(dtheta); %Time derivative of angle of paddle set 2
    elseif action3 == -1
        theta3 = @(t)((state3*dtheta - 12*pi/16) - (dtheta)*t); %Angle of second set of paddles as a function of t
        theta3_dot = @(t)(-dtheta); %Time derivative of angle of paddle set 2
    elseif action3 == 0
        theta3 = @(t)((state3*dtheta - 12*pi/16) + 0*t); %Angle of second set of paddles as a function of t
        theta3_dot = @(t)(0); %Time derivative of angle of paddle set 2
    else
        disp('Not an allowable action.')
        return
    end

    %Set 4 (rightmost paddles)
    if action4 == 1
        theta4 = @(t)((state4*dtheta - 12*pi/16) + (dtheta)*t); %Angle of second set of paddles as a function of t
        theta4_dot = @(t)(dtheta); %Time derivative of angle of paddle set 4
    elseif action4 == -1
        theta4 = @(t)((state4*dtheta - 12*pi/16) - (dtheta)*t); %Angle of second set of paddles as a function of t
        theta4_dot = @(t)(-dtheta); %Time derivative of angle of paddle set 4
    elseif action4 == 0
        theta4 = @(t)((state4*dtheta - 12*pi/16) + 0*t); %Angle of second set of paddles as a function of t
        theta4_dot = @(t)(0); %Time derivative of angle of paddle set 4
    else
        disp('Not an allowable action.')
        return
    end
    

    %Positions on paddles through time
    Y1 = [L1+s1*cos(theta1(t)) yL+s1*sin(theta1(t)); L1+s1*cos(theta1(t)) yU-s1*sin(theta1(t))]; %Set 1
    Y2 = [L2+s2*cos(theta2(t)) yL+s2*sin(theta2(t)); L2+s2*cos(theta2(t)) yU-s2*sin(theta2(t))]; %Set 2
    Y3 = [L3+s3*cos(theta3(t)) yL+s3*sin(theta3(t)); L3+s3*cos(theta3(t)) yU-s3*sin(theta3(t))];
    Y4 = [L4+s4*cos(theta4(t)) yL+s4*sin(theta4(t)); L4+s4*cos(theta4(t)) yU-s4*sin(theta4(t))];
    HotDog = [X;Y1;Y2;Y3;Y4];
    NHD = length(HotDog);

    %Calc velocities
    V1 = [-s1*sin(theta1(t))*theta1_dot(t) s1*cos(theta1(t))*theta1_dot(t); -s1*sin(theta1(t))*theta1_dot(t) -s1*cos(theta1(t))*theta1_dot(t)];
    V2 = [-s2*sin(theta2(t))*theta2_dot(t) s2*cos(theta2(t))*theta2_dot(t); -s2*sin(theta2(t))*theta2_dot(t) -s2*cos(theta2(t))*theta2_dot(t)];
    V3 = [-s3*sin(theta3(t))*theta3_dot(t) s3*cos(theta3(t))*theta3_dot(t); -s3*sin(theta3(t))*theta3_dot(t) -s3*cos(theta3(t))*theta3_dot(t)];
    V4 = [-s4*sin(theta4(t))*theta4_dot(t) s4*cos(theta4(t))*theta4_dot(t); -s4*sin(theta4(t))*theta4_dot(t) -s4*cos(theta4(t))*theta4_dot(t)];
    V_HD = [0*X;V1;V2;V3;V4];
    
    %Form Stokes Matrix from paddle positions
    M = form_reg_stokes_matrixXX(HotDog,HotDog,epsilon,mu);

    %Form Matrix of Ones for Full System
    S = zeros(2,2*NHD);
    on = ones(1,NHD);
    S(1,1:NHD) = on;
    S(2,NHD+1:2*NHD) = on;
    
    %Form Bigass Matrix
    A = zeros(2*NHD+2);
    A(1:2*NHD,1:2*NHD) = M; %top left block
    A(2*NHD+1:2*NHD+2,1:2*NHD) = -S; %bottom left block
    A(1:2*NHD,2*NHD+1:2*NHD+2) = -S'; %top right block

    %Form RHS of Bigass System
    B = zeros(2*NHD+2,1);
    V = reshape(V_HD,[],1);
    B(1:2*NHD) = V;

    %Solve Bigass System AY=B
    Y = A\B;
    Ut = Y(end-1:end); 
    F = Y(1:end-2);
%     F = reshape(F,NHD,2);

    %Power Calculation
    Pt = dot(F(:),V(:));

    