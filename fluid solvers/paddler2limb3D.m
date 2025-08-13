
%2D HOTDOG SWIM RIGID PADDLES


function [Ut, Pt] = paddler2limb3D(t,x,params,config_start,action)
    
    

    %Get paddle positions based on config_start
    state1 = config_start(1);
    state2 = config_start(2);
    
    %Get actions
    action1 = action(1);
    action2 = action(2);


    %Hotdog body
   %
    Lbody = 10;        % body length
    Lp    = params.Lp1;         % paddle length 
    Nth   = 20;        % number of points around the circumference
    ds    = 2*pi/Nth;  % mesh point spacing
    
    Np = round(Lp/ds); % points along the paddle, not including the attachment point 
    sp = linspace(0,Lp,Np+1)';
    
    
    % mesh of the cylinder (not the points) -- in the vertical direction
    %
    [X0,Y0,Z0,Z1,Z2,Xwg,Ywg,Zwg]=cylindermesh_roundtip_grid(Lbody,Nth);
    
    % get the points unmeshed
    %
    [Xw,Xe1,Xe2,dAw,dAe]=cylindermesh_roundtip(Lbody,Nth);
    Nwall = length(Xw);
    Nbot  = length(Xe1);
    Ntop  = length(Xe2);
    Xbody = [Xw; Xe1; Xe2]; % put all the points gether
    
        
    %LEGS
    L1 = params.L1; %Attach limb set 1 at this point in x space (between -6,6)
    L2 = params.L2; %Attach limb set 2 at this point in x space
    Zbase = [L1 L2];
    ds = 2*pi/Nth; %paddle point spacing
    epsilon = 1.2*ds; %Reg stokes 
    mu=1;
    dtheta = params.dtheta;


    %state*pi/12 - 9pi/12 = angle
    %Parametrize state-action pair as a function of time
    
    %Set 1 (leftmost paddles)
    if action1 == 1
        theta1 = @(t)((-state1*dtheta - 12*pi/16) - (dtheta)*t); %Angle of first set of paddles as a function of t
        theta1_dot = @(t)(-dtheta); %Time derivative of angle of paddle set 1
    elseif action1 == -1
        theta1 = @(t)((-state1*dtheta - 12*pi/16) + (dtheta)*t); %Angle of first set of paddles as a function of t
        theta1_dot = @(t)(dtheta); %Time derivative of angle of paddle set 1
    elseif action1 == 0
        theta1 = @(t)((-state1*dtheta - 12*pi/16) + 0*t); %Angle of first set of paddles as a function of t
        theta1_dot = @(t)(0); %Time derivative of angle of paddle set 1
    else
        disp('Not an allowable action.')
        return
    end
    
    %Set 2 (rightmost paddles)
    if action2 == 1
        theta2 = @(t)((-state2*dtheta - 12*pi/16) - (dtheta)*t); %Angle of second set of paddles as a function of t
        theta2_dot = @(t)(-dtheta); %Time derivative of angle of paddle set 2
    elseif action2 == -1
        theta2 = @(t)((-state2*dtheta - 12*pi/16) + (dtheta)*t); %Angle of second set of paddles as a function of t
        theta2_dot = @(t)(dtheta); %Time derivative of angle of paddle set 2
    elseif action2 == 0
        theta2 = @(t)((-state2*dtheta - 12*pi/16) + 0*t); %Angle of second set of paddles as a function of t
        theta2_dot = @(t)(0); %Time derivative of angle of paddle set 2
    else
        disp('Not an allowable action.')
        return
    end
    
    XL1 = [-1 + sp*cos(theta1(t)), 0*sp, Zbase(1) + sp*sin(theta1(t))];
    XL2 = [-1 + sp*cos(theta2(t)), 0*sp, Zbase(2) + sp*sin(theta2(t))];
      
    UL1 = [-sp*sin(theta1(t))*theta1_dot(t), 0*sp, sp*cos(theta1(t))*theta1_dot(t)];
    UL2 = [-sp*sin(theta2(t))*theta2_dot(t), 0*sp, sp*cos(theta2(t))*theta2_dot(t)];
    
    % limbs on the other size
    %
    XL1m = XL1; 
    XL1m(:,1) = -XL1m(:,1);
    XL2m = XL2; 
    XL2m(:,1) = -XL2m(:,1);
    
    UL1m = UL1;
    UL1m(:,1) = -UL1m(:,1);
      
    UL2m = UL2;
    UL2m(:,1) = -UL2m(:,1);

    
    
    % put all the points gether
    %
    Xall = [Xbody; XL1; XL1m; XL2; XL2m];
    Up   = [0*Xbody; UL1; UL1m; UL2; UL2m];
    Nall = length(Xall);
  
  %  System to solve for the swimming velocity
  %   Xlab = Xp + XT
  %   Ulab = Up + U
  %
  %   M*F - U = Up
  %   S*F     = 0
  %

    S = [[ones(1,Nall) , zeros(1,Nall), zeros(1,Nall) ];
       [zeros(1,Nall), ones(1,Nall) , zeros(1,Nall) ];
       [zeros(1,Nall), zeros(1,Nall), ones(1,Nall)  ];
    ];
    M = form_reg_stokes_matrix_3D(Xall,epsilon,mu);
    Z = zeros(3);
  
  % big ol' matrix to invert and the RHS
  %
    A = [ [M -S'];
        [S  Z]
      ];
    rhs = [Up(:); zeros(3,1)];

    Y = A\rhs;
    Ut = Y(end-2:end); 
    F = Y(1:end-3);
    V = reshape(Up,[],1);
    
    %POW
    Pt = dot(F(:),V(:));


