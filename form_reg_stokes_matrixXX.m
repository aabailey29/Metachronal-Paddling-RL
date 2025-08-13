%
% form the regularized stokelets matrix which maps forces to velocity
%  the forces are stored in locations X0 and the points where the velocity
%  is evaluated are stored at points X
%
function M = form_reg_stokes_matrixXX(X,X0,epsilon,mu);

  % record the number of points involved
  %  the stokelets matrix should be of size 2*Nx by 2*Ns
  %
  Ns = size(X0,1);    % number of source points
  Nt = size(X ,1);    % number of target points
  
  % make rectangular matrices of the points for fast computation
  %
  X0m = X0(:, ones(1,Nt));
  Xm  =  X(:, ones(1,Ns));
  
  Y0m = X0(:,2*ones(1,Nt));
  Ym  =  X(:,2*ones(1,Ns));
  
  % form quadratic terms
  %
  XX = (Xm-X0m').^2;
  YY = (Ym-Y0m').^2;
  XY = (Xm-X0m').*(Ym-Y0m');

  % radius and regularized radius
  %
  R    = sqrt( XX + YY );
  Re   = sqrt( R.^2 + epsilon.^2);
  
  % some coefficients
  %
  P1 = log(Re+epsilon) - epsilon*(Re+2*epsilon)./(Re.*(Re+epsilon));
  P2 =                           (Re+2*epsilon)./(Re.*(Re+epsilon).^2);
  
  
  M = [ [-P1 + P2.*XX,       P2.*XY]; 
        [      P2.*XY, -P1 + P2.*YY]
      ];
  M = M/(4*pi*mu);