function[Pre_train,Pre_test,time]= FME(X1,Y1,X2,Y2, gamma, mu)
    % Inputs:
    % Y - binary label matrix of size m x c
    % X - sample set of size d x m (each column is a sample)
    % gamma - regularization parameter
    % sigma - parameter for the Gaussian kernel
    % mu - regularization parameter
   tic;
    sigma=1;
    X=X1';
    Y=Y1;
    % Dimensions
    [d, m] = size(X);
    [m_y, c] = size(Y);
    
    % Check dimension consistency
    if m ~= m_y
        error('Number of samples in X and Y must be the same.');
    end
    
    % Compute H using the Gaussian kernel
    H = computeH(X, sigma);
    
    % Construct the diagonal matrix U
    U = diag(any(Y, 2)); % U(i, i) = 1 if Y(i, :) has any non-zero entry, else 0
    
    % Compute X_c
    Xc = X * H; 
    
    % Step 1: Compute N
    N = Xc' * ((gamma * (Xc * Xc') + eye(d)) \ Xc);
    
    % Step 2: Compute the graph Laplacian matrix M
    W = pdist2(X', X'); % Compute the distance matrix
    sigma_W = mean(W(:)); % Define a suitable sigma for Gaussian kernel
    W = exp(-W.^2 / (2 * sigma_W^2)); % Gaussian similarity function
    D = diag(sum(W, 2)); % Degree matrix
    M = D - W; % Graph Laplacian
    
    % Step 3: Compute the optimal F using equation (13)
    % Ensure dimensions match: (m x m) + (m x m) + (m x m) - (m x m) * (m x m)
    F = (U + M + mu * gamma * H - mu * gamma^2 * N) \ (U * Y);
    
    % Step 4: Compute the optimal projection matrix W using equation (9)
    % Ensure dimensions match: (d x d) * (d x m)
    A = (gamma * (gamma * X * H * X' + eye(d)) \ (X * H));
    W = A * F; % W should be (d x c)
    
    % Step 5: Compute the parameter b
    % Ensure dimensions match for b: (c x 1)
    b = 1 / m * (F' * ones(m, 1) - W' * X * ones(m, 1));
    
    % Display results
    Pre_train=W'*X1';
    Pre_test=W'*X2';
   Pre_train=  assignLabelsToHighestValueRowwise(Pre_train');
    Pre_test=assignLabelsToHighestValueRowwise(Pre_test');
   time=toc; 
%     disp('Optimal F:');
%     disp(F);
%     disp('Optimal W:');
%     disp(W);
%     disp('Parameter b:');
%     disp(b);
end

function H = computeH(X, sigma)
    % Inputs:
    % X - sample set of size d x m (each column is a sample)
    % sigma - parameter for the Gaussian kernel
    
    % Compute pairwise squared Euclidean distances
    dist = pdist2(X', X').^2;
    
    % Compute Gaussian kernel matrix
    H = exp(-dist / (2 * sigma^2));
    
    % Ensure H is symmetric (it should be already, but this is a safeguard)
    H = (H + H') / 2;
end
