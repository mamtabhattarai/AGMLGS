function [Xnew, time] = funLCGO1(X1, par)
    al = par.al;
    Bt = par.bt;
    X = X1;
    [N, D] = size(X);  % N = n (sample size), D = d (dimension)
    d1 = par.k;  % Target reduced dimension

    % Initialize P and compute initial YY
    P = ones(D, d1);  % Size D x d1
    YY = P' * X';  % Size d1 x N

    % Compute sigma and initialize matrices R and W
    sig = (1 / N^2) * norm(X)^2;
    R = evalkernel(X, X, 'rbfp', sig) + eye(N) * 10^100;  % Size N x N
    W = evalkernel(X, X, 'rbf', sig);  % Size N x N

    % Initialize Laplacian matrix L and identity matrix I
    L = diag(sum(W, 2)) - W;  % Size N x N
    I = eye(N);  % Size N x N
    S = ones(N, N);  % Size N x N

    % Start timing the operation
    tic;

    % Iteration variable
    j = 1;

    % Iterate until the stop criteria are met
    % Ensure the stop criterion is a scalar
    while (norm(P' * (X' - X' * S))^2 + al * sum(sum(S' * R .* S)) + Bt * trace(S' * L * S) >= 0) && (j <= 10)
        % Update S
        S = (YY' * YY + al * W + Bt * L) \ YY' * YY;  % Size N x N

        % Update M and solve the generalized eigenvalue problem
        M = X' * (I - S - S' + S' * S) * X;  % Size D x D
        C = X' * X;  % Size D x D
        [V, D] = eig(M, C);  % V is size D x D, D is size D x D

        % Sort eigenvalues and select corresponding eigenvectors
        [d, ind] = sort(diag(D));  % d is size D x 1, ind is size D x 1
        P = V(:, ind(1:d1));  % Size D x d1

        % Update YY
        YY = P' * X';  % Size d1 x N

        % Increment the iteration counter
        j = j + 1;
    end

    % End timing the operation
    time = toc;

    % Compute the new representation of X
    Xnew = P' * X';  % Size d1 x N
    Xnew = Xnew';  % Size N x d1
end
