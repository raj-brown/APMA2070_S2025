function [loss,gradients] = modelLoss(parameters,X,T,X0,T0,U0)
U = model(parameters,X,T);

% Calculate Residuals.
gradientsU = dlgradient(sum(U,"all"),{X,T},EnableHigherDerivatives=true);
Ux = gradientsU{1};
Ut = gradientsU{2};
Uxx = dlgradient(sum(Ux,"all"),X,EnableHigherDerivatives=true);
f = Ut + U.*Ux - (0.01./pi).*Uxx;
zeroTarget = zeros(size(f), "like", f);
lossF = mse(f, zeroTarget);

% Calculate data loss.
U0Pred = model(parameters,X0,T0);
lossU = mse(U0Pred, U0);
loss = lossF + lossU;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);

end