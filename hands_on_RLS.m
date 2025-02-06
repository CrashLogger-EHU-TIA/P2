function hands_on_RLS %[outputArg1,outputArg2] = hands_on_RLS(inputArg1,inputArg2)
%HANDS_ON_RLS Summary of this function goes here
%   Detailed explanation goes here

clear all;
close all;
clc;

load Boston.mat;

var_names = Boston.Properties.VariableNames

X = Boston.lstat;
Y = Boston.medv;

mdl = fitlm(X,Y,"VarNames",var_names([end-1 end]))

figure("Name","Auto Plot")
plot(mdl)

ci = coefCI(mdl, 0.05)

% Same thing but by hand, and not a very good approximation past 2 decimal
% places
CI_beta = [mdl.Coefficients.Estimate-1.96*mdl.Coefficients.SE, mdl.Coefficients.Estimate+1.96*mdl.Coefficients.SE]

[ypred, y_ci] = predict(mdl, [5; 10; 15], "Prediction", "Observation")


figure("Name", "Manual Plot")
subplot(1,3,1)

plot_pred = predict(mdl, X);
scatter(X,Y)
hold on;
plot(X,plot_pred);
hold off;
title("Modelo de regresión lineal");
xlabel("lstat");
ylabel("medv")

subplot(1,3,2)
plotResiduals(mdl, "fitted", "Marker","o")
title("Grafico de residuos");
xlabel("Fitted")
ylabel("Residuos")

subplot(1,3,3)
plotResiduals(mdl, "fitted", "Marker","o", "ResidualType","studentized")
title("Grafico de residuos studentizados");
xlabel("Fitted")
ylabel("Residuos studentizados")

figure("Name","Leverage");
plotDiagnostics(mdl, "leverage", "Marker","o", "MarkerEdgeColor","k")
xlabel("Line Index");
ylabel("Leverage")

figure(4);
plot(mdl.Diagnostics.Leverage, mdl.Residuals.Studentized, 'k.')
xlabel("Leverage");
ylabel("Residuos studentizados")

end