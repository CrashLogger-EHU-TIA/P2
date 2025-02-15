function hands_on_RLM

%% Base de datos Boston

load Boston.mat
close all;

var_names = Boston.Properties.VariableNames

% Ajustar modelo RLM basado en lstat, age para predecir medv

X1 = Boston.lstat;
X2 = Boston.age;
Y = Boston.medv;

mdl = fitlm([X1, X2], Y, "VarNames", var_names([13 7 14]))

% Ajustar modelo RLS basado en lstat para predecir medv, y comparar con el
% anterior modelo

X1 = Boston.lstat;
Y = Boston.medv;

mdl0 = fitlm([X1], Y, "VarNames", var_names([13 14]))

% Ajustar un modelo de RLM basado en TODOS los predictores para predecir
% medv


% El RMS Error ha bajado mucho, el R² ha subido significativamente.
% Great success!!

X = Boston{:,1:13};
Y = Boston{:,14};

mdl = fitlm([X], Y, "VarNames", var_names)

% Ajustar un modelo de RLM basado en CASI TODOS los predictores para
% predecir medv (nos saltamos AGE solo!)
selection = [1:6, 8:13];
X = Boston{:, selection};
Y = Boston{:, 14};

mdl = fitlm([X], Y, "VarNames", var_names([[selection], 14]))

% Ajustar modelo RLM basado en lstat, age para predecir medv con lstat y
% age combinados también

% Con esto vemos que el peso de age es muy bajo (Estimate) pero el pValue
% de la multiplicación nos dice que no deberíamos quitar age! No al menos
% cuando aislamos lstat y age.

X1 = Boston.lstat;
X2 = Boston.age;
X3 = X1 .* X2;
Y = Boston.medv;

% Using {} to make cell arrays

%% Ajustar modelo RP basado en lstat, lstat² para predecir medv

% Es un mejor modelo que el lineal segun el error!

X1 = Boston.lstat;
X2 = X1.^2;
Y = Boston.medv;

% Using {} to make cell arrays

mdl1 = fitlm([X1, X2], Y, "VarNames", {var_names{[13]}, 'lstat²', var_names{14}})

figure(1);
scatter(X1, Y);
title("Modelo Lineal VS Modelo Cuadrático");
xlabel("lstat")
ylabel("medv")
hold on;

[x_ord, ord] = sort(X1);

ypred0 = predict(mdl0, x_ord);
ypred1 = predict(mdl1, [x_ord, x_ord.^2]);

plot(x_ord, ypred0, LineWidth=2)
plot(x_ord, ypred1, LineWidth=2)

hold off;

% Analizar residuos
figure(2)

subplot(1,2,1);
plotResiduals(mdl1, "fitted", "Marker","o");
xlabel("Fitted");
ylabel("Residuos");

subplot(1,2,2);
plotResiduals(mdl1, "fitted", "Marker","o", "ResidualType","studentized");
xlabel("Fitted");
ylabel("Residuos Studentizados");

clear all;

%% Base de datos CARSEATS

load Carseats.mat;

% Ajustar modelo de RLM para predecir Sales en base a Todos los predictores
% y un par de terminos de interacción (Income x Advertising y Price x Age)

%% FORMA 1: A través de 'Matriz de términos (T)'

T = eye(size(Carseats, 2))

% Vaciamos la columna correspondiente a la columna en Carseats que tiene
% nuestra respuesta

T(1,1) = 0;

% para crear los términos de interacción, añadimos filas que incluyan
% ambos elementos.

% Income x Advertising
TI_1 = zeros(1,size(Carseats, 2));
TI_1([3 4]) = 1;
% Price & Age
TI_2 = zeros(1,size(Carseats, 2));
TI_2([6 8]) = 1;

T = [T; TI_1; TI_2];

% Si le damos la tabla + filtrar no tenemos que decirle los nombres de las
% variables.
% Si tenemos que decirle que variables son cualitativas.
mdl = fitlm(Carseats,T, "CategoricalVars",[7 10 11])

%% FORMA 2: A través de 'Notación Wilkinson'
% What a fucking shitshow
% Tenemos que decirle que variables son cualitativas.

mdl = fitlm(Carseats,'Sales ~ CompPrice + Income + Advertising + Population + Price + ShelveLoc + Age + Education + Urban + US + Income:Advertising + Price:Age', "CategoricalVars",[7 10 11])


end

