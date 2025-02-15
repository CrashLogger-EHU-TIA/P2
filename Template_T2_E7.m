function Template_T2_E7
% Este script contiene la resolución del ejercicio aplicado 7 del Tema 2
% de la asignatura 'Técnicas de Inteligencia Artificial'


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EJERCICIO 7 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Cargar base de datos
load Auto.mat
close all;

var_names = Auto.Properties.VariableNames

fprintf('\n\n')
disp('%%%%%%%%%%%%%%%%% EJERCICIO 7 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')


% Remover valores NaN
Auto = rmmissing(Auto);

% Apartado 1 - Producir una matriz con las diferentes gráficas de dispersión (scatterplots) para las
% diferentes variables de la base de datos. 

figure("Name","APT. 1: Mas relevantes")
for(i = 2:8)
    subplot(2,4,i-1);
    X = Auto{:,i};
    scatter(X, Auto.mpg)
    title(i)
    xlabel(var_names(i));
    ylabel("Efficiency [mpg]");
end

figure("Name","APT.2: Todos")
plotmatrix(Auto{:,1:end-1})

% Apartado 2 - Calcular la matriz de correlación entre las diferentes variables. 
% Únicamente excluir la variable \textcolor{granate}{name}, única no numérica.

Matriz_Corr = corr(Auto{:,1:end-1})

% Apartado 3 -  Ajustar un modelo de regresi´ on lineal múltiple que tenga 
% mpg como respuesta y el resto de variables, excepto name, como predictores.

fprintf('\n\n')
disp('%%%%%%%%%%%%%%%%% APARTADO 3 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')

T = eye(size(Auto,2)-1);
AutoNoNames = Auto(:,(1:end-1));
T(1,1) = 0;
mdl = fitlm(AutoNoNames,T,"CategoricalVars",[size(AutoNoNames,2)])

disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')
disp('1: Sí, hay una clara relación entre predictores y respuesta, pero no en todos los predictores.')
disp('2: Los predictores relacionados con el motor (displacement, horsepower, cylinders), el vehículo como tal (weight, year) e incluso el origen del vehículo tienen relación. Acceleration, sin embargo, es resultado de otros factores (motor VS peso, sobre todo).')
disp('3: Que a mas nuevo el vehículo, mas eficiente será.')

% Apartado 4 -  Visualiza gráficas de residuos y de influencia (leverage) y comenta los posibles
% problemas que existan con el ajuste por mínimos cuadráticos. ¿Los gráficos de
% residuos sugieren la presencia de valores atípicos inusualmente grandes? ¿El gráfico de
% influencia sugiere la presencia de observaciones con influencia (leverage) inusualmente
% alta?

figure(4)

% Analizamos residuos
subplot(1,3,1);
plotResiduals(mdl, "fitted", "Marker","o");
xlabel("Fitted");
ylabel("Residuals");

fprintf('\n\n')
disp('%%%%%%%%%%%%%%%%% APARTADO 4 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')

disp('Los residuos cobran la forma de U característica de una regresión lineal que intenta aproximar algo mas parabólico.')

% Analizamos high leverage points
subplot(1,3,2);
plotDiagnostics(mdl, "leverage", "Marker","o")
xlabel("Fitted");
ylabel("Leverage");

disp('Hay un punto en particular, observación #14, que tiene un leverage muy alto. El #29 es también considerablemente mayor al resto.')

% Analizamos leverage vs studentized residuals
subplot(1,3,3);
plot(mdl.Diagnostics.Leverage, mdl.Residuals.Studentized, "o");
xlabel("Leverage");
ylabel("Residuos Studentizados")

% Apartado 5 -  Introduce términos de interacción en el modelo. ¿Es posible que algún término de
% interacción sea estadísticamente significativo?

% Introducimos términos de interacción en el modelo

fprintf('\n\n')
disp('%%%%%%%%%%%%%%%%% APARTADO 5 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')

T1_1 = [0, 1, 1, 0, 0, 0, 0, 0];
T1_2 = [0, 1, 0, 1, 0, 0, 0, 0];
T1_3 = [0, 1, 0, 0, 1, 0, 0, 0];
T1_4 = [0, 0, 1, 1, 0, 0, 0, 0];
T1_5 = [0, 0, 1, 0, 1, 0, 0, 0];
T1_6 = [0, 0, 0, 1, 1, 0, 0, 0];

T = [T; T1_1; T1_2; T1_3; T1_4; T1_5; T1_6];

mdl = fitlm(AutoNoNames,T,"CategoricalVars",[size(AutoNoNames,2)])

disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')
disp('Definitivamente hay términos de interacción muy importantes: Todos los elementos del motor tienen relaciónes entre sí.')

% Apartado 6 -  Prueba diferentes transformaciones de las variables
% Introducimos términos de interacción en el modelo

fprintf('\n\n')
disp('%%%%%%%%%%%%%%%%% APARTADO 6 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')

X1 = Auto.horsepower;
X2 = X1.^2;
X3 = sqrt(X1);
X4 = log(X1);
Y = Auto.mpg;

% Using {} to make cell arrays

%mdl1 = fitlm([X1, X2, X3, X4], Y, "VarNames", {var_names{[4]}, 'horsepower²', 'Sqrt(horsepower)', 'log(horsepower)', var_names{1}})
mdl0 = fitlm([X1], Y, "VarNames", {var_names{[4]}, var_names{1}})
mdl1 = fitlm([X1, X2,], Y, "VarNames", {var_names{[4]},'horsepower²', var_names{1}})
mdl2 = fitlm([X1, X3], Y, "VarNames", {var_names{[4]},'Sqrt(horsepower)', var_names{1}})
mdl3 = fitlm([X1, X4], Y, "VarNames", {var_names{[4]},'log(horsepower)', var_names{1}})

figure("Name","Comparación de modelos")
scatter(X1, Y);
title("Modelo Lineal Vs Cuadrático Vs De raiz cuadrada Vs Logarítmico");
xlabel("lstat")
ylabel("medv")
hold on;

[x_ord, ord] = sort(X1);

ypred0 = predict(mdl0, x_ord);
ypred1 = predict(mdl1, [x_ord, x_ord.^2]);
ypred2 = predict(mdl2, [x_ord, sqrt(x_ord)]);
ypred3 = predict(mdl3, [x_ord, log(x_ord)]);

plot(x_ord, ypred0, LineWidth=2)
plot(x_ord, ypred1, LineWidth=2)
plot(x_ord, ypred2, LineWidth=2)
plot(x_ord, ypred3, LineWidth=2)

legend('Datos','Lineal','Cuadrático', 'Raiz cuadrada', 'Logarítmico')

hold off;

disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')
disp('La mejor aproximación es la cuadrática, pero todas las variables transformadas son mejores aproximacione que la recta original.')


%% Conclusión


fprintf("Usando todo lo aprendido, podemos hacer una aproximación para el problema inicial, con menos predictores y precisión similar al uso de todos los predictores directamente")


clear("T")

AutoNoNames = Auto(:,(1:end));

AutoNoNames(:,end) = [];
AutoNoNames(:,2) = [];
AutoNoNames(:,2) = [];
AutoNoNames(:,4) = [];
newTabCol = AutoNoNames{:,3}.^2;
AutoNoNames.("Displacement_^2") = newTabCol;

T = eye(size(AutoNoNames,2));
T(1,1) = 0;

T1_1 = [0, 1, 0, 0, 0, 1];
T1_2 = [0, 0, 0, 1, 0, 1];

T = [T; T1_1; T1_2];

mdl = fitlm(AutoNoNames,T,"CategoricalVars",[size(AutoNoNames,2)-1])

hold off;