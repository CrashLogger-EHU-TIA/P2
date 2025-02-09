function Template_T2_E6
% Este script contiene la resolución del ejercicio aplicado 6 del Tema 2
% de la asignatura 'Técnicas de Inteligencia Artificial'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EJERCICIO 6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
clc;
close all;

% Cargar base de datos
load Auto.mat

var_names = Auto.Properties.VariableNames


disp('%%%%%%%%%%%%%%%%% EJERCICIO 6 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')

% Apartado 1 - Ajustar un modelo de regresión lineal simple que tenga mpg como respuesta y
% horsepower como predictor

disp('%%%%%%%%%%%%%%%%% Apartado 1 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

% Remover valores NaN

Auto = rmmissing(Auto);

X = Auto.horsepower;
Y = Auto.mpg;

mdl = fitlm(X,Y,"VarNames",var_names([4 1]))




% Apartado 2 - Visualiza gráficamente predictor vs respuesta, y la recta de regresión.

figure(1)
plot(mdl)

figure(2)

plot_pred = predict(mdl, X);
scatter(X,Y, "DisplayName","Data In")
hold on;
plot(X,plot_pred, "DisplayName", "Modelo");
hold off;
title("Modelo de regresión lineal");
xlabel("horsepower");
ylabel("mpg")

% Responde a las siguientes preguntas:
% a) ¿Existe relación entre predictor y respuesta?
fprintf("Hay una relación clara entre predictor y respuesta: A mas alta potencia menor es la eficiencia.\n")
% b) ?Cómo de fuerte es la relación entre predictor y respuesta?
fprintf("Es bastante claramente descendente. La relación es particularmente fuerte en motores de menos de 120 caballos de potencia.\n")
% c) ¿Es la relación entre predictor y respuesta negativa o positiva?
fprintf("Negativa, a mas alto el predictor mas baja la respuesta y viceversa.\n")
% d) ¿Cuál es el mpg estimado para un horsepower igual a 98? ¿Cuáles son los
% intervalos de confianza y predicción del 95%?
fprintf('\n')

% Predicción e intervalos de confianza del 95%
[ypred, y_ci_Conf] = predict(mdl, [98], "Prediction", "Curve");
fprintf("La eficiencia esperada es de %f mpg.\nSu intervalo de confianza es entre %f y %f.\n", ypred, y_ci_Conf(1), y_ci_Conf(2));
[~,y_ci_Conf]=predict(mdl,X,"Prediction","curve");

%Predicción e intervalos de predicción del 95%
[~, y_ci_Pred] = predict(mdl, [98], "Prediction", "Observation");
fprintf("Su intervalo de predicción está entre %f y %f.\n", y_ci_Pred(1), y_ci_Pred(2))
[~,y_ci_Pred]=predict(mdl,X,"Prediction","observation");

input("Pulsa cualquier tecla para continuar (Apartado 2)\n")
close(1)

disp('%%%%%%%%%%%%%%%%% Apartado 2 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

legend;
% Enseñar las rayitas de forma gráfica cool
figure(2)
hold on;
plot(X,y_ci_Conf,"." ,"DisplayName","Confidence bounds") ;
plot(X,y_ci_Pred,"DisplayName","Prediction bounds") ;
xline(98, "DisplayName", "98hp marker")
yline(ypred, "DisplayName","Predicted value")
hold off;

% Apartado 3 - Visualiza residuos y puntos de alta influencia (high leverage points) y comenta los
% posibles problemas que existan con el ajuste por mínimos cuadráticos.
fprintf('\n')

input("Pulsa cualquier tecla para continuar (Apartado 3)\n")
close all

disp('%%%%%%%%%%%%%%%%% Apartado 3 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

% Ploteamos datos y modelo de regresión

figure(3);

subplot(2,2,1)

plot_pred = predict(mdl, X);
scatter(X,Y, "DisplayName","Data In")
hold on;
plot(X,plot_pred, "DisplayName", "Modelo");
hold off;
title("Modelo de regresión lineal");
xlabel("Potencia [hp]");
ylabel("Eficiencia [mpg]")

% Analizamos residuos

subplot(2,2,2)

plotResiduals(mdl, "fitted", "Marker","o", "ResidualType","studentized")
title("Grafico de residuos studentizados");
xlabel("Fitted")
ylabel("Residuos studentizados")

% Analizamos high leverage points

subplot(2,2,3)

plotDiagnostics(mdl, "leverage", "Marker","o")
title("Leverage")
xlabel("Line Index");
ylabel("Leverage")

% Analizamos leverage vs studentized residuals

subplot(2,2,4)
plot(mdl.Diagnostics.Leverage, mdl.Residuals.Studentized, "*")
title("Leverage vs Residuos")
xlabel("Leverage");
ylabel("Residuos studentizados")

fprintf("Los resuduos studentizados han tomado, como era de esperar,una forma mas o menos curva.\n")

fprintf("Hay una cantidad no negligible de puntos con un alto leverage,indicando que hay\nmultiples observaciones distantes de nuestra predicción.\n");

