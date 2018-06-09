%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         Máster en Ingeniería Informática - UCLM                       %%
%%         Diseño de Sistemas Inteligentes - TRABAJO CLUSTERING          %%
%                                                                         %
%Elaborado por: -Balmaceda Torres, Gustavo Adolfo                         %
%                                                 Ciudad Real, Junio/2018 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all, clear, clc   % Cerrar ventanas gráficas y borrar memoria/consola

%LECTURA de las columnas necesarias del archivo de entrenamiento (_train) del conjunto de datos de desarrollo.
%%%%Columnas:
%Anon Student Id, Cuenta de Problem Name,	Cuenta de Step Name, Suma de Step Duration (sec),
%Suma de Correct Step Duration (sec), Suma de Error Step Duration (sec),	Suma de Correct First Attempt,
%Suma de Problem View,	Suma de Corrects,	Suma de Incorrects,	Suma de Hints, Duracion Promedia por paso (sec),
%Durancion promedia pasos correctos, Duracion promedia de pasos incorrectos,	Promedio de pasos resueltos en primer intento

%Datos_train=xlsread('algebra_2005_2006_Student_train.xls','B2:O401')%Matriz total

Cantidad_Pasos=xlsread('algebra_2005_2006_Student_train.xlsx','C2:C401');
Resueltos_primera_vez=xlsread('algebra_2005_2006_Student_train.xlsx','G2:G401')
Pasos_Correctos=xlsread('algebra_2005_2006_Student_train.xlsx','I2:I401');
Pasos_Incorrectos=xlsread('algebra_2005_2006_Student_train.xlsx','J2:J401');
Consultas=xlsread('algebra_2005_2006_Student_train.xlsx','K2:K401');

%%%%%%%% NORMALIZANDO DATOS: %%%%%%%%
Cantidad_Pasos=(Cantidad_Pasos-mean(Cantidad_Pasos))/std(Cantidad_Pasos);
Resueltos_primera_vez=(Resueltos_primera_vez-mean(Resueltos_primera_vez))/std(Resueltos_primera_vez);
Pasos_Correctos=(Pasos_Correctos-mean(Pasos_Correctos))/std(Pasos_Correctos);
Pasos_Incorrectos=(Pasos_Incorrectos-mean(Pasos_Incorrectos))/std(Pasos_Incorrectos);
Consultas=(Consultas-mean(Consultas))/std(Consultas);

%%%%%%%%%%%%%%%%DATOS A ANALIZAR Y AGRUPADOS%%%%%%%%%%%%%%%%%%%%
DatosFinales= [Cantidad_Pasos Resueltos_primera_vez Pasos_Correctos Pasos_Incorrectos Consultas]

%%%%%%%%%%%%%%%%% DIBUJAR GRÁFICAS DE CARACTERÍSTICAS "PATRONES" %%%%%%%%%%%%%%%%%
figure(1)   % creación de una ventana gráfica
title('Características Alumnos/Patrones de Rendimiento','fontsize',12)% título del gráfico
hold on     % hold on: para dibujar varias gráficas en la misma ventana
% Dibuja Primera y segunda columna (sr->cudrados rojos)
subplot(2,2,1),plot(Cantidad_Pasos,Resueltos_primera_vez,'sr')
axis('square'),box on, hold off     % ejes cuadrados (misma escala)
xlabel('Cantidad Pasos','fontsize',10) % etiquetado del eje-x
ylabel('Resueltos primera vez','fontsize',10) % etiquetado del eje-y
hold on  
% Dibuja primera y tercera columna(sb->cudrados azules)
subplot(2,2,2),plot(Cantidad_Pasos,Pasos_Correctos,'sb')
axis('square'),box on, hold off     % ejes cuadrados (misma escala)
xlabel('Cantidad Pasos','fontsize',10) % etiquetado del eje-x
ylabel('Pasos Correctos','fontsize',10) % etiquetado del eje-y
hold on    
% Dibuja primera y cuarta columna(sy->cudrados amarillos)
subplot(2,2,3),plot(Cantidad_Pasos,Pasos_Incorrectos,'sy')
axis('square'),box on, hold off     % ejes cuadrados (misma escala)
xlabel('Cantidad Pasos','fontsize',10) % etiquetado del eje-x
ylabel('Pasos Incorrectos','fontsize',10) % etiquetado del eje-y
hold on    
% Dibuja tercera y quinta columna(sy->cudrados cyan)
subplot(2,2,4),plot(Pasos_Correctos,Consultas,'sc')
axis('square'),box on, hold off     % ejes cuadrados (misma escala)
xlabel('Pasos Correctos','fontsize',10) % etiquetado del eje-x
ylabel('Consultas','fontsize',10) % etiquetado del eje-y
hold on    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% APLICANDO   ALGORITMO   K - m e d i a s  y B I C            %%
% BIC- Permite calcular la cantidad de grupos óptimos a         %
%considerar en el Algoritmo                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Kmax=10;
for K=2:Kmax
% m=2;                       % parámetro de fcm, 2 es el defecto
% MaxIteraciones=100;        % número de iteraciones
% Tolerancia= 1e-5;          % tolerancia en el criterio de para
% Visualizacion=0;           % 0/1
% opciones=[m,MaxIteraciones,Visualizacion];
% [center,U,obj_fcn] = fcm(DatosFinales, K,opciones);

% Parámetros de salida de FCM:              
% center    centroides de los grupos
% U         matriz de pertenencia individuo cluster 
% obj_fun   función objetivo

% PARÁMETROS de salida de K-MEANS similares a Fuzzi C-Means              
% cidx(i):       devuelve el conglogmerado al que pertencece el dato i
% ctrs = center: centroides de los grupos
% sumd:          suma de la distancia intracluster 
% D = U:         matriz de distancia de cada objeto (fila) a cada centroide (columna)
 
opts = statset('Display','iter','MaxIter',100);
[cidx, ctrs,sumd,U] = kmeans(DatosFinales, K,'Replicates',1, 'Distance','sqEuclidean','Options',opts);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Asignación de individuo a grupo, maximizando el nivel de pertenencia al grupo
for j=1:K
maxU=max(U); % calculo del máximo nivel de pertenencia de los individuos
individuos=find(U(j,:)==maxU) % calcula los individuos del grupo i que alcanzan el máximo
cidx(individuos)=j;           % asigna estos individuos al grupo i
grado_pertenecia(individuos)=maxU(individuos);
end
[Bic_K,xi]=BIC(K,cidx,DatosFinales);
BICK(K)=Bic_K;
end
%La Figura 2 representa el valor que "K" debe tener para el algoritmo
figure(2)
plot(2:K',BICK(2:K)','s-','MarkerSize',6,'MarkerEdgeColor','r', 'MarkerFaceColor','r')
xlabel('K','fontsize',18)      % etiquetado del eje-x
ylabel('BIC(K)','fontsize',18) % etiquetado del eje-y
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K_Optimo=find(BICK(1:Kmax)==min(BICK(2:Kmax)))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%        APLICANDO   ALGORITMO   K - m e d i a s              %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X=DatosFinales
opts = statset('Display','iter','MaxIter',100);
[cidx, ctrs,sumd,D] = kmeans(X, K_Optimo,'Replicates',1, 'Distance','sqEuclidean','Options',opts);
%%%%%%%%                       
% p a r á m e t r o s   d e   e n t r a d a
% X matriz de datos, filas individuos, columnas atributos
% K= numero de grupos a conformar
% 'Replicates' número de repeticiones, 1 en este caso 
% 'Distance'  distancia usada, 'sqEuclidean'->el cuadrado de la euclídea, también se puede elegir 'city'->L1
%%%%%%%
% p a r á m e t r o s   d e   s a l i d a              
% cidx(i) devuelve el conglogmerado al que pertencece el dato i
% ctrs    centroides de los grupos
% sumd    suma de la distancia intracluster 
% D       matriz de distancia de cada objeto (fila) a cada centroide (columna)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% R e p r e s e n t a c i ó n   d e    l a    s o l u c i ó n %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(3) %% Figura 3- Representación de individuos
plot(DatosFinales(cidx==1,1),DatosFinales(cidx==1,2),'*','MarkerSize',6,...
                  'MarkerEdgeColor','r','MarkerFaceColor','r');
hold on
plot(DatosFinales(cidx==2,1),DatosFinales(cidx==2,2),'o','MarkerSize',6,...
                  'MarkerEdgeColor','b', 'MarkerFaceColor','b');
hold on
plot(DatosFinales(cidx==3,1),DatosFinales(cidx==3,2),'s','MarkerSize',6,...
                  'MarkerEdgeColor','y','MarkerFaceColor','y');
hold on
plot(DatosFinales(cidx==4,1),DatosFinales(cidx==4,2),'^','MarkerSize',6,...
                  'MarkerEdgeColor','g','MarkerFaceColor','g');
hold on
plot(DatosFinales(cidx==5,1),DatosFinales(cidx==5,2),'+','MarkerSize',6,...
                  'MarkerEdgeColor','k','MarkerFaceColor','k');
hold on
xlabel('x_1','fontsize',18),ylabel('x_2','fontsize',18)
legend('Grupo 1','Grupo 2','Grupo 3','Grupo 4','Grupo 5'),axis('square'), box on
title('Algoritmo K-Medias - Agrupaciones','fontsize',16)
save ('datos_cluster_1.mat','X') % guarda matriz de datos X