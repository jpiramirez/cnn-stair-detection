close all
a = load('fusion_train_evo.txt');

%a = load('mobilenet_train_evo.txt');

hold on
plot( a(:,1), a(:,2) )
hold on
plot( a(:,1), a(:,3) )