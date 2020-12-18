clc;
clear all;
close all

model = 'mobilenet';  
cmd = sprintf('%s_train_evo.txt',model);
a = load( cmd );


% Accuracy
plot( a(:,1), a(:,3), 'LineWidth', 3 )   
grid on
hold on
plot( a(:,1), a(:,5), 'LineWidth', 3 )   
legend('Training','Validation')
set(gca,'Fontsize',22)

% Loss
figure 
plot( a(:,1), a(:,2), 'LineWidth', 3 )   
grid on
hold on
plot( a(:,1), a(:,4), 'LineWidth', 3 )   
legend('Training','Validation')
set(gca,'Fontsize',22)