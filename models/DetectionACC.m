
clc

data = readtable('./../../cnn-stair-detection/Dataset/test/data_3class.txt');
class = data{:,2};

prediction = load('fusion_3class_450_test.txt');
time_t = prediction(  : , 2 );
prediction = prediction( :, 1 );
disp('time')
mean(time_t)

% ascendint stairs
ind = find( class == 1 ); 
t_up = length( ind );           

% descending stairs
ind = find( class == 2 );
t_down = length( ind );

% level-ground
ind = find( class == 0 );
t_ground = length( ind );

t_data = length( class );

up_tp = 0;
down_tp = 0;
ground_tp = 0;
for i = 1 : length( class ) 
    if class(i) == 1 && prediction(i) == 1 
        up_tp = up_tp + 1;
    elseif class(i) == 2 && prediction(i) == 2 
        down_tp = down_tp + 1;
    elseif class(i) == 0 && prediction(i) == 0
        ground_tp = ground_tp + 1;             
    end
end

ACC = ( up_tp + down_tp + ground_tp) / t_data
down_acc = down_tp / t_down
up_acc = up_tp / t_up
ground_acc = ground_tp / t_ground

    
    
% 
% tpr = tp/p;
% fnr = fn/p;
% tnr = tn/n;
% fpr = fp/n;
% 
% den = sqrt( (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) );
% 
% specificity = tn/(tn + fp)
% sensitivity = tp/(tp + fn)
% acc = (tp + tn) / (p + n)
% mcc = (tp*tn - fp*fn) / den
% 
% 
