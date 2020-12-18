
clc

data = readtable('./..//Dataset/test/data.txt');
class = data{:,2};
prediction = load('mobilenet_fusion_400_test.txt');
%prediction = load('mobilenet_400_test.txt');

time_t = prediction(:,2);
prediction = prediction(:,1);

disp('time')
mean(time_t)

% 1 - obstacle
% 0 - non-obstacle
tp = 0;
tn = 0;
fp = 0;
fn = 0;
p = sum( class );
n = length(class) - p;
for i = 1 : length( class ) 
    if class(i) == 1 && prediction(i) == 1 
        tp = tp + 1;
    elseif class(i) == 1 && prediction(i) == 0
        fn = fn + 1;
    elseif class(i) == 0 && prediction(i) == 0
        tn = tn + 1;
    else
        fp = fp + 1;            
    end
end

den = sqrt( (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) );

specificity = tn/(tn + fp)
sensitivity = tp/(tp + fn)
acc = (tp + tn) / (p + n)
mcc = (tp*tn - fp*fn) / den


