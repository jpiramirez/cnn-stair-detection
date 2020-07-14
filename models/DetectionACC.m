clc

data = readtable('data.txt');
class = data{:,2};
%prediction = load('prediction_fusion_e2.txt');
%prediction = load('prediction_mobilenet_e2.txt');

prediction = load('all_fusion_e2.txt');

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
tpr = tp/p
fnr = fn/p
tnr = tn/n
fpr = fp/n
