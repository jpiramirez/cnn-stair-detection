clc

data = readtable('./../Dataset/test/data.txt');
class = data{:,2};

%prediction = load('mobilenet_fusion_1000_test.txt');

%prediction = load('mobilenet_600_test_mac.txt');
prediction = load('mobilenet_fusion_600_test_mac.txt');

prediction = prediction(:,1);

% 1 - obstacle
% 0 - non-obstacle
tp = 0;
tn = 0;
fp = 0;
fn = 0;
p = sum( class )
n = length(class) - p
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
