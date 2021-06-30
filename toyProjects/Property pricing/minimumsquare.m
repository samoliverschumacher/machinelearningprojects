function [base,height] = minimumsquare( numberelements )

height = floor(sqrt(numel(numberelements)))*(rem( (sqrt(numel(numberelements))) , 1)<0.5) + ceil(sqrt(numel(numberelements)))*(rem( (sqrt(numel(numberelements))) , 1)>=0.5);
base =  ceil(sqrt(numel(numberelements)));

end

