
function [output]=three_decimals(input)
c=num2cell(input);
c_fmt = c;                                              %// Temporary cell array
idx = cellfun(@isnumeric, c(:));                        %// Locate numbers
c_fmt(idx) = cellfun(@(x){sprintf('%.3f', x)}, c(idx)); %// Format as strings, .3 is after decimal numbers

output = str2double(c_fmt);
end