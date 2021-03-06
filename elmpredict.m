function Y = elmpredict(P,IW,B,LW,TF,TYPE)
% ELMPREDICT Simulate a Extreme Learning Machine
% Syntax
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% Description
% Input
% P   - Input Matrix of Training Set  (R*Q)
% IW  - Input Weight Matrix (N*R)
% B   - Bias Matrix  (N*1)
% LW  - Layer Weight Matrix (N*S)
% TF  - Transfer Function:
%       'sig' for Sigmoidal function (default)
%       'sin' for Sine function
%       'hardlim' for Hardlim function
% TYPE - Regression (0,default) or Classification (1)
if nargin < 6
    error('ELM:Arguments','Not enough input arguments.');
end
% Calculate the Layer Output Matrix H
Q = size(P,2);
BiasMatrix = repmat(B,1,Q);
tempH = IW * P + BiasMatrix;

switch TF
    case 'sig'
        H = 1./ (1+exp(-tempH));
        
    case 'sin'
        H = sin(tempH);
        
    case 'hardlim'
        H = hardlim(tempH);
end
%����ģ�����
Y = (H' * LW)';
if TYPE == 1
    temp_Y = zeros(size(Y));
    for i = 1:size(Y,2)
      [~,index] = max(Y(:,i));
      temp_Y(index,i) = 1;
    end
    Y = vec2ind(temp_Y);
        

end

