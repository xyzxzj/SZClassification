
function [p_value] = my_ttest( X,Y)
%MY_TTEST �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

[trainNum, d] = size(X);

train_y_0=(Y==-1);
train_y_1=(Y==1);
for i=1:d
    [h,p,ci,stats]=ttest2(X(train_y_0,i),X(train_y_1,i));
    p_value(i)=p;
end
end

