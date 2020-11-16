clear all
clc

%%1.��������
load spectra_data.mat

%%2.�������ѵ�����Ͳ��Լ�
temp = randperm(size(NIR,1));
%ѵ����-50������
P_train = NIR(temp(1:50),:)';
T_train = octane(temp(1:50),:)';

%���Լ�-10������
P_test = NIR(temp(51:60),:)';
T_test = octane(temp(51:60),:)';
N = size(P_test,2);

%%3.��һ������
% 3.1. ѵ����
[Pn_train,inputps] = mapminmax(P_train);
Pn_test = mapminmax('apply',P_test,inputps);
% 3.2. ���Լ�
[Tn_train,outputps] = mapminmax(T_train);
Tn_test = mapminmax('apply',T_test,outputps);

%%4.ELMѵ��
[IW1,B1,H1,TF1,TYPE1] = elmtrain(Pn_train,Tn_train,30,'sig',0);
%[IW2,B2,H2,TF2,TYPE2] = elmtrain(H1,Tn_train,20,'sig',0);
%[IW3,B3,H3,TF3,TYPE3] = elmtrain(H2,Tn_train,30,'sig',0);
%LW = pinv(H1') * Tn_train';
%%5.ELM�������
tn_sim01 = elmpredict(Pn_test,IW1,B1,TF1,TYPE1);
%tn_sim02 = elmpredict(tn_sim01,IW2,B2,TF2,TYPE2);
%tn_sim03 = elmpredict(tn_sim02,IW3,B3,TF3,TYPE3);
%����ģ�����
%tn_sim = (tn_sim01' * LW)';
%5.1. ����һ��
T_sim = mapminmax('reverse',tn_sim,outputps);

%%6.����Ա�
result = [T_test' T_sim'];
%6.1.�������
E = mse(T_sim - T_test);

%6.2 ���ϵ��
N = length(T_test);
R2 = (N*sum(T_sim.*T_test)-sum(T_sim)*sum(T_test))^2/((N*sum((T_sim).^2)-(sum(T_sim))^2)*(N*sum((T_test).^2)-(sum(T_test))^2));

%%7.��ͼ
figure(1);
plot(1:N,T_test,'r-*',1:N,T_sim,'b:o');
grid on 
legend('��ʵֵ','Ԥ��ֵ')
xlabel('�������')
ylabel('����ֵ')
string = {'���Լ�����ֵ����Ԥ�����Ա�(ELM)';['(mse = ' num2str(E) ' R^2 = ' num2str(R2) ')']};
title(string)

