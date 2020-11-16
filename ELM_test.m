clear all
clc

%%1.导入数据
load spectra_data.mat

%%2.随机产生训练集和测试集
temp = randperm(size(NIR,1));
%训练集-50个样本
P_train = NIR(temp(1:50),:)';
T_train = octane(temp(1:50),:)';

%测试集-10个样本
P_test = NIR(temp(51:60),:)';
T_test = octane(temp(51:60),:)';
N = size(P_test,2);

%%3.归一化数据
% 3.1. 训练集
[Pn_train,inputps] = mapminmax(P_train);
Pn_test = mapminmax('apply',P_test,inputps);
% 3.2. 测试集
[Tn_train,outputps] = mapminmax(T_train);
Tn_test = mapminmax('apply',T_test,outputps);

%%4.ELM训练
[IW1,B1,H1,TF1,TYPE1] = elmtrain(Pn_train,Tn_train,30,'sig',0);
%[IW2,B2,H2,TF2,TYPE2] = elmtrain(H1,Tn_train,20,'sig',0);
%[IW3,B3,H3,TF3,TYPE3] = elmtrain(H2,Tn_train,30,'sig',0);
%LW = pinv(H1') * Tn_train';
%%5.ELM仿真测试
tn_sim01 = elmpredict(Pn_test,IW1,B1,TF1,TYPE1);
%tn_sim02 = elmpredict(tn_sim01,IW2,B2,TF2,TYPE2);
%tn_sim03 = elmpredict(tn_sim02,IW3,B3,TF3,TYPE3);
%计算模拟输出
%tn_sim = (tn_sim01' * LW)';
%5.1. 反归一化
T_sim = mapminmax('reverse',tn_sim,outputps);

%%6.结果对比
result = [T_test' T_sim'];
%6.1.均方误差
E = mse(T_sim - T_test);

%6.2 相关系数
N = length(T_test);
R2 = (N*sum(T_sim.*T_test)-sum(T_sim)*sum(T_test))^2/((N*sum((T_sim).^2)-(sum(T_sim))^2)*(N*sum((T_test).^2)-(sum(T_test))^2));

%%7.成图
figure(1);
plot(1:N,T_test,'r-*',1:N,T_sim,'b:o');
grid on 
legend('真实值','预测值')
xlabel('样本编号')
ylabel('辛烷值')
string = {'测试集辛烷值含量预测结果对比(ELM)';['(mse = ' num2str(E) ' R^2 = ' num2str(R2) ')']};
title(string)

