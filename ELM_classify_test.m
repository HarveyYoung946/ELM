%% I. ��ջ�������
clear all
clc
 
%% II. ѵ����/���Լ�����
%%
% 1. ��������
load iris_data.mat
 
%%
% 2. �������ѵ�����Ͳ��Լ�
P_train = [];
T_train = [];
P_test = [];
T_test = [];
for i = 1:3
    temp_input = features((i-1)*50+1:i*50,:);
    temp_output = classes((i-1)*50+1:i*50,:);
    n = randperm(50);
    % ѵ�����D�D120������
    P_train = [P_train temp_input(n(1:40),:)'];
    T_train = [T_train temp_output(n(1:40),:)'];
    % ���Լ��D�D30������
    P_test = [P_test temp_input(n(41:50),:)'];
    T_test = [T_test temp_output(n(41:50),:)'];
end
 
%% III. ELM����/ѵ��
[IW,B,LW,TF,TYPE] = elmtrain(P_train,T_train,20,'sig',1);
 
%% IV. ELM�������
T_sim_1 = elmpredict(P_train,IW,B,LW,TF,TYPE);
T_sim_2 = elmpredict(P_test,IW,B,LW,TF,TYPE);
 
%% V. ����Ա�
result_1 = [T_train' T_sim_1'];
result_2 = [T_test' T_sim_2'];
%%
% 1. ѵ������ȷ��
k1 = length(find(T_train == T_sim_1));
n1 = length(T_train);
Accuracy_1 = k1 / n1 * 100;
disp(['ѵ������ȷ��Accuracy = ' num2str(Accuracy_1) '%(' num2str(k1) '/' num2str(n1) ')'])
 
%%
% 2. ���Լ���ȷ��
k2 = length(find(T_test == T_sim_2));
n2 = length(T_test);
Accuracy_2 = k2 / n2 * 100;
disp(['���Լ���ȷ��Accuracy = ' num2str(Accuracy_2) '%(' num2str(k2) '/' num2str(n2) ')'])
 
%% VI. ��ͼ
figure(2)
plot(1:30,T_test,'bo',1:30,T_sim_2,'r-*')
grid on
xlabel('���Լ��������')
ylabel('���Լ��������')
string = {'���Լ�Ԥ�����Ա�(ELM)';['(��ȷ��Accuracy = ' num2str(Accuracy_2) '%)' ]};
title(string)
legend('��ʵֵ','ELMԤ��ֵ')