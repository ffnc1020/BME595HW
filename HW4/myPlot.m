close all
clear variables
clc
%%
load('nn.mat'); 

fig1 = figure(1);

plot(Training_error,'linewidth',2);
hold on
plot(Test_error,'linewidth',2);
legend('Training','Test')
xlabel('Epoch')
ylabel('CE Loss')

% for i = 1:10
% subplot(2,5,i);
% plot(Training_error(i,:),'linewidth',1);
% hold on
% plot(Test_error(i,:),'linewidth',1);
% xlabel('Epoch')
% ylabel('Logistic loss')
% title(strcat('Class ',num2str(i)))
% end
% legend('Training','Test')
% set(fig1,'Position',[0,0,1400,500]);