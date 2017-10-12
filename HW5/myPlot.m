close all
clear variables
clc
%%
load('my4.mat')
fig1 = figure(1);

plot(FC_accuracy,'linewidth',2);
hold on
plot(Le_accuracy,'linewidth',2);
legend('Training','Test')
xlabel('Epoch')
ylabel('Test Accuracy')
legend('Fully Connected','LeNet')

fig2 = figure(2);

plot(FC_training_loss,'linewidth',2);
hold on
plot(FC_test_loss,'linewidth',2);
plot(Le_training_loss,'linewidth',2);
plot(Le_test_loss,'linewidth',2);
legend('FC Training','FC Test','LeNet Training','LeNet Test')
xlabel('Epoch')
ylabel('Loss')
