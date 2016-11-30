clear;
% I = rand(320,320)*5.5;
% I_fft = fft2(I);
% 
% % J = imread('figure5-3.png');
% % J = imnoise(I,'gaussian',1,0.01);
% % J = imnoise(I,'salt & pepper',0.1);
% J = I;
% J_fft = fft2(J);
% 
% J_fft = fftshift(J_fft); % Center FFT
% J_fft = abs(J_fft); % Get the magnitude
% J_fft = log(J_fft+1); % Use log, for perceptual scaling, and +1 since log(0) is undefined
% % J_fft = mat2gray(J_fft); % Use mat2gray to scale the image between 0 and 1
% 
% figure;
% 
% subplot(1,2,1);
% imagesc(J)
% colorbar
% axis equal
% axis off
% 
% subplot(1,2,2);
% imagesc(J_fft); % Display the result
% colorbar
% axis equal
% axis off

std=[1.73,0.87223,0.34858,0.17422,0.08362,0.036857];
fft=[6.0312,4.6683,2.8952,1.7031,0.77638,0.30214];
bright=[3.8622,4.2435,4.3449,4.3524,4.3495,4.3506];
downsampleScale = [1,2,5,10,20,40];



figure;

subplot(2,2,1);
plot(downsampleScale,std,'*-');
xlabel('downsampleScale');
ylabel('std');

subplot(2,2,2);
plot(std,fft,'*-');
xlabel('std');
ylabel('fft');

subplot(2,2,3);
plot(std,bright,'*-');
xlabel('std');
ylabel('bright');

subplot(2,2,4);
plot(fft,bright,'*-');
xlabel('fft');
ylabel('bright');