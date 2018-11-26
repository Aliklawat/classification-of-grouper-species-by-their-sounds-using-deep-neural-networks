function [rr]=spectroo(x,fs)
% Florida Atlantic unversity
% This program done by Ali K Ibrahim to compute spectrogram of sound signal
% and convert it to RGB image. If you have any question,please email me
% (aibrahim2014@fau.edu)
sr=fs;
time=[0:length(x)-1]/sr;
% plot(time,x)
frame_duration=.1; %in seconds  DURATION MAY BE CHANGED
lf=fix(frame_duration*sr); %frame length in samples
po=80;  %percent overlap
nolf=fix((100-po)*lf/100);
[y, noframes]=frames(x, lf, po);
%y is a matrix lf-by-noframes
taper=window(@hamming,lf);
taper=taper*ones(1,noframes);
y=y.*taper;
nfft=2^(fix(log(lf)/log(2))+3)*2;
fy=fft(y,nfft);
afy=abs(fy(1:nfft/2,:));
sfy=afy.*afy;
t=[0:noframes-1]*nolf/sr;
f=[0:nfft/2-1]*sr/nfft;
% figure,
% imagesc(t,f,20*log10(sfy+eps)), axis xy; 
% colormap(jet);
% xlabel('time (s)')
% ylabel('frequency, (Hz)')
y=flipud(sfy);
%%%%%%%%
G=20*log10(sfy+eps);
C = colormap;  % Get the figure's colormap.
L = size(C,1);
% Scale the matrix to the range of the map.
Gs = round(interp1(linspace(min(G(:)),max(G(:)),L),1:L,G));
H = reshape(C(Gs,:),[size(Gs) 3]); % Make RGB image from scaled.
rr=H;
% imrotate(H,90);
% figure,imshow(rr);