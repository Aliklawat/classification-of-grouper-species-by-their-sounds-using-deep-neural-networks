function [y,t,f]=spectroo1(x,fs)
% Florida Atlantic unversity
% This program done by Dr Erdol to find the spectrogram of the signal
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
nfft=2^(fix(log(lf)/log(2))+3);
fy=fft(y,nfft);
afy=abs(fy(1:nfft/2,:));
sfy=afy.*afy;

t=[0:noframes-1]*nolf/sr;
f=[0:nfft/2-1]*sr/nfft;
figure,
imagesc(t,f,20*log10(sfy+eps)), axis xy; colormap(jet);
xlabel('Time (s)','fontweight','bold','fontsize',16)
ylabel('Frequency(Hz)','fontweight','bold','fontsize',16)
y=sfy;
y=20*log10(sfy+eps);
