function [xout,noframes]=frames(xin, fl, po)
%THis code 
% function [xout,noframes]=frames(xin, fl, po)
%arrange xin in frames in xout so that each column is a frame
%noframes is the number of frames

L=length(xin);
ol=fix(po*fl/100);
rl=fl-ol;
 noframes=fix((L-fl-1)/rl+1);
% noframes=417;
begin=1;
fin=begin+fl-1;
xout=zeros(fl,noframes);
for n=1:noframes
    xout(:,n)=xin(begin:fin);
    begin=begin+rl;
    
    fin=begin+fl-1;
end

