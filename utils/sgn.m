function [ z ] = sgn( x )
%SGN �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
z = x;
z(x<=0) = -1;
z(x>0) = 1;
end

