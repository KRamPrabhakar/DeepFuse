clearvars;close all;
savepath     =   './results/';
%% Select the fused luminance channel output of DeepFuse
[FileName,PathName] = uigetfile('*.png','Select the DeepFuse output',...
'E:\DeepHDR\codes\Unsupervised\3_images_05_aug\results\old_expt_1\lum_images\');
fused_l     =   imread(fullfile(PathName, FileName));
% change the range from 0-255 to 16-235
J0   =   im2double(fused_l);
J = imadjust(J0);
L1 = J.*235 + (1-J).*16;

%% Select the input images
[FileName1,PathName1] = uigetfile('*.jpg; *.png; *.tif','Select UE image',...
'/home/lokesh/ram/DeepFuse/iccv/DeepFuse/input_color_images/2Imgs_new/');
im1         = imread(fullfile(PathName1, FileName1));
I(:,:,:,1)  = imresize(im1, [size(L1, 1) size(L1, 2)]);
I1(:,:,:,1) = rgb2ycbcr(I(:,:,:,1));

[FileName3, PathName3] = uigetfile('*.jpg; *.png; *.tif','Select OE image',...
PathName1);
im3         = imread(fullfile(PathName3, FileName3));
I(:,:,:,2)  = imresize(im3, [size(L1, 1) size(L1, 2)]);
I1(:,:,:,2) = rgb2ycbcr(I(:,:,:,2));

% Generate Chrominance part
CG = size(I,3);                                              %Number of channels
NumberOfImages = size(I,4);                                  %Number of images
% Cast to double precision
I1 = double(I1);
% Preallocate
xH = zeros(size(I1,1),size(I1,2),NumberOfImages);
yH = xH;
I_Cb = xH;
I_Cr = xH;
for i = 1:NumberOfImages    
    if CG == 3,
        I_Cb(:,:,i) = I1(:,:,2,i);                         % Store Cb Chrominance
        I_Cr(:,:,i) = I1(:,:,3,i);                         % Store Cr Chrominance
    end
end
% Get the fused Chrominance components by weighted sum
if CG == 3,
    I_Cb128 = abs(double(I_Cb)-128);
    I_Cr128 = abs(double(I_Cr)-128);
    I_CbNew = sum((I_Cb.*I_Cb128)./repmat(sum(I_Cb128,3),[1 1 NumberOfImages]),3);
    I_CrNew = sum((I_Cr.*I_Cr128)./repmat(sum(I_Cr128,3),[1 1 NumberOfImages]),3);
    I_CbNew(isnan(I_CbNew)) = 128;
    I_CrNew(isnan(I_CrNew)) = 128;
end
% Fuse luminance and chrominance part
K1 = L1;
K1(:,:,2) = I_CbNew;
K1(:,:,3) = I_CrNew;
K2 = ycbcr2rgb(uint8(K1));
%%
figure;
subplot 131; imshow(im1,[]); title('Underexposed image');
subplot 132; imshow(im3,[]); title('Overexposure image');
subplot 133; imshow(K2,[]);  title('Fused result');
%% Save result
stop = 0;
remain  =   PathName2;
while stop == 0
    [token, remain] = strtok(remain, '\');
    if strcmp(remain, '\')
        stop    =   1;
    end
end
imwrite(K2, fullfile(savepath, strcat(token, '_result.png')) ,'png');