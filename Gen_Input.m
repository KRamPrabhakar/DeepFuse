input_fol = './input_color_images/2Imgs_new/';
savepath = './input_test_pairs/';
folders = dir(input_fol);
for indx0 = 3:length(folders)
    folder = folders(indx0).name;
    img1 = imread(fullfile(input_fol,folder,'Img_EV-4.jpg'));
    img2 = imread(fullfile(input_fol,folder,'Img_EV4.jpg'));
    [r,c,~] = size(img1);
    if max(r,c) > 512
        ratio = 512/max(r,c);
    else
        ratio = 1;
    end
    img1 = rgb2ycbcr(imresize(img1, ratio));    
    img2 = rgb2ycbcr(imresize(img2, ratio));
    data(:,:,1) = img1(:,:,1);
    data(:,:,2) = img2(:,:,1);
    data(:,:,3) = img2(:,:,1);
    str_name = strcat(folder,'.png');
    imwrite(data, fullfile(savepath,str_name),'png');
    clear data img1 img2
end