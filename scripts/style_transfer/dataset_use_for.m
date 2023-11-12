clc
clear
image_size = '512';
SAVE_PATH = strcat('E:\JointUpsamplingUsingCSPN\version1\dataset\style_transfer\', image_size, '\');
imgs = dir(strcat('E:\JointUpsamplingUsingCSPN\version1\dataset\guide\', image_size, '\*.tif'));

%% load images
M = imread('images/ruins.png');
M = rgb2gray(double(M)./255);

mkdir(SAVE_PATH);
for idx = 1355:1355
%     num = extractBetween(imgs(idx).name,'a','-');
%     num = str2num(num{1,1});
%     if idx+2 ~= num
%         fprintf("a%d\n",idx);
%         break;
%     end
    path = fullfile(imgs(idx).folder, imgs(idx).name);
    %im = imresize(imread(path),[769 512]);
    im = imread(path);
    fprintf("%s\n",imgs(idx).name);
    % imshow(rgb2gray(im));
    im = rgb2gray(double(im)./65535);
    tic
    [gt, ~] = style_transfer(im, M, 10, 4);
    toc
    save_path = fullfile(SAVE_PATH, imgs(idx).name);
    % imwrite(gt, save_path);
    %imwrite(imresize(gt,[768 512]), save_path);
end
