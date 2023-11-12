function prepare_dataset(image_size)
    %% parameters (SP-2 mode)
    addpath("funs\");
    smallNum = 1e-3;  % try 1e-4 for different smoothing results
    lambda = 0.5;
    thr = 1;  % this value should be larger than the maximum intensity value
    rData = 2;
    rSmooth = 2;
    aData = smallNum;
    aSmooth = smallNum;
    bData = thr;
    bSmooth = thr;
    alpha = 0.5;
    stride = 1;
    iterNum = 5;
    % image_size: GT or LR
    SAVE_PATH = strcat('..\..\dataset\MIT-FiveK\Task\GSF\', image_size,'\');
    imgs = dir(strcat('..\..\dataset\MIT-FiveK\Guide\', image_size, '\*.tif'));

    mkdir(SAVE_PATH);
    parfor idx = 1:length(imgs)
        path = fullfile(imgs(idx).folder, imgs(idx).name);
        im = imread(path);
        %% smooth
        Out_SP = generalized_smooth(im, im, lambda, rData, rSmooth, aData, bData, aSmooth, bSmooth, alpha, stride, iterNum);
        gt = uint8(Out_SP);

        save_path = fullfile(SAVE_PATH, imgs(idx).name);
        imwrite(gt, save_path);
    end
end