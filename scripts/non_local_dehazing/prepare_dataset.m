function prepare_dataset(image_size)
    % image_size: GT or LR
    SAVE_PATH = strcat('..\..\dataset\MIT-FiveK\Task\non_local_dehazing\', image_size,'\');
    imgs = dir(strcat('..\..\dataset\MIT-FiveK\Guide\', image_size, '\*.tif'));

    mkdir(SAVE_PATH);
    parfor idx = 1:length(imgs)
        path = fullfile(imgs(idx).folder, imgs(idx).name);
        im = imread(path);

        gamma = 1;
        A = reshape(estimate_airlight(im2double(im).^(gamma)),1,1,3);
        [gt, ~] = non_local_dehazing(im, A, gamma);

        save_path = fullfile(SAVE_PATH, imgs(idx).name);
        imwrite(gt, save_path);
    end
end