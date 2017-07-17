function [out_pca,out_npca] = get_scale_subwindow(im, pos, base_target_sz, scaleFactors, scale_model_sz)

nScales = length(scaleFactors);

for s = 1:nScales
    patch_sz = floor(base_target_sz * scaleFactors(s)); % 取图像金字塔的图像块大小
    
    % 图像范围
    xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
    ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);
    
    % 防出界
    % check for out-of-bounds coordinates, and set them to the values at
    % the borders
    xs(xs < 1) = 1;
    ys(ys < 1) = 1;
    xs(xs > size(im,2)) = size(im,2);
    ys(ys > size(im,1)) = size(im,1);
    
    % 取图像
    % extract image
    im_patch = im(ys, xs, :);
    
    % resize image to model size
    % im_patch_resized = imresize(im_patch, scale_model_sz, 'bilinear');
    im_patch_resized = mexResize(im_patch, scale_model_sz, 'auto');
    
    % extract scale features
    % cell size为4，默认bin为9个
    temp_hog = fhog(single(im_patch_resized), 4);
    % 19x26 → 4x6x32
    
    if s == 1
        dim_scale = size(temp_hog,1)*size(temp_hog,2)*31;
        out_pca = zeros(dim_scale, nScales, 'single');
    end
    
    out_pca(:,s) = reshape(temp_hog(:,:,1:31), dim_scale, 1);
    % 4x6x31 → 744x17 一列是一个尺度样本的特征向量
end

out_npca = [];