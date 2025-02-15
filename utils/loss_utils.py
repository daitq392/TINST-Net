import torch
import torch.nn.functional as F


def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    return loss_var_l2


def cal_content_loss(content_features, target_img, model, img_normalize):
    target_features = get_features(img_normalize(target_img), model)
    content_loss = F.mse_loss(target_features['conv4_2'], content_features['conv4_2'])
    return content_loss


def cal_style_loss(style_features, target_img, model, img_normalize):
    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    target_features = get_features(img_normalize(target_img), model)
    style_loss = 0
    for layer in style_layers:
        if layer in style_features and layer in target_features:
            target_gram = gram_matrix(target_features[layer])
            style_gram = gram_matrix(style_features[layer])
            style_loss += F.mse_loss(target_gram, style_gram)
    return style_loss


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)
