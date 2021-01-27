import cv2
import numpy as np
import torch


def letterbox(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    shape = img.shape[:2]  # current shape [height, width]

    if isinstance(inp_dim, int):
        new_shape = (inp_dim, inp_dim)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    # dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    if top != 0 or bottom != 0 or left != 0 or right != 0:
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,
                                                                       114,
                                                                       114))  # add border
    return img, ratio, (dw, dh)


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_img = cv2.imread(img)
    dim = orig_img.shape[1], orig_img.shape[0]
    img = letterbox(orig_img, inp_dim)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    return img, orig_img, dim


def prep_frame(im, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_img = im
    dim = orig_img.shape[1], orig_img.shape[0]
    img = letterbox(orig_img, inp_dim)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    return img, orig_img, dim


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res
