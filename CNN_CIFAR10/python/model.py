
import numpy as np

def convolve(image, kernel, bias, row_stride, col_stride, same=False):
    assert image.ndim == 3 and kernel.ndim == 3, "Image and kernel must be 3-dimensional"
    C_in, H, W = image.shape
    Ck, Kh, Kw = kernel.shape
    assert C_in == Ck, "Kernel channels must match image channels"

    image = image.astype(np.float32, copy=False)
    kernel = kernel.astype(np.float32, copy=False)
    bias = float(bias)
    if same:
        pad_h = (Kh - 1) // 2
        pad_w = (Kw - 1) // 2
        image = np.pad(image, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        H, W = image.shape[1], image.shape[2]

    H_out = int(np.floor((H - Kh) / row_stride)) + 1
    W_out = int(np.floor((W - Kw) / col_stride)) + 1
    out = np.zeros((H_out, W_out), dtype=np.float32)

    row_out = 0
    for row in range(0, H - Kh + 1, row_stride):
        col_out = 0
        for col in range(0, W - Kw + 1, col_stride):
            sub_img = image[:, row:row+Kh, col:col+Kw]
            conv_val = np.sum(sub_img * kernel) + bias  # sum over all channels & spatial dims
            out[row_out, col_out] = conv_val
            col_out += 1
        row_out += 1
    return out

def convolve_batch(image_batch, kernel_batch, bias_batch, row_stride, col_stride, same=False)
    assert image_batch.ndim == 4 and kernel_batch.ndim == 4, "Image and kernel batches must be 4-dimensional"

    batch_size, C_in, H, W = image_batch.shape
    N_k, Ck, Kh, Kw = kernel_batch.shape

    assert bias_batch.shape[0] == N_k, "Number of biases must be the same as the number of kernels"
    assert C_in == Ck, "Kernel channels must match image channels"

    image = image_batch.astype(np.float32, copy=False)
    kernel_batch = kernel_batch.astype(np.float32, copy=False)
    bias_batch = bias_batch.astype(np.float32, copy=False)

    if same:
        pad_h = (Kh - 1) // 2
        pad_w = (Kw - 1) // 2
        image = np.pad(image, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        H, W = image.shape[1], image.shape[2]

    H_out = int(np.floor((H - Kh) / row_stride)) + 1
    W_out = int(np.floor((W - Kw) / col_stride)) + 1
    out = np.zeros((batch_size, N_k, H_out, W_out), dtype=np.float32)


    for imagenum, image in enumerate(image_batch):
        for kernelnum, kernel in enumerate(kernel_batch):
            bias = bias_batch[kernelnum]
            out[imagenum, kernelnum] = convolve(image, kernel, bias, row_stride, col_stride, same=False)

    return out

