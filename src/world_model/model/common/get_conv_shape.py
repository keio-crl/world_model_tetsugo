def get_conved_size(
    input_hw: tuple[int, int],
    channels: tuple[int, ...],
    kernels: tuple[int, ...],
    strides: tuple[int, ...],
    paddings: tuple[int, ...],
) -> int:
    h, w = input_hw
    for k, s, p in zip(kernels, strides, paddings):
        h = (h + 2 * p - k) // s + 1
        w = (w + 2 * p - k) // s + 1
    # 最終チャンネル数
    c = channels[-1]
    return c * h * w
