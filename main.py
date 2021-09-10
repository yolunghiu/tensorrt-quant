import copy

import numpy as np


def get_KL():
    # 随机生成两个离散型分布
    x = [np.random.uniform(1, 11) for i in range(10)]
    px = x / np.sum(x)
    y = [np.random.uniform(1, 11) for i in range(10)]
    py = y / np.sum(y)

    KL = 0.0
    for i in range(10):
        KL += px[i] * np.log(px[i] / py[i])

    if KL < 0.1:
        print(x)

    print(y)

    return KL


def initial_histograms(self, blob_data, bins=2048):
    # collect histogram of every group channel blob
    th = self.blob_max
    hist, hist_edge = np.histogram(blob_data, bins, range=(0, th))
    self.blob_distubution += hist


def _smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor
    and taking the corresponding amount off the non-zero values.
    Ref: http://web.engr.illinois.edu/~hanj/cs412/bk3/KL-divergence.pdf
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    if not n_nonzeros:
        raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist


def threshold_distribution(distribution, target_bin=128):
    """
    Return the best threshold value.
    Ref: https://github.com//apache/incubator-mxnet/blob/master
                    /python/mxnet/contrib/quantization.py
    Args:
        distribution: list, activations has been processed by
        histogram and normalize,size is 2048
        target_bin: int, the num of bin that is used by quantize,
        Int8 default value is 128
    Returns:
        target_threshold: int, num of bin with the minimum KL
    """
    distribution = distribution[1:]
    length = distribution.size  # 2048
    threshold_sum = sum(distribution[target_bin:])  # 128之后所有bin求和
    kl_divergence = np.zeros(length - target_bin)  # 阈值的取值范围是[128, 2047]，

    for threshold in range(target_bin, length):  # 依次遍历128到2048之间的所有阈值
        sliced_nd_hist = copy.deepcopy(distribution[:threshold])  # 阈值之前的hist

        # generate reference distribution p，p可以看成是真实分布
        p = sliced_nd_hist.copy()
        p[threshold - 1] += threshold_sum  # 把阈值之后的bin都加到最后一个bin（使概率为1，超出阈值的值都映射为最大值）
        threshold_sum = threshold_sum - distribution[threshold]

        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (p != 0).astype(np.int64)
        quantized_bins = np.zeros(target_bin, dtype=np.int64)
        # calculate how many bins should be merged to generate quantized distribution q
        # 例如把256个bin合并成128个，则每两个bin进行一次合并
        num_merged_bins = sliced_nd_hist.size // target_bin

        # merge hist into num_quantized_bins bins，q指的是将真实分布p量化后的分布
        for j in range(target_bin):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()

        # expand quantized_bins into p.size bins
        q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
        for j in range(target_bin):
            start = j * num_merged_bins
            if j == target_bin - 1:
                stop = -1
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        # q[p == 0] = 0
        p = _smooth_distribution(p)
        q = _smooth_distribution(q)
        # p[p == 0] = 0.0001
        # q[q == 0] = 0.0001

        # calculate kl_divergence between q and p
        kl_divergence[threshold - target_bin] = stats.entropy(p, q)

    min_kl_divergence = np.argmin(kl_divergence)
    threshold_value = min_kl_divergence + target_bin

    return threshold_value


if __name__ == '__main__':
    blob_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 3, 4, 4, 4]
    th = max(blob_data)
    hist, hist_edge = np.histogram(blob_data, 10, range=(0, th))
    print(sum(hist / sum(hist)))
    print(hist_edge)

    print(255 // 128)
