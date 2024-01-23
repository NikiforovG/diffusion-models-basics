import numpy as np
import numpy.typing as npt


def unorm(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    # unity norm. results in range of [0,1]
    # assume x (h,w,3)
    xmax = x.max((0, 1))
    xmin = x.min((0, 1))
    return (x - xmin) / (xmax - xmin)  # type:ignore


def norm_all(samples: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    # runs unity norm on all timesteps of all samples
    norm_samples = np.zeros_like(samples)
    for t in range(samples.shape[0]):
        for s in range(samples.shape[1]):
            norm_samples[t, s] = unorm(samples[t, s])
    return norm_samples
