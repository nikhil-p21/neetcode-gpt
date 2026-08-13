import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        e_z = np.exp(z-np.max(z,axis=-1,keepdims=True))
        s_m = e_z/np.sum(e_z,axis=-1,keepdims=True)
        return np.round(s_m,4)
