import numpy as np
import torch
import copy
from scipy.interpolate import CubicSpline


class TimeSeriesAugmenter:
    """
    A class for time series data augmentation, supporting multiple input shapes and backends (NumPy or PyTorch).

    Supported input shapes:
    - [batch, channels, time] → e.g., (N, 31, 104)
    - [batch, time, channels] → e.g., (N, 104, 31)
    - [channels, time]        → e.g., (31, 104)
    - [time, channels]        → e.g., (104, 31)
    - [time]                  → e.g., (104,)

    Parameters
    ----------
    time_axis : int
        Index of the time dimension
    channel_axis : int
        Index of the channel dimension

    Example usage
    -------------
    augmenter = TimeSeriesAugmenter(time_axis=2, channel_axis=1)
    X_aug = augmenter.jitter(X)  # Adds Gaussian noise
    X_perm, Y_perm = augmenter.permutation(X, Y, nPerm=4)  # Permutation with labels
    """

    def __init__(self, time_axis=2, channel_axis=1):
        self.time_axis = time_axis
        self.channel_axis = channel_axis

    # ---------------------- Backend helpers ----------------------
    def _get_backend(self, X):
        if isinstance(X, torch.Tensor):
            return torch
        elif isinstance(X, np.ndarray):
            return np
        else:
            raise TypeError(f"Unsupported data type: {type(X)}")

    def _to_numpy(self, X):
        if isinstance(X, np.ndarray):
            return X
        elif isinstance(X, torch.Tensor):
            return X.detach().cpu().numpy()
        else:
            raise TypeError(f"Unsupported data type: {type(X)}")

    def _to_tensor(self, X, reference=None):
        if reference is None:
            return X
        if isinstance(reference, np.ndarray):
            return np.array(X) if not isinstance(X, np.ndarray) else X
        if isinstance(reference, torch.Tensor):
            device = reference.device
            dtype = reference.dtype
            if isinstance(X, torch.Tensor):
                return X.to(device=device, dtype=dtype)
            return torch.tensor(X, device=device, dtype=dtype)
        return X

    def _randn_like_backend(self, xp, shape, mean=0.0, std=1.0, device=None):
        if xp.__name__ == "numpy":
            return xp.random.normal(mean, std, size=shape)
        elif xp.__name__ == "torch":
            return mean + std * xp.randn(shape, device=device)
        else:
            raise ValueError("Backend not supported.")
            
            

    # ---------------------- Shape helpers ----------------------
    def _ensure_3d(self, X):
        """Normalize any input shape to [batch, channels, time]"""
        xp = self._get_backend(X)
        orig_shape = X.shape

        # 1D → (1,1,time)
        if X.ndim == 1:
            Xn = X[xp.newaxis, xp.newaxis, :]
            return Xn, orig_shape

        # 2D → [channels,time] or [time,channels]
        if X.ndim == 2:
            ch_axis = self.channel_axis if self.channel_axis is not None else 0
            t_axis = self.time_axis if self.time_axis is not None else 1
            Xn = xp.moveaxis(X, [ch_axis, t_axis], [1, 2])
            Xn = Xn[xp.newaxis, ...]  # add batch
            return Xn, orig_shape

        # 3D → move batch/channel/time
        if X.ndim == 3:
            n_dims = X.ndim
            ch_axis = self.channel_axis % n_dims
            t_axis = self.time_axis % n_dims
            remaining = [i for i in range(n_dims) if i not in [ch_axis, t_axis]]
            batch_axis = remaining[0] if remaining else 0
            Xn = xp.moveaxis(X, [batch_axis, ch_axis, t_axis], [0, 1, 2])
            return Xn, orig_shape

        raise ValueError(f"Unsupported input shape: {X.shape}")

    def _restore_shape(self, Xn, orig_shape):
        """Restore original shape after augmentation"""
        xp = self._get_backend(Xn)
        if len(orig_shape) == 1:
            return Xn.reshape(orig_shape)
        if len(orig_shape) == 2:
            ch_axis = self.channel_axis if self.channel_axis is not None else 0
            t_axis = self.time_axis if self.time_axis is not None else 1
            Xn = Xn[0]
            return xp.moveaxis(Xn, [1, 2], [ch_axis, t_axis])
        if len(orig_shape) == 3:
            n_dims = len(orig_shape)
            ch_axis = self.channel_axis % n_dims
            t_axis = self.time_axis % n_dims
            remaining = [i for i in range(n_dims) if i not in [ch_axis, t_axis]]
            batch_axis = remaining[0] if remaining else 0
            return xp.moveaxis(Xn, [0, 1, 2], [batch_axis, ch_axis, t_axis])
        return Xn

    def _is_dataset(self, obj):
        """Verifica se o objeto parece um dataset com .X e .Y"""
        return hasattr(obj, "X") and hasattr(obj, "Y")
    
    def _concat_backend(self, xp, tensors, axis=0):
        """Concatena de forma agnóstica (numpy/torch)."""
        if xp == np:
            return np.concatenate(tensors, axis=axis)
        elif xp == torch:
            return torch.cat(tensors, dim=axis)
        else:
            raise TypeError("Unsupported backend for concatenation.")


    # ---------------------- Augmentations ----------------------
    def jitter(self, X, sigma=0.05):
        """Add Gaussian noise"""
        xp = self._get_backend(X)
        Xn, orig_shape = self._ensure_3d(X)
        noise = self._randn_like_backend(xp, Xn.shape, std=sigma, device=getattr(X, 'device', None))
        return self._to_tensor(self._restore_shape(Xn + noise, orig_shape), X)

    def scaling(self, X, sigma=0.1):
        """Scale each channel independently"""
        xp = self._get_backend(X)
        Xn, orig_shape = self._ensure_3d(X)
        scalingFactor = self._randn_like_backend(xp, (Xn.shape[1], 1), mean=1.0, std=sigma, device=getattr(X, 'device', None))
        Xn = Xn * scalingFactor[None, :, :]
        return self._to_tensor(self._restore_shape(Xn, orig_shape), X)

    def mag_warp(self, X, sigma=0.2, knot=4):
        """Magnitude warping along time"""
        Xn, orig_shape = self._ensure_3d(X)
        n_steps = Xn.shape[2]
        random_warps = np.random.normal(1.0, sigma, size=(Xn.shape[0], knot+2, Xn.shape[1]))
        warp_steps = np.linspace(0, n_steps-1, num=knot+2)
        orig_steps = np.arange(n_steps)
        X_np = self._to_numpy(Xn)
        ret = np.zeros_like(X_np)
        for i, pat in enumerate(X_np):
            warper = np.array([CubicSpline(warp_steps, random_warps[i,:,dim])(orig_steps) for dim in range(X_np.shape[1])])
            ret[i] = pat * warper
        return self._to_tensor(self._restore_shape(ret, orig_shape), X)

    def time_warp(self, X, sigma=0.2, knot=4):
        """Time warping"""
        Xn, orig_shape = self._ensure_3d(X)
        n_steps = Xn.shape[2]
        random_warps = np.random.normal(1.0, sigma, size=(Xn.shape[0], knot+2, Xn.shape[1]))
        warp_steps = np.linspace(0, n_steps-1, num=knot+2)
        orig_steps = np.arange(n_steps)
        X_np = self._to_numpy(Xn)
        ret = np.zeros_like(X_np)
        for i, pat in enumerate(X_np):
            for dim in range(X_np.shape[1]):
                time_warp = CubicSpline(warp_steps, warp_steps*random_warps[i,:,dim])(orig_steps)
                scale = (n_steps-1)/time_warp[-1]
                ret[i, dim] = np.interp(orig_steps, np.clip(scale*time_warp,0,n_steps-1), pat[dim])
        return self._to_tensor(self._restore_shape(ret, orig_shape), X)

    def permutation(self, X, Y=None, nPerm=4, minSegLength=10):
        """Permutation along the time axis with optional labels"""
        xp = self._get_backend(X)
        Xn, orig_shape = self._ensure_3d(X)
        n_steps = Xn.shape[2]
        assert n_steps >= nPerm*minSegLength, "Not enough time steps for permutation"

        while True:
            segs = xp.zeros(nPerm+1, dtype=int)
            if xp.__name__ == "numpy":
                rand_vals = xp.random.randint(minSegLength, n_steps-minSegLength, (nPerm-1,))
                segs[1:-1] = xp.sort(rand_vals)
            else:
                rand_vals = xp.randint(minSegLength, n_steps-minSegLength, (nPerm-1,), device=Xn.device)
                segs[1:-1] = xp.sort(rand_vals).values
            segs[-1] = n_steps
            if xp.min(segs[1:]-segs[:-1]) > minSegLength:
                break

        X_new = xp.zeros_like(Xn)
        Y_new = None
        if Y is not None:
            if Y.ndim == 1:
                Y_new = Y.clone() if hasattr(Y,"clone") else Y.copy()
            else:
                Y_new = xp.zeros_like(Y)

        for i, pat in enumerate(Xn):
            idx = xp.randperm(nPerm) if xp.__name__=="torch" else xp.random.permutation(nPerm)
            pp = 0
            for ii in idx:
                seg_start, seg_end = segs[ii].item(), segs[ii+1].item()
                seg_len = seg_end - seg_start
                X_new[i,:,pp:pp+seg_len] = pat[:,seg_start:seg_end]
                if Y is not None and Y.ndim>1:
                    Y_new[i, pp:pp+seg_len] = Y[i, seg_start:seg_end]
                pp += seg_len
        if Y is not None and Y.ndim==1:
            order = xp.randperm(Xn.shape[0]) if xp.__name__=="torch" else xp.random.permutation(Xn.shape[0])
            X_new = X_new[order]
            Y_new = Y_new[order]

        X_out = self._to_tensor(self._restore_shape(X_new, orig_shape), X)
        return (X_out, Y_new) if Y is not None else X_out
    
   
    
    def augment_dataset(self, data, data_augmentation, Y=None, append_to_dataset=True):
        """
        Apply multiple augmentations to either:
        - a dataset object with attributes X and Y
        - or direct arrays/tensors X and Y
        
        Parameters
        ----------
        data : object or array/tensor
            Either a dataset instance (with X, Y attributes) or a tensor/array representing X.
        data_augmentation : list of str
            List of augmentation techniques to apply. Example: ['Jitter', 'Scale', 'Perm'].
        Y : array/tensor, optional
            Labels corresponding to X, required if data is not a dataset instance.
        append_to_dataset : bool, default=True
            If True, augmentations are appended to the dataset (only if dataset object).
            If False, returns only the augmented samples.
        
        Returns
        -------
        - If append_to_dataset=True and input is dataset: returns modified dataset
        - If append_to_dataset=False: returns (aug_X, aug_Y)
        """
    
        # --- Detecta se a entrada é um dataset ou um tensor direto ---
        if self._is_dataset(data):
            dataset = data
            X_data, Y_data = dataset.X, dataset.Y
            backend = self._get_backend(X_data)
        else:
            X_data, Y_data = data, Y
            backend = self._get_backend(X_data)
            if Y_data is None:
                raise ValueError("Y must be provided when passing X directly (not a dataset).")
    
        # Cópia para evitar alterar o original
        X_copy = copy.deepcopy(X_data)
        Y_copy = copy.deepcopy(Y_data)
        augmented_X_list, augmented_Y_list = [], []
    
        # --- Loop das técnicas ---
        for aug in data_augmentation:
            if aug == 'Jitter':
                print("Applying JITTER")
                X_tmp = self.jitter(X_copy)
                Y_tmp = Y_copy
            elif aug == 'Scale':
                print("Applying SCALE")
                X_tmp = self.scaling(X_copy)
                Y_tmp = Y_copy
            elif aug == 'Perm':
                print("Applying PERMUTATION")
                X_tmp, Y_tmp = self.permutation(X_copy, Y_copy)
            elif aug == 'TimeW':
                print("Applying TIME WARP")
                X_tmp = self.time_warp(X_copy)
                Y_tmp = Y_copy
            elif aug == 'MagW':
                print("Applying MAGNITUDE WARP")
                X_tmp = self.mag_warp(X_copy)
                Y_tmp = Y_copy
            else:
                continue
    
            augmented_X_list.append(X_tmp)
            augmented_Y_list.append(Y_tmp)
    
            # Se for dataset e append=True
            if self._is_dataset(data) and append_to_dataset:
                dataset.add_sample(X_tmp, Y_tmp)
    
        # --- Retorno ---
        if append_to_dataset and self._is_dataset(data):
            return dataset
    
        # Caso contrário, retorna apenas as amostras aumentadas
        aug_X = self._concat_backend(backend, augmented_X_list, axis=0)
        aug_Y = self._concat_backend(backend, augmented_Y_list, axis=0)
        return aug_X, aug_Y
