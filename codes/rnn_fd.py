"""
Recurrent Neural Network (RNN) where each cell is Finite Difference (FD) operator.

Please cite this reference: 
Jian Sun, Zhan Niu, Kristopher A. Innanen, Junxiao Li, and Daniel O. Trad, (2020), 
"A theory-guided deep-learning formulation and optimization of seismic waveform inversion," 
GEOPHYSICS 85: R87-R99.

@author: jiansun
- Penn State
- Updated By Jian on Feb. 6, 2020
"""
import numpy as np
import torch
import torch.nn.functional as F
import math

class convModel(torch.nn.Module):
    """
    A class for the convolution model, 
        i.e., seismogram = reflectivity @*@ wavelet.
    """
    def __init__(self, wavelet, dz, 
                 dtnew=0.001, 
                 ntnew=1000, 
                 dtype=torch.float32, device='cpu'):
        """
        Args:
            dz in m
            wavelet(tensor):    size [nt]
        """
        super(convModel, self).__init__()
        self.dz = dz
        self.dtnew = dtnew
        self.ntnew = ntnew
        self.dtype = dtype
        self.device = device
        self.halflength_wavelet = len(wavelet) // 2

        # padding zeros on wavelet preparing for 
        # time-domain convlution performing in frequency domain
        # after convolution, length should be wavelet.shape[1] + ntnew -1
        # self.wavelet_freq has size: [1, nt + ntnew-1, 2], 
        # the last dimension has real and image part for complex number
        self.wavelet = torch.cat([wavelet.type(dtype), torch.zeros(ntnew - 1, dtype=dtype, device=wavelet.device)], dim=0)   
        self.wavelet_freq = torch.view_as_real(torch.fft.fft(self.wavelet))[None, ...]
        
    def forward(self, vmodel, density=None):
        """
        units of dz and vmodel must be set correspondingly.
        1st, relfectivity needs to be obtained using vmodel.
        2nd, convert depth-related reflectivity into time domain.
        3rd, perform time-domain convolution of reflectivity and wavelet in frequency domain.
        """
        # 1st, calculate reflectivity with output size: [-1, nz-1]
        if density is not None:
            vmodel = vmodel * density
        reflectivity_depth = (vmodel[:, 1:] - vmodel[:, :-1]) / (vmodel[:, 1:] + vmodel[:, :-1])

        # 2nd, convert reflectivity from depth into time domain
        reflectivity_time = torch.zeros((vmodel.shape[0], self.wavelet.shape[0]), 
                                        dtype=self.dtype, device=self.device)
        
        dt_vector = self.dz / vmodel[:, :-1]                            # shape: [-1, nz-1]
        time_old = torch.cumsum(dt_vector, dim=1)                       # old time vector for reflectivity
        
        newtime_index = torch.round(time_old / self.dtnew).long()       # new time index
        newtime_index[newtime_index < 0] = -100
        newtime_index[newtime_index >= self.ntnew] = -100
        row_index, col_index = torch.where(newtime_index != -100)       # specify [0, ntnew-1] length of relfectivity
        reflectivity_time[row_index, newtime_index[row_index, col_index].view(-1)] \
            = reflectivity_depth[row_index, col_index].view(-1)
        
        # plt.plot(np.arange(0, self.wavelet.shape[0])*self.dtnew, reflectivity_time.cpu().numpy()[0])
        # 3rd, perform time convolution using frequency multiplication.
        reflectivity_freq = torch.view_as_real(torch.fft.fft(reflectivity_time, dim=1))
        seismogram_freq = self.complex_product(reflectivity_freq, self.wavelet_freq)
        seismogram = torch.fft.irfft(torch.view_as_complex(seismogram_freq.contiguous()), n=seismogram_freq.shape[1], dim=1)
        # return seismogram[:, self.halflength_wavelet - 1:-self.halflength_wavelet]
        return seismogram[:, :self.ntnew]
    
    def complex_product(self, x, y):
        """
        Performs product of complex number.
            where the last dimension of x and y equals to 2, 
            which represents real and image parts of a complex number.
        """
        ndim = len(x.shape) - 1
        x = x.transpose(ndim, 0)
        y = y.transpose(ndim, 0)

        real = x[0] * y[0] - x[1] * y[1]
        imag = x[0] * y[1] + x[1] * y[0]
        xy = torch.cat([real[None, :], imag[None, :]], dim=0).transpose(ndim, 0)
        return xy

        
class convModel_oldPytorch(torch.nn.Module):
    """
    A class for the convolution model, 
        i.e., seismogram = reflectivity @*@ wavelet.
    """
    def __init__(self, wavelet, dz, 
                 dtnew=0.001, 
                 ntnew=1000, 
                 onesided=False, 
                 dtype=torch.float32, device='cpu'):
        """
        Args:
            dz in m
            wavelet(tensor):    size [nt]
        """
        super(convModel_oldPytorch, self).__init__()
        self.dz = dz
        self.dtnew = dtnew
        self.ntnew = ntnew
        self.dtype = dtype
        self.device = device
        self.onesided = onesided
        self.halflength_wavelet = len(wavelet) // 2

        # padding zeros on wavelet preparing for 
        # time-domain convlution performing in frequency domain
        # after convolution, length should be wavelet.shape[1] + ntnew -1
        # self.wavelet_freq has size: [1, nt + ntnew-1, 2], 
        # the last dimension has real and image part for complex number
        self.wavelet = torch.cat([wavelet.type(dtype), torch.zeros(ntnew - 1, dtype=dtype, device=wavelet.device)], dim=0)   
        self.wavelet_freq = torch.rfft(self.wavelet, signal_ndim=1, onesided=self.onesided)[None, :, :]
        
    def forward(self, vmodel, density=None):
        """
        units of dz and vmodel must be set correspondingly.
        1st, relfectivity needs to be obtained using vmodel.
        2nd, convert depth-related reflectivity into time domain.
        3rd, perform time-domain convolution of reflectivity and wavelet in frequency domain.
        """
        # 1st, calculate reflectivity with output size: [-1, nz-1]
        if density is not None:
            vmodel = vmodel * density
        reflectivity_depth = (vmodel[:, 1:] - vmodel[:, :-1]) / (vmodel[:, 1:] + vmodel[:, :-1])

        # 2nd, convert reflectivity from depth into time domain
        reflectivity_time = torch.zeros((vmodel.shape[0], self.wavelet_freq.shape[1]), 
                                        dtype=self.dtype, device=self.device)
        dt_vector = self.dz / vmodel[:, :-1]                            # shape: [-1, nz-1]
        time_old = torch.cumsum(dt_vector, dim=1)                       # old time vector for reflectivity
        newtime_index = torch.round(time_old / self.dtnew).long()       # new time index
        newtime_index[newtime_index < 0] = -100
        newtime_index[newtime_index >= self.ntnew] = -100
        row_index, col_index = torch.where(newtime_index != -100)       # specify [0, ntnew-1] length of relfectivity
        reflectivity_time[row_index, newtime_index[row_index, col_index].view(-1)] \
            = reflectivity_depth[row_index, col_index].view(-1)

        # 3rd, perform time convolution using frequency multiplication.
        reflectivity_freq = torch.rfft(reflectivity_time, signal_ndim=1, onesided=self.onesided)
        seismogram_freq = self.complex_product(reflectivity_freq, self.wavelet_freq)
        if self.onesided:
            if seismogram_freq.shape[1] % 2 == 0:   # even number
                trunc_tensor = seismogram_freq[:, 1:-1]
            else:
                trunc_tensor = seismogram_freq[:, 1:]
            inv_index = torch.arange(trunc_tensor.shape[1] - 1, -1, -1).long()
            inv_tensor = trunc_tensor[inv_index]
            seismogram_freq = torch.cat([seismogram_freq, inv_tensor], dim=1)
        seismogram = torch.irfft(seismogram_freq, signal_ndim=1, onesided=False)
        return seismogram[:, self.halflength_wavelet - 1:-self.halflength_wavelet]

    def complex_product(self, x, y):
        """
        Performs product of complex number.
            where the last dimension of x and y equals to 2, 
            which represents real and image parts of a complex number.
        """
        ndim = len(x.shape) - 1
        x = x.transpose(ndim, 0)
        y = y.transpose(ndim, 0)

        real = x[0] * y[0] - x[1] * y[1]
        imag = x[0] * y[1] + x[1] * y[0]
        xy = torch.cat([real[None, :], imag[None, :]], dim=0).transpose(ndim, 0)
        return xy


""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
"""                                     1D propagator (single time step)                                                   """
""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """


class Propagator1D(torch.nn.Module):
    """
    This class only performs a single time-step wave propagation,
         where, num_vels is considered as batch_size;
                num_shots is considered as channels.
    """
    def __init__(self, nz, dz, dt, 
                 npad=0, order=2, 
                 freeSurface=True, 
                 dtype=torch.float32, 
                 device='cpu'):
        super(Propagator1D, self).__init__()
        self.dtype = dtype
        self.device = device
        self.k_size = order + 1
        self.npad = npad
        self.freeSurface = freeSurface
        if freeSurface:
            self.nz_pad = nz + npad
        else:
            self.nz_pad = nz + 2 * npad
            
        # Setup convolutional kernel (default: 2nd-order 3*1)
        if order == 4:
            # Laplacian kernel for 4th-order FD forward propagation.
            self.kernel1D = self.__make_tensor1D([-(dt / dz)**2 / 12,
                                                 (dt / dz)**2 / 12 * 16,
                                                 (dt / dz)**2 / 12 * (-30),
                                                 (dt / dz)**2 / 12 * 16,
                                                 (dt / dz)**2 / 12 * (-1)])
        else:
            # Laplacian kernel for 2nd-order FD forward propagation (default).
            self.kernel1D = self.__make_tensor1D([(dt / dz)**2,
                                                 -2 * ((dt / dz)**2),
                                                 (dt / dz)**2])
        
        # #### Make kernelX and kernelZ for regularizer ####
        self.kernelZ = self.__make_tensor1D([-1 / 2, 0, 1 / 2])
                
    def __make_tensor1D(self, a):
        # Prepare a filter in shape of [1, 1, Width]
        a = np.asarray(a)
        a = a.reshape([1, 1] + list(a.shape))
        return torch.as_tensor(a, dtype=self.dtype, device=self.device)
        
    def __sponge_bcMask(self):
        # Keep tmp as numpy array, because tensor does not support negative step like [::-1]
        tmp = np.exp(-0.0015**2 * np.arange(self.npad, 0, -1)**2)  # small -- > large
        wmask = np.ones([self.nz_pad, 1])
        # add bottom_mask
        wmask[-self.npad:, :] *= tmp[::-1][:, None]
        if self.freeSurface is False:
            # add top_mask
            wmask[:self.npad, :] *= tmp[:, None]
        return torch.as_tensor(wmask, dtype=self.dtype, device=self.device)
    
    def ___tensor_pad(self, input_tensor):
        """
        This function is to padding velocity tensor for implementing the absorbing boudary condition.
            input_tensor: is a 3D tensor, shape=[num_vels, 1, nz]
            output_tensor: is also a 3D tensor, shape=[num_vels, 1, self.nz_pad]
        """   
        batch_size = input_tensor.shape[0]
        if self.freeSurface:
            vpadTop = input_tensor
        else:
            vtop = torch.ones((batch_size, 1, self.npad), device=self.device) * input_tensor[:, :, :1]
            vpadTop = torch.cat((vtop, input_tensor), -1)  # padding on axis=2 (nz)
        
        vbottom = torch.ones((batch_size, 1, self.npad), device=self.device) * input_tensor[:, :, -1:]
        output_tensor = torch.cat([vpadTop, vbottom], -1)  # padding on axis=2 (nz)
        return output_tensor
        
    def ___step_rnncell(self, source, u_prev, u_):
        """
        This function is to implement the forward propagation for a single time-step.
        Input:
            u_prev: state at time step t-dt,   in shape of [batch_size, self.num_shots, self.nz_pad]
            u_:     state at time step t, also in shape of [batch_size, self.num_shots, self.nz_pad]

            source: a list of tensor: [fs, zs]; 
                    where fs: [batch_size], zs: [batch_size, self.num_shots]
        Output: 
            u_next: state at time step t+dt
        """        
        # For conv1d: out_size = [in_size + 2*pad_size - kernel_size - (kernel_size-1)*(dilation-1)]/sride + 1
        # To achieve "same", the pad_size we need is: (dilation=1, stride=1)
        # padding=(self.k_size-1)/2
        u_partial = F.conv1d(u_, self.kernel1D.repeat(u_.shape[1], 1, 1), padding=(self.k_size - 1) // 2, groups=u_.shape[1])
        u_next = u_partial * self.velocity**2 + 2 * u_ - u_prev
        
        # Adding source
        for ivel in range(u_next.shape[0]):
            for ishot in range(u_next.shape[1]):
                if self.freeSurface:
                    u_next[ivel, ishot, source[1][ivel, ishot]] += source[0][ivel]
                else:
                    u_next[ivel, ishot, source[1][ivel, ishot] + self.npad] += source[0][ivel]

        # Applying absorbing boundary mask
        if self.npad != 0:
            wmask = self.__sponge_bcMask().reshape([1, 1, self.nz_pad])
            u_next = u_next * wmask
        return u_, u_next  

    def forward(self, vel_noPad, sources_info, prev_wavefield, curr_wavefield):
        """
        Forward propagating for a single time-step from (t-dt & t --> t+dt).
        Input:
            sources_info: a list contains [wavelet, zs, zr],
                          wavelet(tensor): shape [num_vels]
                          zs(tensor): shape [num_vels, num_shots]
                          zr(tensor): shape [num_vels, num_shots]
            Initial states (tensor):
                    1. prev_wavefield(tensor): wavefield at time step t-dt
                    2. curr_wavefield(tensor): wavefield at time step t
                       size: [num_vels, num_shots, self.nz_pad]
            vel_noPad: A PyTorch tensor for velocity model, shape = [num_vels, nz].
                    - Initial velocity model for inversion process, i.e., requires_grad=True.
                    - True velocity model for forward modeling propagation.
        Output:
            Save prev_wavefield & curr_wavefield for next time step prop.
            yt_pred: [num_vels, num_shots] because for 1D, only 1 receiver for 1 shot.
                extracting seismogram at receiver locations using zr.
        """
        # Padding the velocity model (Should repadding velocity every time for absorbing)
        self.velocity = self.___tensor_pad(vel_noPad[:, None, :])  # After shape: [num_vels, 1, nz_pad]

        row = torch.arange(prev_wavefield.shape[0])[:, None].repeat(1, prev_wavefield.shape[1])
        col = torch.arange(prev_wavefield.shape[1])[None, :].repeat(prev_wavefield.shape[0], 1)

        prev_wavefield, curr_wavefield = self.___step_rnncell(sources_info[:2], prev_wavefield, curr_wavefield)
        if self.freeSurface:
            yt_pred = curr_wavefield[row.view(-1), col.view(-1), sources_info[2].view(-1)].reshape(prev_wavefield.shape[:2])
        else:
            yt_pred = curr_wavefield[row.view(-1), col.view(-1), sources_info[2].view(-1) + self.npad].reshape(prev_wavefield.shape[:2])

        # Add Regularization term, shape: [num_vels, 1]
        # 1. for sparse model
        # regularizer = vel_noPad.pow(2).sum(dim=-1)  
        # 2.1 for flatest model (1D derivative)
        # gradVel = vel_noPad[:, :, 1:] - vel_noPad[:, :, :-1]
        # regularizer = gradVel.pow(2).sum(dim=-1)  
        # 2.2 for flatest model (2D derivatives)
        # gradVelZ = F.conv1d(vel_noPad, self.kernelZ, padding=0)
        # regularizer = gradVelZ.pow(2).sum(dim=-1)
        regularizer = torch.tensor([[0]], dtype=self.dtype, device=self.device)
        return prev_wavefield, curr_wavefield, yt_pred, regularizer


""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
"""                                     1D RNN forward modeling (full/truncated time step)                                 """
""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """


class rnn1D(torch.nn.Module):
    """
    Using Propagator1D to perform the forward modeling with given wavelets through multiple time-steps
    """
    def __init__(self, nz, zs, zr, dz, dt, 
                 npad=0, 
                 order=2, 
                 vmax=6000, 
                 freeSurface=True, 
                 dtype=torch.float32, 
                 device='cpu'):
        super(rnn1D, self).__init__()
        """
        vpred(tensor): require_grad=True, shape: [num_vels, nz]
        zs(tensor): in shape of [num_vels, num_shots]
        zr(tensor): in shape of [num_vels, num_shots]
        """
        # Stability condition for forward modeling (Lines et al., 1999, Geophysics)
        # vmax*dt/dz < XXX
        # for 1D, 2nd-order: 1; 4th-order: np.sqrt(3)/2
        # for 2D, 2nd-order: 1/np.sqrt(2); 4th-order: np.sqrt(3/8)
        # for 3D, 2nd-order: 1/np.sqrt(3); 4th-order: 1/2
        if order == 2:
            # print("Stability Condition: {} < {}".format(vpred.max().cpu() * dt / dz, 1))
            assert vmax * dt / dz < 1, "Current parameters setting do NOT meet the stability condition."
        elif order == 4:
            # print("Stability Condition: {} < {}".format(vpred.max().cpu() * dt / dz, np.sqrt(3) / 2))
            assert vmax * dt / dz < np.sqrt(3) / 2, "Current parameters setting do NOT meet the stability condition."
        self.dtype = dtype
        self.device = device
        self.zs, self.zr = zs, zr
        if freeSurface:
            self.nz_pad = nz + npad
        else:
            self.nz_pad = nz + 2 * npad
        
        # define finite-difference operator    
        self.fd = Propagator1D(nz, dz, dt, npad, order, freeSurface, dtype, device).to(device)
        
    def forward(self, vmodel, segment_wavelet, prev_state=None, curr_state=None, option=0):
        """
        Input:
            prev_state(tensor): 3D tensor, shape: [num_vels, num_shots, nz_pad]
            curr_state(tensor): 3D tensor, shape: [num_vels, num_shots, nz_pad]
            segment_wavelet(tensor): [num_vels, len_tSeg] or [len_tSeg]
        Ouput:
            prev_state & curr_state for next time-segment, which is depending on option you choose.
            segment_ytPred(tensor): [num_vels, num_shots, len_tSeg]
        """
        if prev_state is None:
            prev_state = torch.zeros([vmodel.shape[0], self.zs.shape[1], self.nz_pad], dtype=self.dtype, device=self.device)
            curr_state = torch.zeros([vmodel.shape[0], self.zs.shape[1], self.nz_pad], dtype=self.dtype, device=self.device)

        # if segment_wavelet.ndim == 1:   # Not sure why (PyTorch version?), but this does not works on CyberLAMP cluster.
        if len(segment_wavelet.shape) == 1:
            segment_wavelet = segment_wavelet.repeat(prev_state.shape[0], 1)
        
        segment_ytPred = []
        avg_regularizer = []
        for it in range(segment_wavelet.shape[1]):
            # in case that zs & zr are not matched to the batch_size=num_vels
            segment_sources = [segment_wavelet[:, it], self.zs[:vmodel.shape[0], :], self.zr[:vmodel.shape[0], :]]
            prev_state, curr_state, seg_ytPred, regularizer = self.fd(vmodel, segment_sources, prev_state, curr_state)

            segment_ytPred.append(seg_ytPred)
            avg_regularizer.append(regularizer)
            # for option 1, we want save the middle states for next time segement
            if option == 1 and it == (len(segment_sources[0]) - 1) // 2:
                prev_save = prev_state.detach().clone()
                curr_save = curr_state.detach().clone()

        # for option 0, we save the last two wavefields for next time-segment
        if option == 0:
            prev_save = prev_state
            curr_save = curr_state
        # for next time segement, option 2 makes sure it always start from time_step 0 with zero initials
        elif option == 2:
            prev_save = prev_state.new_zeros(prev_state.shape, dtype=self.dtype, device=self.device)
            curr_save = curr_state.new_zeros(curr_state.shape, dtype=self.dtype, device=self.device)
        
        segment_ytPred = torch.stack(segment_ytPred, dim=-1)  # shape: [num_vels, num_shots, len_tSeg]
        avg_regularizer = torch.stack(avg_regularizer, dim=-1).mean(dim=-1)  # shape:[num_vels]
        return prev_save, curr_save, segment_ytPred, avg_regularizer


""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
"""                                     2D propagator (single time step)                                                   """
""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """


class Propagator2D(torch.nn.Module):
    """
    This class only performs a single time-step wave propagation,
         where, num_vels is considered as batch_size;
                num_shots is considered as channels.
    """
    def __init__(self, nz, nx, dz, dt, 
                 npad=0, order=2, 
                 freeSurface=True, 
                 log_para=1e-6, vmax=6000,
                 dtype=torch.float32, device='cpu'):
        super(Propagator2D, self).__init__()
        self.dtype = dtype
        self.device = device
        self.vmax = vmax
        self.k_size = order + 1
        self.nx, self.npad = nx, npad
        self.nx_pad = nx + 2 * npad
        self.freeSurface = freeSurface
        self.log_para = log_para
        if freeSurface:
            self.nz_pad = nz + npad
        else:
            self.nz_pad = nz + 2 * npad
        self.pad = npad

        # Setup convolutional kernel (default: 2nd-order 3*3)
        dx = dz
        self.dt = dt
        self.dz = dz
        self.dx = dz
        if order == 4:
            # Laplacian kernel for 4th-order FD forward propagation.
            self.kernel2D = self.___make_tensor([[0, 0, -(dt / dz)**2 / 12, 0, 0],
                                                [0, 0, (dt / dz)**2 / 12 * 16, 0, 0],
                                                [-(dt / dx)**2 / 12, (dt / dx)**2 / 12 * 16, 
                                                 -30 / 12 * ((dt / dx)**2 + (dt / dz)**2), 
                                                 (dt / dx)**2 / 12 * 16, -(dt / dx)**2 / 12],
                                                [0, 0, (dt / dz)**2 / 12 * 16, 0, 0],
                                                [0, 0, -(dt / dz)**2 / 12, 0, 0]])
            self.kernel2Dx = self.___make_tensor([[0 ,      0,       0],
                                                  [-1/(2*dx),  0, 1/(2*dx)],
                                                  [0,       0,   0]])
            self.kernel2Dz = self.___make_tensor([[0,  -1/(2*dz),      0],
                                                  [0,   0,         0],
                                                  [0,   1/(2*dz),    0]])
        else:
            # Laplacian kernel for 2nd-order FD forward propagation (default).
            #dz_s = F.Conv2d() 1/dz 
            self.kernel2D = self.___make_tensor([[0, (dt / dz)**2, 0],
                                                [(dt / dx)**2, 
                                                 -2 * ((dt / dx)**2 + (dt / dz)**2), 
                                                 (dt / dx)**2],
                                                [0, (dt / dz)**2, 0]])
            self.kernel2Dx = self.___make_tensor([[0 ,      0,       0],
                                                  [-1/(2*dx),  0, 1/(2*dx)],
                                                  [0,       0,   0]])
            self.kernel2Dz = self.___make_tensor([[0,  -1/(2*dz),      0],
                                                  [0,   0,         0],
                                                  [0,   1/(2*dz),    0]])

        self.bx, self.bz, self.cx, self.cz, self.alpha_x, self.alpha_z, self.belta_x, self.belta_z = self.___make_pml(device)
        self.cx = F.conv2d(self.bx[None, None, ...], self.kernel2Dx, padding=1)[0, 0] 
        self.cz = F.conv2d(self.bz[None, None, ...], self.kernel2Dz, padding=1)[0, 0]
        self.delta = self.bx + self.bz

        # #### Make kernelX and kernelZ for regularizer ####
        self.kernelX = self.___make_tensor([[-1 / 2, 0, 1 / 2]])
        self.kernelZ = self.___make_tensor([[-1 / 2], [0], [1 / 2]])

    def ___make_tensor(self, a):
        # Prepare a filter in shape of [1, 1, Height, Width]
        a = np.asarray(a)
        a = a.reshape([1, 1] + list(a.shape))
        #print('a type is',a.dtype)
        return torch.as_tensor(a, dtype=self.dtype, device=self.device)
        
    def __sponge_bcMask(self):
        # Keep tmp as numpy array, because tensor does not support negative step like [::-1]
        tmp = np.exp(-0.0015**2 * np.arange(self.npad, 0, -1, dtype=np.float32)**2)  # small -- > large
        wmask = np.ones([self.nz_pad, self.nx_pad], dtype=np.float32)
        # add bottom_mask
        wmask[-self.npad:, self.npad:-self.npad] *= tmp[::-1][:, None]
        if self.freeSurface is False:
            # add top_mask
            wmask[:self.npad, self.npad:-self.npad] *= tmp[:, None]
        # add left_mask
        wmask[:, :self.npad] *= tmp[None, :]
        # add right_mask
        wmask[:, -self.npad:] *= tmp[::-1][None, :]
        return torch.as_tensor(wmask, dtype=self.dtype, device=self.device)
    
    def ___tensor_pad(self, input_tensor):
        """
        This function is to padding velocity tensor for implementing the absorbing boudary condition.
            input_tensor: is a 4D tensor, shape=[batch_size, 1, nz, nx]
            output_tensor: is also a 4D tensor, shape=[batch_size, 1, nz_pad, nx_pad]
        """  

        batch_size = input_tensor.shape[0]
        if self.freeSurface:
            vpadTop = input_tensor
        else:
            vtop = torch.ones((batch_size, 1, self.npad, self.nx), dtype=self.dtype, device=self.device) * input_tensor[:, :, :1, :]
            vpadTop = torch.cat((vtop, input_tensor), -2)  # padding on axis=2 (nz)
        
        vbottom = torch.ones((batch_size, 1, self.npad, self.nx), dtype=self.dtype, device=self.device) * input_tensor[:, :, -1:, :]
        vpadBottom = torch.cat([vpadTop, vbottom], -2)  # padding on axis=2 (nz)

        vleft = torch.ones((batch_size, 1, self.nz_pad, self.npad), dtype=self.dtype, device=self.device) * vpadBottom[:, :, :, :1]
        vpadLeft = torch.cat([vleft, vpadBottom], -1)  # padding on axis=3 (nx)

        vright = torch.ones((batch_size, 1, self.nz_pad, self.npad), dtype=self.dtype, device=self.device) * vpadBottom[:, :, :, -1:]
        output_tensor = torch.cat([vpadLeft, vright], -1)  # padding on axis=3 (nx)
        return output_tensor
    
    def ___make_pml(self, device='cpu'):
        """
        This function is to make PML layer.
        Input:
            u_prev: state at time step t-dt,   in shape of [self.num_vels, self.num_shots, self.nz_pad, self.nx_pad]
            u_:     state at time step t, also in shape of [self.num_vels, self.num_shots, self.nz_pad, self.nx_pad]

            source: a list of tensor: [fs, zs, xs]; 
                    where fs: [self.num_vels], 
                          zs: [self.num_vels, self.num_shots],
                          xs: [self.num_vels, self.num_shots]
        Output: 
            u_next: state at time step t+dt, in shape of [self.num_vels, self.num_shots, self.nz_pad, self.nx_pad]
        """ 
        bx = torch.zeros([self.nz_pad,self.nx_pad], dtype=self.dtype, device='cpu')
        bz = torch.zeros([self.nz_pad,self.nx_pad], dtype=self.dtype, device='cpu') 
        cx = torch.zeros([self.nz_pad,self.nx_pad], dtype=self.dtype, device='cpu')
        cz = torch.zeros([self.nz_pad,self.nx_pad], dtype=self.dtype, device='cpu')
        alpha_x = torch.zeros([self.nz_pad,self.nx_pad], dtype=self.dtype, device='cpu')
        alpha_z = torch.zeros([self.nz_pad,self.nx_pad], dtype=self.dtype, device='cpu') 
        belta_x = torch.zeros([self.nz_pad,self.nx_pad], dtype=self.dtype, device='cpu')
        belta_z = torch.zeros([self.nz_pad,self.nx_pad], dtype=self.dtype, device='cpu')

        max_vp = self.vmax
        # max_vp = 4500
        # max_vp = torch.max(self.velocity)
        a = 0
        b = 1
        p_x = torch.zeros([self.nz_pad,self.nx_pad], dtype=self.dtype, device='cpu')
        p_z = torch.zeros([self.nz_pad,self.nx_pad], dtype=self.dtype, device='cpu') 
        pmlx = torch.zeros([self.nz_pad,self.nx_pad], dtype=self.dtype, device='cpu')
        pmlz = torch.zeros([self.nz_pad,self.nx_pad], dtype=self.dtype, device='cpu')
        # print('self.nz_pad',self.nz_pad,self.nx_pad,self.pad)
        #Index = 2
        #wmask = np.ones([self.nx_pad, self.nx_pad], dtype=np.float32)
        for i in range(self.nz_pad):
            for j in range(self.nx_pad):
                if (j <= self.pad-1): 
                    p_x[i,j] = (self.pad -j)*self.dx
                if (j >= (self.nx_pad-self.pad)):
                    p_x[i,j] = (j + self.pad - self.nx_pad+1)*self.dx
                if (self.freeSurface ==0 and i <= self.pad-1):
                    p_z[i,j] = (self.pad -i)*self.dz
                if (i >= (self.nz_pad-self.pad)):
                    p_z[i,j] = (i + self.pad - self.nz_pad+1)*self.dz 

                pmlx[i,j] = -1.5*max_vp*math.log(self.log_para)/(self.pad*self.dx)**3 * (a*p_x[i,j] +b*p_x[i,j]**2)   
                pmlz[i,j] = -1.5*max_vp*math.log(self.log_para)/(self.pad*self.dz)**3 * (a*p_z[i,j] +b*p_z[i,j]**2)
                bx[i,j] = pmlx[i,j]#*0.01 #*self.dt
                bz[i,j] = pmlz[i,j]#*0.01 #*self.dt

                alpha_x[i,j] = (1-0.5*self.dt*bx[i,j])/(1+0.5*self.dt*bx[i,j])
                alpha_z[i,j] = (1-0.5*self.dt*bz[i,j])/(1+0.5*self.dt*bz[i,j])
                belta_x[i,j] = 1/(1+0.5*self.dt*bx[i,j])
                belta_z[i,j] = 1/(1+0.5*self.dt*bz[i,j])
        return bx.to(device), bz.to(device), \
               cx.to(device), cz.to(device), \
               alpha_x.to(device), alpha_z.to(device), \
               belta_x.to(device), belta_z.to(device)

    def ___step_rnncell(self, source, u_prev, u_,vx_pre,vz_pre):
        """
        This function is to implement the forward propagation for a single time-step.
        Input:
            u_prev: state at time step t-dt,   in shape of [self.num_vels, self.num_shots, self.nz_pad, self.nx_pad]
            u_:     state at time step t, also in shape of [self.num_vels, self.num_shots, self.nz_pad, self.nx_pad]

            source: a list of tensor: [fs, zs, xs]; 
                    where fs: [self.num_vels], 
                          zs: [self.num_vels, self.num_shots],
                          xs: [self.num_vels, self.num_shots]
        Output: 
            u_next: state at time step t+dt, in shape of [self.num_vels, self.num_shots, self.nz_pad, self.nx_pad]
        """        
        # For conv2d (squared filter): 
        #       out_size = [in_size + 2*pad_size - kernel_size - (kernel_size-1)*(dilation-1)]/stride + 1
        # To achieve "same", the pad_size we need is: (dilation=1, stride=1)
        #       padding=(self.k_size-1)/2

        ux_x = F.conv2d(u_, self.kernel2Dx.repeat(u_.shape[1], 1, 1, 1), padding=1, groups=u_.shape[1])  # 1st partial_x
        uz_z = F.conv2d(u_, self.kernel2Dz.repeat(u_.shape[1], 1, 1, 1), padding=1, groups=u_.shape[1])
        #print('torch.max(vx_pre),torch.max(vz_pre)',torch.max(vx_pre),torch.max(vz_pre))
        
        vx =  self.alpha_x[None, None, ...] * vx_pre + self.dt * ux_x * self.belta_x[None, None, ...] 
        vz =  self.alpha_z[None, None, ...] * vz_pre + self.dt * uz_z * self.belta_z[None, None, ...]  
        vx_x = F.conv2d(vx, self.kernel2Dx.repeat(u_.shape[1], 1, 1, 1), padding=1, groups=u_.shape[1])
        vz_z = F.conv2d(vz, self.kernel2Dz.repeat(u_.shape[1], 1, 1, 1), padding=1, groups=u_.shape[1]) 
        u_partial = F.conv2d(u_, self.kernel2D.repeat(u_.shape[1], 1, 1, 1), padding=(self.k_size - 1) // 2, groups=u_.shape[1])     
        
        u_next = 1/(self.delta * self.dt+1)[None, None, ...] \
                 * ((u_partial * self.velocity**2  + 2*u_ - u_prev) 
                    + self.dt * self.delta[None, None, ...] * u_
                    - (self.dt * self.velocity)**2 * (self.bx[None, None, ...] * vx_x 
                                                      + self.cx[None, None, ...] * vx
                                                      + self.bz[None, None, ...] * vz_z
                                                      + self.cz[None, None, ...] * vz))
                                                                                                                                                                           
        # Adding source
        for ivel in range(u_next.shape[0]):
            for ishot in range(u_next.shape[1]):
                if self.freeSurface:
                    u_next[ivel, ishot, source[1][ivel, ishot], source[2][ivel, ishot] + self.npad].data += source[0][ivel]
                else:
                    u_next[ivel, ishot, source[1][ivel, ishot] + self.npad, source[2][ivel, ishot] + self.npad].data += source[0][ivel]

        # Applying absorbing boundary mask
        # if self.npad != 0:
        #      wmask = self.__sponge_bcMask()[None, None, :, :]
        #      u_next = u_next * wmask
        return u_, u_next, vx, vz 

    def forward(self, vel_noPad, sources_info, prev_wavefield, curr_wavefield,vx,vz):
        """
        Forward propagating for a single time-step from (t-dt & t --> t+dt).
        Input:
            sources_info: a list contains [wavelet, zs, xs, zr, xr],
                          wavelet(tensor): shape [num_vels]
                          zs(tensor): shape [num_vels, num_shots]
                          xs(tensor): shape [num_vels, num_shots]
                          zr(tensor): shape [num_vels, num_shots, num_receivers]
                          xr(tensor): shape [num_vels, num_shots, num_receivers]
            Initial states (tensor):
                        1. prev_wavefield(tensor): wavefield at time step t-dt, 
                        2. curr_wavefield(tensor): wavefield at time step t
                            size: [num_vels, num_shots, self.nz_pad, , self.nx_pad]
            vel_noPad: A PyTorch tensor for velocity model with grid interval dz=dx, 
                        - shape = [num_vels, nz, nx]. 
                        - Initial velocity model for inversion process, i.e., requires_grad=True.
                        - True velocity model for forward modeling propagation.
        Output:
            Save prev_wavefield & curr_wavefield for next time step prop.
            yt_pred: [num_vels, num_shots, num_receivers]
                extracting seismogram at receiver locations using [zr, xr].
        """
        # Padding the velocity model (Should repadding velocity every time for absorbing)
        self.velocity = self.___tensor_pad(vel_noPad[:, None, :, :])  # After shape: [num_vels, 1, nz_pad, nx_pad]
        if isinstance(sources_info[3], int):
            num_receivers = self.nx
        else:
            num_receivers = sources_info[4].shape[2]
        row = torch.arange(prev_wavefield.shape[0])[:, None, None].repeat([1, prev_wavefield.shape[1], num_receivers])
        col = torch.arange(prev_wavefield.shape[1])[None, :, None].repeat([prev_wavefield.shape[0], 1, num_receivers])
        # print("Debug Check 1: prev_wavefield: {}, curr_wavefield: {}, vel_noPad: {}".format(prev_wavefield.shape, curr_wavefield.shape, vel_noPad.shape))
        
        prev_wavefield, curr_wavefield, vx, vz = self.___step_rnncell(sources_info[:3], prev_wavefield, curr_wavefield, vx, vz)
        if self.freeSurface:
            depth_index = sources_info[3]
        else:
            depth_index = sources_info[3] + self.npad
        if isinstance(sources_info[3], int):
            yt_pred = curr_wavefield[:, :, sources_info[3], self.npad:self.npad + num_receivers].reshape(list(sources_info[1].shape) + [-1])
        else:
            yt_pred = curr_wavefield[row.view(-1), col.view(-1), depth_index.view(-1), sources_info[4].view(-1) + self.npad].reshape(list(sources_info[1].shape) + [-1])
        
        # Add Regularization term, shape: [num_vels, 1]
        # 1. for sparse model
        # regularizer = vel_noPad.pow(2).sum(dim=-1)  
        # 2.1 for flatest model (1D derivative)
        # gradVel = vel_noPad[:, :, 1:] - vel_noPad[:, :, :-1]
        # regularizer = gradVel.pow(2).sum(dim=-1)  
        # 2.2 for flatest model (2D derivatives)
        # gradVelZ = F.conv1d(vel_noPad, self.kernelZ, padding=0)
        # regularizer = gradVelZ.pow(2).sum(dim=-1)
        regularizer = torch.tensor([[0]], dtype=self.dtype, device=self.device)
        return prev_wavefield, curr_wavefield, yt_pred, regularizer, vx, vz


""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
"""                                     2D RNN forward modeling (full/truncated time step)                                 """
""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """


class rnn2D(torch.nn.Module):
    """
    Using Propagator1D to perform the forward modeling with given wavelets through multiple time-steps
    """
    def __init__(self, nz, nx, zs, xs, zr, xr, dz, dt, 
                 npad=0, order=2, vmax=6000, 
                 log_para=1e-6,
                 freeSurface=True, 
                 dtype=torch.float32, device='cpu'):
        super(rnn2D, self).__init__()
        """
        zs(tensor): in shape of [num_vels, num_shots]
        xs(tensor): in shape of [num_vels, num_shots]
        zr(tensor): in shape of [num_vels, num_shots, num_receivers]
        xr(tensor): in shape of [num_vels, num_shots, num_receivers]
        """
        # Stability condition for forward modeling (Lines et al., 1999, Geophysics)
        # vmax*dt/dz < XXX
        # for 1D, 2nd-order: 1; 4th-order: np.sqrt(3)/2
        # for 2D, 2nd-order: 1/np.sqrt(2); 4th-order: np.sqrt(3/8)
        # for 3D, 2nd-order: 1/np.sqrt(3); 4th-order: 1/2
        if order == 2:
            # print("Stability Condition: {} < {}".format(vpred.max().cpu() * dt / dz, 1))
            assert vmax * dt / dz < 1 / np.sqrt(2), "Current parameters setting do NOT meet the stability condition."
        elif order == 4:
            # print("Stability Condition: {} < {}".format(vpred.max().cpu() * dt / dz, np.sqrt(3) / 2))
            assert vmax * dt / dz < np.sqrt(3 / 8), "Current parameters setting do NOT meet the stability condition."
        self.dtype = dtype
        self.device = device
        self.zs, self.xs = zs, xs
        self.zr, self.xr = zr, xr
        self.nx_pad = nx + 2 * npad
        if freeSurface:
            self.nz_pad = nz + npad
        else:
            self.nz_pad = nz + 2 * npad
        
        # define the finite-difference operator for each time step
        self.fd = Propagator2D(nz, nx, dz, dt, npad, order, freeSurface, log_para, vmax, dtype, device).to(device)

    def forward(self, vmodel, segment_wavelet, prev_state=None, curr_state=None, option=0, with_checkpoint=True):
        """
        Input:
            vmodel(tensor): require_grad=True for training RNN, shape: [num_vels, nz, nx]
            prev_state(tensor): 4D tensor, shape: [num_vels, num_shots, nz_pad, nx_pad]
            curr_state(tensor): 4D tensor, shape: [num_vels, num_shots, nz_pad, nx_pad]
            segment_wavelet(tensor): [num_vels, len_tSeg] or [len_tSeg]
        Ouput:
            prev_state & curr_state for next time-segment, which is depending on option you choose.
            segment_ytPred(tensor): [num_vels, num_shots, nt(len_tSeg), num_receivers]
        option:
            option only affects the returned prev_state & curr_state, 
            which is related to the segment option for data and wavelet in gen_Segment2d(xxx, option=0);
            where,
                option=0, the returned prev_state & curr_state are wavefileds at two last time steps of the current time-segment;
                option=1, the returned prev_state & curr_state are wavefileds at the midterm of the current time-segment;
                option=2, the returned prev_state & curr_state are zero-initialized wavefields (time_step=0).
        """
        if prev_state is None:
            prev_state = torch.zeros([vmodel.shape[0], self.zs.shape[1], self.nz_pad, self.nx_pad], 
                                     dtype=self.dtype, device=self.device)
            curr_state = torch.zeros([vmodel.shape[0], self.zs.shape[1], self.nz_pad, self.nx_pad], 
                                     dtype=self.dtype, device=self.device)
        # print("Debug Check 2: prev_state: {}, curr_state: {}, vmodel: {}".format(prev_state.shape, curr_state.shape, vmodel.shape))

        # if segment_wavelet.ndim == 1:
        if len(segment_wavelet.shape) == 1:
            segment_wavelet = segment_wavelet.repeat(prev_state.shape[0], 1)

        vx = torch.zeros_like(prev_state)
        vz = torch.zeros_like(prev_state)
        segment_ytPred = []
        avg_regularizer = []
        for it in range(segment_wavelet.shape[1]):
            segment_sources = [segment_wavelet[:, it], 
                               self.zs[:vmodel.shape[0], :], 
                               self.xs[:vmodel.shape[0], :], 
                               self.zr if isinstance(self.zr, int) else self.zr[:vmodel.shape[0], :, :], 
                               self.xr if isinstance(self.zr, int) else self.xr[:vmodel.shape[0], :, :]]
            # print(segment_sources[0].shape, segment_sources[1].shape, segment_sources[2].shape, segment_sources[3].shape, segment_sources[4].shape)
            if with_checkpoint:
                from torch.utils.checkpoint import checkpoint
                prev_state, curr_state, seg_ytPred, regularizer, vx, vz = checkpoint(self.fd, vmodel, segment_sources, prev_state, curr_state, vx, vz)
            else:
                prev_state, curr_state, seg_ytPred, regularizer, vx, vz = self.fd(vmodel, segment_sources, prev_state, curr_state, vx, vz)
            segment_ytPred.append(seg_ytPred)
            avg_regularizer.append(regularizer)
            # for option 1, we want save the middle states for next time segement
            if option == 1 and it == (len(segment_sources[0]) - 1) // 2:
                prev_save = prev_state.detach().clone()  
                curr_save = curr_state.detach().clone()  

        # for option 0, we save the last two wavefields for next time-segment
        if option == 0:
            prev_save = prev_state 
            curr_save = curr_state 
        # for next time-segement, option 2 makes sure it always start from time_step 0 with zero initials
        elif option == 2:
            prev_save = prev_state.new_zeros(prev_state.shape, dtype=self.dtype, device=prev_state.device)
            curr_save = curr_state.new_zeros(curr_state.shape, dtype=self.dtype, device=curr_state.device)
        
        segment_ytPred = torch.stack(segment_ytPred, dim=-2)  # shape: [num_vels, num_shots, nt(len_tSeg), num_receivers]
        avg_regularizer = torch.stack(avg_regularizer, dim=-1).mean(dim=-1)  # shape:[num_vels]
        return prev_save, curr_save, segment_ytPred, avg_regularizer

