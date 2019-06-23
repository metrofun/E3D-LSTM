import copy
import torch
import torch.nn.functional as F
import torch.nn as nn

class E3DLSTM(nn.Module):
  def __init__(self, input_shape, hidden_size, num_layers, kernel_size, tau):
    super().__init__()


    self._tau = tau
    self._cells = []

    input_shape = list(input_shape)
    for i in range(num_layers):
      cell = E3DLSTMCell(input_shape, hidden_size, kernel_size)
      # NOTE hidden state becomes input to the next cell
      input_shape[0] = hidden_size
      self._cells.append(cell)
      # Hook to register submodule
      setattr(self, 'cell{}'.format(i), cell)

  def forward(self, input):
    # NOTE (seq_len, batch, input_shape)
    batch_size = input.size(1)
    c_history_states = []
    h_states = []
    outputs = []

    for step, x in enumerate(input):
      for cell_idx, cell in enumerate(self._cells):
        if step == 0:
          c_history, m, h = self._cells[cell_idx].init_hidden(batch_size, self._tau)
          c_history_states.append(c_history)
          h_states.append(h)

        # NOTE c_history and h are coming from the previous time stamp, but we iterate over cells
        # should we take o as x? if so we need to change input dimensions
        c_history, m, h = cell(x, c_history_states[cell_idx], m, h_states[cell_idx])
        c_history_states[cell_idx] = c_history
        h_states[cell_idx] = h
        # NOTE hidden state of previous LSTM is passed as input to the next one
        x = h

      outputs.append(h)

    # Concat along the channels
    return torch.cat(outputs, dim=1)

class E3DLSTMCell(nn.Module):
  def __init__(self, input_shape, hidden_size, kernel_size):
    super().__init__()

    in_channels = input_shape[0]
    self._input_shape = input_shape
    self._hidden_size = hidden_size

    # Make same output
    padding = ((torch.tensor(kernel_size) - 1) / 2).tolist()

    # memory gates: input, cell(input modulation), forget
    self.weight_xi = nn.Conv3d(in_channels, hidden_size, kernel_size, padding=padding)
    self.weight_hi = nn.Conv3d(hidden_size, hidden_size, kernel_size, padding=padding, bias=False)

    self.weight_xg = copy.deepcopy(self.weight_xi)
    self.weight_hg = copy.deepcopy(self.weight_hi)

    self.weight_xr = nn.Conv3d(in_channels, hidden_size, kernel_size, padding=padding)
    self.weight_hr = nn.Conv3d(hidden_size, hidden_size, kernel_size, padding=padding, bias=True)

    memory_shape = list(input_shape)
    memory_shape[0] = hidden_size
    self.layer_norm = nn.LayerNorm(memory_shape)

    # for spatiotemporal memory
    self.weight_xi_prime = nn.Conv3d(in_channels, hidden_size, kernel_size, padding=padding)
    self.weight_mi_prime = nn.Conv3d(hidden_size, hidden_size, kernel_size, padding=padding, bias=False)

    self.weight_xg_prime = copy.deepcopy(self.weight_xi_prime)
    self.weight_mg_prime = copy.deepcopy(self.weight_mi_prime)

    self.weight_xf_prime = copy.deepcopy(self.weight_xi_prime)
    self.weight_mf_prime = copy.deepcopy(self.weight_mi_prime)

    self.weight_xo = nn.Conv3d(in_channels, hidden_size, kernel_size, padding=padding)
    self.weight_ho = nn.Conv3d(hidden_size, hidden_size, kernel_size, padding=padding, bias=False)
    self.weight_co = copy.deepcopy(self.weight_ho)
    self.weight_mo = copy.deepcopy(self.weight_ho)

    self.weight_111 = nn.Conv3d(hidden_size + hidden_size, hidden_size, 1)

  def forward(self, x, c_history, m, h):
    # R is CxT×H×W (temporal depth, spatial size, and the number of feature map channels)
    r = torch.sigmoid(self.weight_xr(x) + self.weight_hr(h))
    i = torch.sigmoid(self.weight_xi(x) + self.weight_hi(h))
    g = torch.tanh(self.weight_xg(x) + self.weight_hg(h))

    batch_size = r.size(0)
    channels = r.size(1)
    r_flatten = r.view(batch_size, -1, channels)
    c_history_flatten = c_history.view(batch_size, -1, channels)

    # Attention mechanism
    recall = torch.bmm(
      #THWxC x taoTHWxC' = THW x taoTHW
      F.softmax(torch.bmm(r_flatten, torch.transpose(c_history_flatten, 1, 2)), dim=1),
      #taoTHWxC
      c_history_flatten
    ).view(*i.shape)

    c = i * g + self.layer_norm(c_history[-1] + recall)

    i_prime = torch.sigmoid(self.weight_xi_prime(x) + self.weight_mi_prime(m))
    g_prime = torch.tanh(self.weight_xg_prime(x) + self.weight_mg_prime(m))
    f_prime = torch.sigmoid(self.weight_xf_prime(x) + self.weight_mf_prime(m))

    m = i_prime * g_prime + f_prime * m
    o = torch.sigmoid(self.weight_xo(x) + self.weight_ho(h) + self.weight_co(c) + self.weight_mo(m))
    h = o * torch.tanh(self.weight_111(torch.cat([c, m], dim=1)))

    # TODO is it correct FIFO?
    c_history = torch.cat([c_history[1:], c[None, :]], dim=0)

    return (c_history, m, h)

  def init_hidden(self, batch_size, tau):
    memory_shape = list(self._input_shape)
    memory_shape[0] = self._hidden_size
    c_history = torch.zeros(tau, batch_size, *memory_shape)
    m = torch.zeros(batch_size, *memory_shape)
    h = torch.zeros(batch_size, *memory_shape)

    return (c_history, m, h)

if __name__ == '__main__':
    seq = 6
    batch = 4
    c, d, w, h = 2, 3, 8, 8
    # gradient check
    # TODO only works for odd kernel sizes
    net = E3DLSTM(input_shape=(c, d, w, h), hidden_size=8, num_layers=4, kernel_size=(3, 5, 5), tau = 4)
    loss_fn = torch.nn.MSELoss()

    # seq x batch x timeframes x channels x width x height
    input = torch.randn(seq, batch, c, d, w, h)

    output = net(input)
    output = output[0][0][0].double()
    target = torch.randn(8, 8).double()

    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print(res)
