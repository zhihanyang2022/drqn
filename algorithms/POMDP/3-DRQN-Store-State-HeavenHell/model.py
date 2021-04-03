import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# from config import burn_in_length

class DRQN(nn.Module):

    def __init__(self, num_inputs, num_outputs, sequence_length):

        super(DRQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=16, batch_first=True)
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x, hidden=None, inference=True):
        # x [batch_size, sequence_length, num_inputs]
        out, hidden = self.lstm(x, hidden)
        if not inference:
            out, _ = pad_packed_sequence(sequence=out, batch_first=True, total_length=self.sequence_length)

        out = F.relu(self.fc1(out))
        qvalue = self.fc2(out)

        return qvalue, hidden


    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, batch_size, sequence_length, gamma):

        # def slice_burn_in(item):
        #     return item[:, burn_in_length:, :]

        # batch.state is a list of tensors of shape (seq_length, input_dim)
        # so seq.size()[0] = the length of the sequence
        lengths = np.array([seq.size()[0] for seq in batch.state])
        max_length = np.max(lengths)

        # ===== compute loss mask =====

        # for example, if sequence_length == 3, then lower_triangular_matrix =
        # 1 0 0
        # 1 1 0
        # 1 1 1
        # suppose lengths == np.array([2, 3, 1]), then lengths - 1 == np.array([1, 2, 0]) and
        # the loss_mask computed from lower_triangular_matrix[lengths-1] is
        # 1 1 0
        # 1 1 1
        # 1 0 0
        # which corresponds to lengths correctly

        lower_triangular_matrix = np.tril(np.ones((sequence_length, sequence_length)))
        loss_mask = lower_triangular_matrix[lengths-1]  # first convert from 1-based to 0-based indexing
        loss_mask = torch.tensor(loss_mask)  # has shape (bs, seq_len)

        states = pack_padded_sequence(
            pad_sequence(batch.state, batch_first=True),
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False
        )  # ready to be inputted into DRQN

        next_states = pack_padded_sequence(
            pad_sequence(batch.next_state, batch_first=True),
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False
        )

        # max_length == sequence_length most of the times, but not always
        actions = pad_sequence(batch.action, batch_first=True).view(batch_size, max_length, -1).long()  # has shape (bs, seq_len, 1)
        rewards = pad_sequence(batch.reward, batch_first=True).view(batch_size, max_length, -1)  # has shape (bs, seq_len, 1)
        masks   = pad_sequence(batch.mask,   batch_first=True).view(batch_size, max_length, -1)  # has shape (bs, seq_len, 1)

        h0 = torch.stack([rnn_state[0,0,:] for rnn_state in batch.rnn_state]).unsqueeze(0).detach()  # has shape (1, bs, hidden_size)
        c0 = torch.stack([rnn_state[0,1,:] for rnn_state in batch.rnn_state]).unsqueeze(0).detach()  # has shape (1, bs, hidden_size)

        h1 = torch.stack([rnn_state[1,0,:] for rnn_state in batch.rnn_state]).unsqueeze(0).detach()  # has shape (1, bs, hidden_size)
        c1 = torch.stack([rnn_state[1,1,:] for rnn_state in batch.rnn_state]).unsqueeze(0).detach()  # has shape (1, bs, hidden_size)

        # states = torch.stack(batch.state).view(batch_size, sequence_length, online_net.num_inputs)
        # next_states = torch.stack(batch.next_state).view(batch_size, sequence_length, online_net.num_inputs)
        # actions = torch.stack(batch.action).view(batch_size, sequence_length, -1).long()
        # rewards = torch.stack(batch.reward).view(batch_size, sequence_length, -1)
        # masks = torch.stack(batch.mask).view(batch_size, sequence_length, -1)
        # rnn_state = torch.stack(batch.rnn_state).view(batch_size, sequence_length, 2, -1)

        # [h0, c0] = rnn_state[:, 0, :, :].transpose(0, 1)
        # h0 = h0.unsqueeze(0).detach()
        # c0 = c0.unsqueeze(0).detach()

        # [h1, c1] = rnn_state[:, 1, :, :].transpose(0, 1)
        # h1 = h1.unsqueeze(0).detach()
        # c1 = c1.unsqueeze(0).detach()

        pred, _ = online_net(states, (h0, c0), inference=False)
        next_pred, _ = target_net(next_states, (h1, c1), inference=False)

        # if burn_in_length > 0:
        #     pred = slice_burn_in(pred)
        #     next_pred = slice_burn_in(next_pred)
        #     actions = slice_burn_in(actions)
        #     rewards = slice_burn_in(rewards)
        #     masks = slice_burn_in(masks)
        
        pred = pred.gather(2, actions).squeeze()  # has shape (bs, seq_len)
        
        target = rewards + masks * gamma * next_pred.max(2, keepdim=True)[0]
        target = target.squeeze()  # has shape (bs, seq_len)

        loss = torch.mean(((pred - target.detach()) ** 2) * loss_mask.double())
        # loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online_net.parameters(), 1.0)
        optimizer.step()

        return loss

    def get_action(self, state, hidden):
        state = state.unsqueeze(0).unsqueeze(0)

        qvalue, hidden = self.forward(state, hidden)
            
        _, action = torch.max(qvalue, 2)
        return action.numpy()[0][0], hidden
