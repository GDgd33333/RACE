import torch.nn as nn
import torch.nn.functional as F

'''
1.模块分离：
    QMIXRNNAgent_SR 和 QMIXRNNAgent_W 将状态表示和Q值计算分离，便于实现参数共享
    这种设计在进化算法中特别有用，可以单独突变Q值计算部分

2.隐藏状态处理：
    所有RNN类都实现了init_hidden()方法，统一初始化接口
    隐藏状态使用与模型参数相同的设备和数据类型

3.进化算法适配：
    QMIXRNNAgent_W可以独立于状态表示网络进行突变
    前馈网络版本(FFAgent)为简单环境提供更高效的替代方案
'''

class QMIXRNNAgent(nn.Module):
    """
    QMIX RNN Agent 类，实现了基于GRU的循环神经网络智能体。
    用于在QMIX算法中处理部分可观察的环境。
    """

    def __init__(self, input_shape, args):
        """
        初始化QMIXRNNAgent

        参数:
            input_shape (int): 输入观测的维度
            args (Namespace): 包含模型超参数的命名空间，包括:
                - rnn_hidden_dim (int): RNN隐藏层维度
                - n_actions (int): 动作空间大小
        """
        super(QMIXRNNAgent, self).__init__()
        self.args = args

        # 第一层全连接网络，将输入映射到隐藏空间
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)

        # GRU循环神经网络单元，用于处理时序信息
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        # 第二层全连接网络，输出每个动作的Q值
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)


    def init_hidden(self):
        # make hidden states on same device as model
        """
        初始化隐藏状态

        返回:
            torch.Tensor: 初始化为零的隐藏状态，形状为(1, rnn_hidden_dim)
        """
        # 创建一个与fc1权重相同设备、相同数据类型的零张量
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        """
        前向传播
        参数:
            inputs (torch.Tensor): 当前观测，形状为(batch_size, input_shape)
            hidden_state (torch.Tensor): 上一个时间步的隐藏状态
        返回:
            tuple: 包含:
                - q (torch.Tensor): 动作Q值，形状为(batch_size, n_actions)
                - h (torch.Tensor): 新的隐藏状态
        """
        x = F.relu(self.fc1(inputs))  # 通过第一层并应用ReLU激活
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)  # 调整隐藏状态形状
        h = self.rnn(x, h_in)  # 通过GRU单元更新隐藏状态
        q = self.fc2(h)  # 计算Q值
        return q, h


class QMIXRNNAgent_SR(nn.Module):
    """
    状态表示(State Representation)版本的QMIX RNN Agent。
    只输出状态表示，不计算Q值，用于共享状态表示。
    """

    def __init__(self, input_shape, args):
        """
        初始化QMIXRNNAgent_SR

        参数:
            input_shape (int): 输入观测的维度
            args (Namespace): 包含模型超参数的命名空间
        """
        super(QMIXRNNAgent_SR, self).__init__()
        self.args = args

        # 第一层全连接网络
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)

        # GRU循环神经网络单元
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

    def init_hidden(self):
        """
        初始化隐藏状态
        返回:
            torch.Tensor: 初始化为零的隐藏状态
        """
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        """
        前向传播，只返回状态表示
        参数:
            inputs (torch.Tensor): 当前观测
            hidden_state (torch.Tensor): 上一个时间步的隐藏状态

        返回:
            torch.Tensor: 状态表示(隐藏状态)
        """
        x = F.relu(self.fc1(inputs), inplace=True)  # 使用原地操作节省内存
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)  # 更新隐藏状态
        return hh


class QMIXRNNAgent_W(nn.Module):
    """
    Q值计算版本的QMIX RNN Agent。
    从共享的状态表示计算Q值，通常与QMIXRNNAgent_SR配合使用。
    """

    def __init__(self, input_shape, args):
        """
        初始化QMIXRNNAgent_W
        参数:
            input_shape (int): 保留参数，实际未使用
            args (Namespace): 包含模型超参数的命名空间
        """
        super(QMIXRNNAgent_W, self).__init__()
        self.args = args

        # 全连接层，从状态表示映射到Q值
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        # 可选：使用层归一化
        if getattr(self.args, "use_layer_norm", False):
            self.layer_norm = nn.LayerNorm(args.rnn_hidden_dim)

    def forward(self, inputs, shared_state_embedding):
        """
        前向传播，从共享状态表示计算Q值
        参数:
            inputs (torch.Tensor): 保留参数，实际未使用
            shared_state_embedding (torch.Tensor): 共享的状态表示
        返回:
            torch.Tensor: 动作Q值
        """
        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(shared_state_embedding))
        else:
            q = self.fc2(shared_state_embedding)
        return q


class FFAgent(nn.Module):
    """
    前馈神经网络Agent，不使用RNN，适用于完全可观察环境。
    """

    def __init__(self, input_shape, args):
        """
        初始化FFAgent

        参数:
            input_shape (int): 输入观测的维度
            args (Namespace): 包含模型超参数的命名空间
        """
        super(FFAgent, self).__init__()
        self.args = args

        # 三层全连接网络
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        """
        为保持接口一致性，返回零张量

        返回:
            torch.Tensor: 零张量
        """
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        """
        参数:
            inputs (torch.Tensor): 当前观测
            hidden_state (torch.Tensor): 保留参数，实际未使用
        返回:
            tuple: 包含:
                - q (torch.Tensor): 动作Q值
                - h (torch.Tensor): 最后一层隐藏表示(为保持接口一致性)
        """
        x = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(x))  # 第二层隐藏表示
        q = self.fc3(h)  # 输出Q值
        return q, h