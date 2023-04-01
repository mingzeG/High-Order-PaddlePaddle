import paddle

class Identity(paddle.nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x