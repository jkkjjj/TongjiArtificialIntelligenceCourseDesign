import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)
    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        score = self.run(x)
        return 1 if nn.as_scalar(score) >= 0 else -1
    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            no_mistakes = True
            for x, y in dataset.iterate_once(1):
                prediction = self.get_prediction(x)
                actual = nn.as_scalar(y)
                if prediction != actual:
                    self.w.update(x, actual)
                    no_mistakes = False
            if no_mistakes:
                break


class RegressionModel(object):
    def __init__(self):
        """
        Initialize a new RegressionModel instance.
        """
        self.hidden_size = 256  # 调整隐藏层大小
        self.learning_rate = 0.01  # 降低学习率
        self.w1 = nn.Parameter(1, self.hidden_size)  # 输入层到隐藏层的权重
        self.b1 = nn.Parameter(1, self.hidden_size)  # 隐藏层的偏置
        self.w2 = nn.Parameter(self.hidden_size, 1)  # 隐藏层到输出层的权重
        self.b2 = nn.Parameter(1, 1)  # 输出层的偏置

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        hidden = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        output = nn.AddBias(nn.Linear(hidden, self.w2), self.b2)
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
        Returns: a loss node
        """
        prediction = self.run(x)
        return nn.SquareLoss(prediction, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        batch_size = 200
        while True:
            for x, y in dataset.iterate_forever(batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
                self.w1.update(gradients[0], -self.learning_rate)
                self.b1.update(gradients[1], -self.learning_rate)
                self.w2.update(gradients[2], -self.learning_rate)
                self.b2.update(gradients[3], -self.learning_rate)

                # 检查损失是否低于阈值
                if nn.as_scalar(loss) < 0.02:
                    return


class DigitClassificationModel(object):
    def __init__(self):
        """
        Initialize a new DigitClassificationModel instance.
        """
        self.hidden_size = 200
        self.learning_rate = 0.5
        self.w1 = nn.Parameter(784, self.hidden_size)  # 输入层到隐藏层的权重
        self.b1 = nn.Parameter(1, self.hidden_size)  # 隐藏层的偏置
        self.w2 = nn.Parameter(self.hidden_size, 10)  # 隐藏层到输出层的权重
        self.b2 = nn.Parameter(1, 10)  # 输出层的偏置

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        hidden = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        output = nn.AddBias(nn.Linear(hidden, self.w2), self.b2)
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        prediction = self.run(x)
        return nn.SoftmaxLoss(prediction, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        batch_size = 100
        while True:
            for x, y in dataset.iterate_forever(batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
                self.w1.update(gradients[0], -self.learning_rate)
                self.b1.update(gradients[1], -self.learning_rate)
                self.w2.update(gradients[2], -self.learning_rate)
                self.b2.update(gradients[3], -self.learning_rate)

                # 检查验证准确率是否达到阈值
                validation_accuracy = dataset.get_validation_accuracy()
                if validation_accuracy >= 0.975:
                    return


import numpy as np
import nn  # 假设 nn 是一个提供神经网络相关操作的模块


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.
    """

    def __init__(self):
        # 我们的数据集包含五种不同语言的单词，五种语言的组合字母表总共包含47个独特的字符。
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # 初始化模型参数
        self.hidden_size = 128  # 隐藏层大小
        self.Wx = nn.Parameter(self.num_chars, self.hidden_size)
        self.Wh = nn.Parameter(self.hidden_size, self.hidden_size)
        self.b = nn.Parameter(1, self.hidden_size)
        self.W_output = nn.Parameter(self.hidden_size, len(self.languages))
        self.b_output = nn.Parameter(1, len(self.languages))

    def run(self, xs):
        """
        Runs the model for a batch of examples.
        """
        batch_size = xs[0].shape[0]
        h = nn.Linear(xs[0], self.Wx) + nn.AddBias(self.b)

        for x in xs[1:]:
            h = nn.Add(nn.Linear(x, self.Wx), nn.Linear(h, self.Wh))
            h = nn.AddBias(h, self.b)
            h = nn.ReLU(h)

        scores = nn.Linear(h, self.W_output)
        scores = nn.AddBias(scores, self.b_output)
        return scores

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.
        """
        scores = self.run(xs)
        return nn.SoftmaxLoss(scores, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        optimizer = nn.Optimizer(self.parameters(), lr=0.01)

        for epoch in range(20):  # 假设训练20个epoch
            total_loss = 0.0
            num_batches = 0
            for xs, y in dataset.iterate_batches(batch_size=32):
                loss = self.get_loss(xs, y)
                optimizer.update(loss)
                total_loss += loss.item()
                num_batches += 1

            print(f"Epoch {epoch + 1}, Loss: {total_loss / num_batches}")

    def parameters(self):
        """
        Returns a list of all parameters in the model.
        """
        return [self.Wx, self.Wh, self.b, self.W_output, self.b_output]
