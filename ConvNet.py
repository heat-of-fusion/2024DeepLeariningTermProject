import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE):
        super(InceptionModule, self).__init__()

        self.input_size = INPUT_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.output_size= OUTPUT_SIZE

        self.leaky_relu = nn.LeakyReLU(0.05)

        self.path_1 = nn.Sequential(
            nn.Conv2d(INPUT_SIZE, HIDDEN_SIZE, kernel_size = (1, 1), stride = (1, 1), padding = 'same'),
            nn.BatchNorm2d(HIDDEN_SIZE),
            nn.LeakyReLU(),
            nn.Conv2d(HIDDEN_SIZE, OUTPUT_SIZE // 4, kernel_size = (3, 3), stride = (1, 1), padding = 'same'),
            nn.BatchNorm2d(OUTPUT_SIZE // 4),
            nn.LeakyReLU(),
        )

        self.path_2 = nn.Sequential(
            nn.Conv2d(INPUT_SIZE, HIDDEN_SIZE, kernel_size = (1, 1), stride = (1, 1), padding = 'same'),
            nn.BatchNorm2d(HIDDEN_SIZE),
            nn.LeakyReLU(),
            nn.Conv2d(HIDDEN_SIZE, OUTPUT_SIZE // 4, kernel_size = (5, 5), stride = (1, 1), padding = 'same'),
            nn.BatchNorm2d(OUTPUT_SIZE // 4),
            nn.LeakyReLU(),
        )

        self.path_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size = (3, 3), stride = (1, 1), padding = 1),
            nn.Conv2d(INPUT_SIZE, OUTPUT_SIZE // 4, kernel_size = (1, 1), stride = (1, 1), padding = 'same'),
            nn.BatchNorm2d(OUTPUT_SIZE // 4),
            nn.LeakyReLU(),
        )

        self.path_4 = nn.Sequential(
            nn.Conv2d(INPUT_SIZE, OUTPUT_SIZE // 4, kernel_size = (1, 1), stride = (1, 1), padding = 'same'),
            nn.BatchNorm2d(OUTPUT_SIZE // 4),
            nn.LeakyReLU(),
        )

        return

    def forward(self, x):
        path_1_out = self.path_1(x)
        path_2_out = self.path_2(x)
        path_3_out = self.path_3(x)
        path_4_out = self.path_4(x)

        out = torch.cat([path_1_out, path_2_out, path_3_out, path_4_out], dim = 1)

        return out

class InceptionResBlock(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE):
        super(InceptionResBlock, self).__init__()

        self.leaky_relu = nn.LeakyReLU(0.05)

        self.conv_1 = InceptionModule(INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE)
        self.conv_2 = InceptionModule(HIDDEN_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

        self.bn_1 = nn.BatchNorm2d(HIDDEN_SIZE)
        self.bn_2 = nn.BatchNorm2d(OUTPUT_SIZE)

        return

    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.leaky_relu(out)

        out = self.conv_2(out)
        out = self.bn_2(out)

        out = x + out

        return out

class CustomInceptionResNetSingle(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, N_CLASSES):
        super(CustomInceptionResNetSingle, self).__init__()

        self.leaky_relu = nn.LeakyReLU(0.05)
        self.softmax = nn.Softmax(dim = 1)

        self.conv_net = nn.Sequential(
            nn.Conv2d(INPUT_SIZE, HIDDEN_SIZE, kernel_size = (3, 3), stride = (1, 1), padding = 'same'),
            nn.BatchNorm2d(HIDDEN_SIZE),
            nn.LeakyReLU(),
        )

        self.setup_flag = False
        self.hidden_size = HIDDEN_SIZE
        self.n_classes = N_CLASSES

        return

    def setup(self, batch_shape = [32, 1, 28, 28]):
        batch_size = batch_shape[0]

        dummy_batch = torch.zeros(batch_shape)
        self.fc_dim = self.forward(dummy_batch, bypass = True).view(batch_size, -1).shape[-1]

        self.fc_net = nn.Sequential(
            nn.Linear(self.fc_dim, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.LeakyReLU(0.05),
            nn.Linear(self.hidden_size, self.n_classes),
            nn.Softmax(dim=1)
        )

        self.setup_flag = True

        return f'Model Setup Completed!'

    def check_setup(self):
        assert self.setup_flag, f'Model Setup Incompleted! Use model.setup() to complete setup'

    def forward(self, x, bypass = False):
        if not bypass:
            self.check_setup()
        print(1)
        out = self.conv_net(x)
        print(2)

        if bypass:
            return out
        print(3)

        out = out.view(-1, self.fc_dim)
        print(4)
        out = self.fc_net(out)
        print(5)

        return out

class CustomInceptionResNet(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, N_CLASSES):
        super(CustomInceptionResNet, self).__init__()

        self.leaky_relu = nn.LeakyReLU(0.05)
        self.softmax = nn.Softmax(dim = 1)

        self.conv_net = nn.Sequential(
            InceptionResBlock(INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(),
            InceptionResBlock(HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(),
            InceptionResBlock(HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(),
            InceptionResBlock(HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(),
            InceptionResBlock(HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(),
            InceptionResBlock(HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),
            InceptionResBlock(HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(),
            InceptionResBlock(HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(),
            InceptionResBlock(HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(),
            InceptionResBlock(HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(),
            InceptionResBlock(HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(),
            InceptionResBlock(HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),
        )

        self.setup_flag = False
        self.hidden_size = HIDDEN_SIZE
        self.n_classes = N_CLASSES

        return

    def setup(self, batch_shape = [32, 1, 28, 28]):
        batch_size = batch_shape[0]

        dummy_batch = torch.zeros(batch_shape)
        self.fc_dim = self.forward(dummy_batch, bypass = True).view(batch_size, -1).shape[-1]

        self.fc_net = nn.Sequential(
            nn.Linear(self.fc_dim, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.LeakyReLU(0.05),
        )

        self.j_net = nn.Sequential(
            nn.Linear(self.hidden_size, 14),
            nn.Softmax(dim=1)
        )

        self.m_net = nn.Sequential(
            nn.Linear(self.hidden_size, 10),
            nn.Softmax(dim = 1)
        )

        self.setup_flag = True

        return f'Model Setup Completed!'

    def check_setup(self):
        assert self.setup_flag, f'Model Setup Incompleted! Use model.setup() to complete setup'

    def forward(self, x, bypass = False):
        if not bypass:
            self.check_setup()

        out = self.conv_net(x)

        if bypass:
            return out

        out = out.view(-1, self.fc_dim)
        out = self.fc_net(out)
        out_j = self.j_net(out)
        out_m = self.m_net(out)

        return out_j, out_m

class CustomMultiConvNet(nn.Module):
    def __init__(self, INPUT_SIZE, N_CLASSES = (14, 10), HIDDEN_SIZE = 512):
        super(CustomMultiConvNet, self).__init__()

        self.leaky_relu = nn.LeakyReLU(0.05)
        self.softmax = nn.Softmax(dim = 1)

        self.conv_1 = nn.Conv2d(INPUT_SIZE, HIDDEN_SIZE // 2, kernel_size = (3, 3), stride = (1, 1), padding = 'same')
        self.conv_2 = nn.Conv2d(HIDDEN_SIZE // 2, HIDDEN_SIZE, kernel_size = (3, 3), stride = (1, 1), padding = 'same')
        self.conv_3 = nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size = (3, 3), stride = (1, 1), padding = 'same')
        self.conv_4 = nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size = (3, 3), stride = (1, 1), padding = 'same')

        self.bn_1 = nn.BatchNorm2d(HIDDEN_SIZE // 2)
        self.bn_2 = nn.BatchNorm2d(HIDDEN_SIZE)
        self.bn_3 = nn.BatchNorm2d(HIDDEN_SIZE)
        self.bn_4 = nn.BatchNorm2d(HIDDEN_SIZE)

        self.pool_1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.pool_2 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))

        self.setup_flag = False
        self.j_classes, self.m_classes = N_CLASSES
        self.hidden_size = HIDDEN_SIZE

        return

    def setup(self, batch_shape = [32, 1, 28, 28]):
        batch_size = batch_shape[0]

        dummy_batch = torch.zeros(batch_shape)
        self.fc_dim = self.forward(dummy_batch, bypass = True).view(batch_size, -1).shape[-1]

        self.fc_1 = nn.Linear(self.fc_dim, self.hidden_size)
        self.fc_j = nn.Linear(self.hidden_size, self.j_classes)
        self.fc_m = nn.Linear(self.hidden_size, self.m_classes)

        self.bn_5 = nn.BatchNorm1d(self.hidden_size)

        self.setup_flag = True

        return f'Model Setup Completed!'

    def check_setup(self):
        assert self.setup_flag, f'Model Setup Incompleted! Use model.setup() to complete setup'

    def forward(self, x, bypass = False):
        if not bypass:
            self.check_setup()

        out = self.leaky_relu(self.bn_1(self.conv_1(x)))
        out = self.leaky_relu(self.bn_2(self.conv_2(out)))
        out = self.pool_1(out)

        out = self.leaky_relu(self.bn_3(self.conv_3(out)))
        out = self.leaky_relu(self.bn_4(self.conv_4(out)))
        out = self.pool_1(out)

        if bypass:
            return out

        out = out.view(-1, self.fc_dim)
        out = self.leaky_relu(self.bn_5(self.fc_1(out)))
        # out = self.softmax(self.fc_2(out))
        out_j = self.softmax(self.fc_j(out))
        out_m = self.softmax(self.fc_m(out))

        return out_j, out_m

class CustomConvNet(nn.Module):
    def __init__(self, INPUT_SIZE, N_CLASSES):
        super(CustomConvNet, self).__init__()

        self.leaky_relu = nn.LeakyReLU(0.05)
        self.softmax = nn.Softmax(dim = 1)

        self.conv_1 = nn.Conv2d(INPUT_SIZE, 256, kernel_size = (3, 3), stride = (1, 1), padding = 'same')
        self.conv_2 = nn.Conv2d(256, 512, kernel_size = (3, 3), stride = (1, 1), padding = 'same')
        self.conv_3 = nn.Conv2d(512, 512, kernel_size = (3, 3), stride = (1, 1), padding = 'same')
        self.conv_4 = nn.Conv2d(512, 512, kernel_size = (3, 3), stride = (1, 1), padding = 'same')

        self.bn_1 = nn.BatchNorm2d(256)
        self.bn_2 = nn.BatchNorm2d(512)
        self.bn_3 = nn.BatchNorm2d(512)
        self.bn_4 = nn.BatchNorm2d(512)

        self.pool_1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.pool_2 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))

        self.setup_flag = False
        self.n_classes = N_CLASSES

        return

    def setup(self, batch_shape = [32, 1, 28, 28]):
        batch_size = batch_shape[0]

        dummy_batch = torch.zeros(batch_shape)
        self.fc_dim = self.forward(dummy_batch, bypass = True).view(batch_size, -1).shape[-1]

        self.fc_1 = nn.Linear(self.fc_dim, 512)
        self.fc_2 = nn.Linear(512, self.n_classes)

        self.bn_5 = nn.BatchNorm1d(512)

        self.setup_flag = True

        return f'Model Setup Completed!'

    def check_setup(self):
        assert self.setup_flag, f'Model Setup Incompleted! Use model.setup() to complete setup'

    def forward(self, x, bypass = False):
        if not bypass:
            self.check_setup()

        out = self.leaky_relu(self.bn_1(self.conv_1(x)))
        out = self.leaky_relu(self.bn_2(self.conv_2(out)))
        out = self.pool_1(out)

        out = self.leaky_relu(self.bn_3(self.conv_3(out)))
        out = self.leaky_relu(self.bn_4(self.conv_4(out)))
        out = self.pool_1(out)

        if bypass:
            return out

        out = out.view(-1, self.fc_dim)
        out = self.leaky_relu(self.bn_5(self.fc_1(out)))
        out = self.softmax(self.fc_2(out))

        return out

class PretrainedEffiNet(nn.Module):
    def __init__(self, pretrained_model):
        super(PretrainedEffiNet, self).__init__()

        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim = 1)

        self.conv_1 = nn.Conv2d(1, 3, kernel_size = (3, 3), stride = (1, 1), padding = 'same')
        self.pretrained_model = pretrained_model

        self.fc_j = nn.Linear(1000, 14)
        self.fc_m = nn.Linear(1000, 10)

        self.bn_1 = nn.BatchNorm2d(3)

        return

    def forward(self, x):
        x = self.bn_1(self.leaky_relu(self.conv_1(x)))
        x = self.pretrained_model(x)
        out_j = self.softmax(self.fc_j(x))
        out_m = self.softmax(self.fc_m(x))

        return out_j, out_m

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dummy_model = CustomConvNet(1, 10)
    dummy_batch = torch.zeros([32, 1, 28, 28])

    dummy_model.setup(dummy_batch.shape)
    dummy_model = dummy_model.to(device)

    print(f'Out Shape: {dummy_model(dummy_batch.to(device)).shape}')