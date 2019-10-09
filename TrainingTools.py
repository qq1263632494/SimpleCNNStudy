import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


class Classifier:
    def __init__(self, nn=None):
        # print(nn)
        self.nn = nn.cuda()
        # self.nn = nn

    def fit(self, train_set, batch_size, optim, loss_func, epoch, lr):
        global optimizer
        if optim == 'adam':
            optimizer = torch.optim.Adam(self.nn.parameters(), lr=lr)
        if optim == 'sgd':
            optimizer = torch.optim.SGD(self.nn.parameters(), lr=lr)
        if optim == 'asgd':
            optimizer = torch.optim.ASGD(self.nn.parameters(), lr=lr)
        if optim == 'adadelta':
            optimizer = torch.optim.Adadelta(self.nn.parameters(), lr=lr)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        i = 0
        list_x = []
        list_y = []
        from progressbar import ShowProcess
        bar = ShowProcess(epoch * len(train_set) / batch_size)
        for t in range(epoch):
            for step, (b_x, b_y) in enumerate(train_loader):
                b_x_t = b_x.cuda()
                b_y_t = b_y.cuda()
                # b_x_t = b_x
                # b_y_t = b_y
                output = self.nn(b_x_t)
                loss = loss_func(output, b_y_t)
                list_x.append(i)
                list_y.append(loss.item())
                i += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar.show_process()
        bar.close()
        print('\033[1;32m初始误差：\033[0m' + '\033[1;31m%.5f\033[0m' % list_y[0])
        print('\033[1;32m最终误差：\033[0m' + '\033[1;31m%.5f\033[0m' % list_y[-1])
        msg = str(type(self.nn))
        msg += '\n solver=' + optim + ' epoch=' + str(epoch) + ' learning rate =' + str(lr)
        plt.title(msg)
        plt.plot(list_x[10:], list_y[10:])
        plt.savefig('pic.png', bbox_inches='tight')
        plt.show()

    def load(self, path):
        self.nn = torch.load(path)

    def save(self, path):
        torch.save(self.nn, path)

    def evaluate(self, test_set, batch_size_test):
        test_loader = DataLoader(test_set, batch_size=batch_size_test, shuffle=False)
        list_pred = []
        list_true = []
        list_score = []
        for step, (t_x, t_y) in enumerate(test_loader):
            t_x_t = t_x.cuda()
            # t_x_t = t_x
            test_output = self.nn(t_x_t)
            pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
            list_pred.append(pred_y)
            list_true.append(t_y)
        from sklearn.metrics import accuracy_score
        total_length = len(test_set)
        iter_times = total_length / batch_size_test
        final_score = 0
        for i in range(int(iter_times)):
            final_score += accuracy_score(y_true=list_true[i], y_pred=list_pred[i])
        final_score /= iter_times
        print('%.4f' % final_score)
        return final_score

    def fit_with_LBFGS(self, train_set, batch_size, loss_func, epoch, lr):
        optimizer = torch.optim.LBFGS(self.nn.parameters(), lr=lr)
        loss_func = torch.nn.CrossEntropyLoss()
        list_x = []
        list_y = []
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        from progressbar import ShowProcess
        bar = ShowProcess(epoch * len(train_set) / batch_size)
        for t in range(epoch):
            for step, (b_x, b_y) in enumerate(train_loader):
                def closure():
                    b_x_t = b_x.cuda()
                    b_y_t = b_y.cuda()
                    output = self.nn(b_x_t)
                    loss = loss_func(output, b_y_t)
                    list_y.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    return loss

                optimizer.step(closure)
                bar.show_process()
        bar.close()
        for i in range(len(list_y)):
            list_x.append(i)
        print(list_y[0])
        print(list_y[-1])
        msg = str(type(self.nn))
        msg += '\n solver= lbgfs' + ' epoch=' + str(epoch) + ' learning rate =' + str(lr)
        plt.title(msg)
        plt.plot(list_x, list_y)
        plt.savefig('pic.png', bbox_inches='tight')
        plt.show()
