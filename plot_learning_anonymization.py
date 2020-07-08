import keras as ke
from matplotlib import pyplot as plt


class PlotLearning(ke.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 1
        self.x = []
        self.losses = []
        self.val_losses = []
        self.aux = []
        self.val_aux = []
        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.aux.append(logs.get('mean_absolute_percentage_error'))
        self.val_aux.append(logs.get('val_mean_absolute_percentage_error'))
        self.i += 1

        plt.clf()
        plt.subplot(211)
        plt.yscale('log')
        plt.plot(self.x, self.losses, 'g-x')
        plt.plot(self.x, self.val_losses, 'r-x')
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.grid(which='major', axis='both')

        plt.subplot(212)
        plt.plot(self.x, self.aux, 'g-x')
        plt.plot(self.x, self.val_aux, 'r-x')
        #plt.ylim(0, 1)
        plt.yscale('log')
        plt.xlabel('Generation')
        plt.ylabel('MAPE')
        plt.grid(which='major', axis='both')

        plt.show()
        plt.pause(0.01)

