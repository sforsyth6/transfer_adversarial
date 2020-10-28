# from https://medium.com/dsnet/the-1-cycle-policy-an-experiment-that-vanished-the-struggle-in-training-neural-nets-184417de23b9
import math
import matplotlib.pyplot as plt

def update_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr

def update_mom(optimizer, mom):
    for g in optimizer.param_groups:
        g['momentum'] = mom

class CLR():
    def __init__(self, train_dataloader, base_lr=1e-5, max_lr=100):
        self.base_lr = base_lr # The lower boundary for learning rate (initial lr)
        self.max_lr = max_lr # The upper boundary for learning rate
        self.bn = len(train_dataloader) - 1 # Total number of iterations used for this test run (lr is calculated based on this)
        ratio = self.max_lr/self.base_lr # n
        self.mult = ratio ** (1/self.bn) # q = (max_lr/init_lr)^(1/n)
        self.best_loss = 1e9 # our assumed best loss
        self.iteration = 0 # current iteration, initialized to 0
        self.lrs = []
        self.losses = []

    def calc_lr(self, loss):
        self.iteration +=1
        if math.isnan(loss) or loss > 4 * self.best_loss: # stopping criteria (if current loss > 4*best loss)
            return -1
        if loss < self.best_loss and self.iteration > 1: # if current_loss < best_loss, replace best_loss with current_loss
            self.best_loss = loss
        mult = self.mult ** self.iteration # q = q^i
        lr = self.base_lr * mult # lr_i = init_lr * q
        self.lrs.append(lr) # append the learing rate to lrs
        self.losses.append(loss) # append the loss to losses
        return lr

    def plot(self, start=10, end=-5): # plot lrs vs losses
        plt.xlabel("Learning Rate")
        plt.ylabel("Losses")
        plt.plot(self.lrs[start:end], self.losses[start:end])
        plt.xscale('log') # learning rates are in log scale


class OneCycle():
    def __init__(self, nb, max_lr, momentum_vals=(0.95, 0.85), prcnt= 10 , div=10):
        self.nb = nb # total number of iterations including all epochs
        self.div = div # the division factor used to get lower boundary of learning rate
        self.step_len =  int(self.nb * (1- prcnt/100)/2)
        self.high_lr = max_lr # the optimum learning rate, found from LR range test
        self.low_mom = momentum_vals[1]
        self.high_mom = momentum_vals[0]
        self.prcnt = prcnt # percentage of cycle length,we annihilate learning rate below the lower learnig rate (default is 10)
        self.iteration = 0
        self.lrs = []
        self.moms = []

    def calc(self): # calculates learning rate and momentum for the batch
        self.iteration += 1
        lr = self.calc_lr()
        mom = self.calc_mom()
        return (lr, mom)

    def calc_lr(self):
        if self.iteration==self.nb: # exactly at `d`
            self.iteration = 0
            self.lrs.append(self.high_lr/self.div)
            return self.high_lr/self.div
        if self.iteration > 2 * self.step_len: # case c-d
            ratio = (self.iteration - 2 * self.step_len) / (self.nb - 2 * self.step_len)
            lr = self.high_lr * ( 1 - ratio * (1-(1/self.div))/self.div
        elif self.iteration > self.step_len: # case b-c
            ratio = 1- (self.iteration -self.step_len)/self.step_len
            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
        else : # case a-b
            ratio = self.iteration/self.step_len
            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
        self.lrs.append(lr)
        return lr

    def calc_mom(self):
        if self.iteration==self.nb: # exactly at `d`
            self.iteration = 0
            self.moms.append(self.high_mom)
            return self.high_mom
        if self.iteration > 2 * self.step_len: # case c-d
            mom = self.high_mom
        elif self.iteration > self.step_len: # case b-c
            ratio = (self.iteration -self.step_len)/self.step_len
            mom = self.low_mom + ratio * (self.high_mom - self.low_mom)
        else : # case a-b
            ratio = self.iteration/self.step_len
            mom = self.high_mom - ratio * (self.high_mom - self.low_mom)
        self.moms.append(mom)
        return mom
