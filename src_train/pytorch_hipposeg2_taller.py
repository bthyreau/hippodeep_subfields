import torch
import nibabel
import numpy as np
import os

#bundle = np.load("../data/bundle64_aug00.npy", mmap_mode="r")

from pylab import *

ion()

import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0")

torch.backends.cudnn.benchmark=False # faster on GTX1080TI

class HippoModel(nn.Module):
    def __init__(self):
        super(HippoModel, self).__init__()
        self.conv0a = nn.Conv3d(1, 12, 3, padding=1)
        self.conv0ap = nn.Conv3d(12, 12, 3, padding=1)
        self.conv0b = nn.Conv3d(12, 12, 3, padding=1)
        self.bn0a = nn.BatchNorm3d(12)

        self.ma1 = nn.MaxPool3d(2)
        self.conv1a = nn.Conv3d(12, 12, 3, padding=1)
        self.conv1ap = nn.Conv3d(12, 12, 3, padding=1)
        self.conv1b = nn.Conv3d(12, 12, 3, padding=1)
        self.bn1a = nn.BatchNorm3d(12)

        self.ma2 = nn.MaxPool3d(2)
        self.conv2a = nn.Conv3d(12, 16, 3, padding=1)
        self.conv2ap = nn.Conv3d(16, 16, 3, padding=1)
        self.conv2b = nn.Conv3d(16, 16, 3, padding=1)
        self.bn2a = nn.BatchNorm3d(16)

        self.ma3 = nn.MaxPool3d(2)
        self.conv3a = nn.Conv3d(16, 32, 3, padding=1)
        self.conv3ap = nn.Conv3d(32, 32, 3, padding=1)
        self.conv3b = nn.Conv3d(32, 24, 3, padding=1)
        self.bn3a = nn.BatchNorm3d(24)

        # up

        self.conv2u = nn.Conv3d(24, 16, 3, padding=1)
        self.conv2up = nn.Conv3d(16, 16, 3, padding=1)
        self.bn2u = nn.BatchNorm3d(16)
        self.conv2v = nn.Conv3d(16+16, 16, 3, padding=1)

        # up

        self.conv1u = nn.Conv3d(16, 12, 3, padding=1)
        self.conv1up = nn.Conv3d(12, 12, 3, padding=1)
        self.bn1u = nn.BatchNorm3d(12)
        self.conv1v = nn.Conv3d(12+12, 12, 3, padding=1)

        # up

        self.conv0u = nn.Conv3d(12, 12, 3, padding=1)
        self.conv0up = nn.Conv3d(12, 12, 3, padding=1)
        self.bn0u = nn.BatchNorm3d(12)
        self.conv0v = nn.Conv3d(12+12, 12, 3, padding=1)

        self.conv1x = nn.Conv3d(12, 13, 1, padding=0)

    def forward(self, x):
        x = F.elu(self.conv0a(x))
        self.li0 = x = F.elu(self.conv0ap (F.elu(self.bn0a(self.conv0b(x))) ))

        x = self.ma1(x)
        x = F.elu(self.conv1ap( F.elu(self.conv1a(x)) ))
        self.li1 = x = F.elu(self.bn1a(self.conv1b(x)))

        x = self.ma2(x)
        x = F.elu(self.conv2ap( F.elu(self.conv2a(x)) ))
        self.li2 = x = F.elu(self.bn2a(self.conv2b(x)))

        x = self.ma3(x)
        x = F.elu(self.conv3ap( F.elu(self.conv3a(x)) ))
        self.li3 = x = F.elu(self.bn3a(self.conv3b(x)))

        x = F.interpolate(x, scale_factor=2, mode="nearest")

        x = F.elu(self.conv2up( F.elu(self.bn2u(self.conv2u(x))) ))
        x = torch.cat([x, self.li2], 1)
        x = F.elu(self.conv2v(x))

        self.lo1 = x
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        x = F.elu(self.conv1up( F.elu(self.bn1u(self.conv1u(x))) ))
        x = torch.cat([x, self.li1], 1)
        x = F.elu(self.conv1v(x))

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        self.la1 = x

        x = F.elu(self.conv0up( F.elu(self.bn0u(self.conv0u(x))) ))
        x = torch.cat([x, self.li0], 1)
        x = F.elu(self.conv0v(x))

        self.out = x = self.conv1x(x)
        #x = torch.sigmoid(x)
        return x






ids = open("../shufids1946.txt").read().split()
label_codes0 = np.array([0, 203, 204, 205, 206, 208, 209, 210, 211, 212, 214, 215, 226], np.uint8)
assert (sorted(label_codes0) == label_codes0).all()

#label_names = 
def iterate_minibatches(maxidx, batchsize = 16):
    assert maxidx <= len(ids)
    assert maxidx % batchsize == 0
    perms = np.random.permutation(maxidx)

    #output = torch.zeros(batchsize, 13, 128,128,128, dtype=torch.float32, device=device)
    orig = np.zeros((batchsize,  1, 128,128,128), dtype=np.float32)
    output = np.zeros((batchsize, 1, 128,128,128), dtype=np.uint8)

    for i in range(0, maxidx, batchsize):
        idx_pick = perms[i:i+batchsize]
        for n in range(batchsize):
            subjname = ids[idx_pick[n]]
            augnum = np.random.permutation([0,1,2,3,4,7,8,9])[0]
            side = np.random.randint(2)
            if side == 0:
                fn = "../data/generated/g_%s_001_orig_boxL_v%04d_intens.nii.gz" % (subjname, augnum)
                orig[n,0] = np.asarray(nibabel.load(fn).dataobj) # useless: slope, mul, float64
                orig[n,0] -= orig[n,0].mean()
                orig[n,0] /= orig[n,0].std()
                fn = "../data/generated/g_%s_001_orig_boxL_v%04d_labels.nii.gz" % (subjname, augnum)
                output[n, 0] = np.searchsorted(label_codes0, np.asarray(nibabel.load(fn).dataobj))
            else:
                fn = "../data/generated/g_%s_001_orig_boxR_v%04d_intens.nii.gz" % (subjname, augnum)
                orig[n,0] = np.asarray(nibabel.load(fn).dataobj)[::-1] # useless: slope, mul, float64
                orig[n,0] -= orig[n,0].mean()
                orig[n,0] /= orig[n,0].std()
                fn = "../data/generated/g_%s_001_orig_boxR_v%04d_labels.nii.gz" % (subjname, augnum)
                output[n, 0] = np.searchsorted(label_codes0, np.asarray(nibabel.load(fn).dataobj)[::-1])
        yield orig, output



borig = np.zeros((6, 1, 128,128,128), dtype=np.float32)
boutput = np.zeros((6, 1, 128,128,128), dtype=np.uint8)
augnum = 0
for n, subjname in enumerate(ids[-6:]):
    fn = "../data/generated/g_%s_001_orig_boxL_v%04d_intens.nii.gz" % (subjname, augnum)
    borig[n,0] = np.asarray(nibabel.load(fn).dataobj) # useless: slope, mul, float64
    borig[n,0] -= borig[n,0].mean()
    borig[n,0] /= borig[n,0].std()
for n, subjname in enumerate(ids[-6:]):
    fn = "../data/generated/g_%s_001_orig_boxL_v%04d_labels.nii.gz" % (subjname, augnum)
    boutput[n] = np.asarray(nibabel.load(fn).dataobj)


from collections import OrderedDict
import pickle

import matplotlib.pyplot as plt
figloss, _ = plt.subplots(dpi=48, nrows=2)
plt.ion()
figloss.show()
from collections import OrderedDict
import pickle
Dloss = OrderedDict()
Dloss.update( OrderedDict({0: [0,0]}) )

qt_processEvents = lambda : None
try:
    import PyQt5.QtWidgets
    qt_processEvents = PyQt5.QtWidgets.QApplication.processEvents
except:
    pass
try:
    from PyQt4.QtGui import qApp
    qt_processEvents = qApp.processEvents
except:
    pass


def redraw_plot_loss():
    figloss.axes[0].clear()
    figloss.axes[0].plot(np.array(list(Dloss.keys()))[-120:], np.array(list(Dloss.values()))[-120:,0], "+-")
    figloss.axes[1].clear()
    figloss.axes[1].plot(np.array(list(Dloss.keys()))[-12:], np.array(list(Dloss.values()))[-12:,0], "+-")
    figloss.tight_layout()
    plt.draw()
    for i in range(20):
        qt_processEvents()





import torch.optim as optim

net = HippoModel()
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=0.0005)


if 0:
    torch.save(model.state_dict(), "torchparams/params_head.pt")
    net = Model()
    net.load_state_dict(torch.load("torchparams/params_head.pt"))


epoch = 0
Dloss.clear()
Dloss.update( OrderedDict({0: [0,0]}) )
num_epochs = 100
train_batches = 0

loss_fn = nn.CrossEntropyLoss()

for epoch in range(epoch, num_epochs):

    if (epoch == 35 or epoch == 85):
            optimizer.param_groups[0]["lr"] /= 10
            print("changing to", optimizer.param_groups[0]["lr"])

    try:
        if epoch == 0:
            nibabel.Nifti1Image(np.rollaxis(np.asarray(borig[:,0]), 0, 4), np.identity(4)).to_filename("/tmp/test_orig.nii")
            nibabel.Nifti1Image(np.rollaxis(np.asarray(boutput[:,0]), 0, 4), np.identity(4)).to_filename("/tmp/test_labs.nii")
        print("Apply on test data")
        with torch.no_grad():
            out1 = np.asarray(net(torch.from_numpy(borig[:,0:1].copy()).to(device)).to("cpu"))
        out2 = np.zeros((6, 128,128,128), np.uint8)
        for s in range(6):
            out2[s] = out1[s].argmax(axis=0)
        nibabel.Nifti1Image(np.rollaxis(out2, 0, 4), np.identity(4)).to_filename("/tmp/test6_0.nii")
        print("Done.")
        redraw_plot_loss()
    except:
        pass

    train_err = 0.
    train_batches = 0
    loss = torch.tensor([0])
    start_time = time.time()
    batchsize=4
    tmp_err = []
    ct = time.time()

    torch.save(net.state_dict(), "torchparams/params_hipposub2_tall1_%05d_%05d_train2.pt" % (epoch, train_batches))
    for inputs, labels in iterate_minibatches(1920, batchsize):
        err = loss.item() # it was computed before (at previous loop) but defered the (blocking) read to here, after the heavy CPU/IO
        in1 = torch.from_numpy(inputs).to(device)
        out1 = torch.from_numpy(labels[:,0,...]).to(device, dtype=torch.long)

        optimizer.zero_grad()   # zero the gradient buffers
        prediction = net(in1)
        loss = loss_fn(prediction, out1)
        loss.backward()
        optimizer.step()

        tmp_err.append(float(err))
        train_err += err
        train_batches += 1
        if train_batches % 80 == 0:
            print( "% 5d err: %8.8f  (%.2f s)" % (train_batches, np.mean(tmp_err), time.time() - start_time))
            Dloss[epoch + train_batches / ((1920. * 1)/batchsize)] = (np.mean(tmp_err), 0)
            del tmp_err[:]
        if train_batches % 20 == 0:
            redraw_plot_loss()
        if not np.isfinite(train_err):
            break
    print("Epoch {:4d} train {:.8f}\t{} ({:.3f}s)".format(epoch,  train_err / train_batches, time.ctime(), time.time() - start_time))

