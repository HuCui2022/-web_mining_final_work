import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sbn

loss_of_resnet = pd.Series([0.0984,0.0449,0.0397,0.0372,0.0362,0.0281,0.0265,0.0256,0.0255,0.0251,0.0207,0.0195,0.0188,0.0182,0.0182,0.0182,0.0182,0.0182,0.0182,0.0182,0.0182,0.0182,0.0182,0.0182,0.0182,0.0182])
ap_resnet = pd.Series([0.754,0.783,0.791,0.794, 0.801,0.824,0.833,0.832, 0.829,0.827,0.845,0.847,0.845, 0.846,0.845,0.847,0.845, 0.846,0.845,0.847,0.845, 0.846,0.845,0.847,0.845, 0.846])
rc_resnet = pd.Series([0.812,0.837,0.845,0.845,0.853,0.876,0.879,0.881,0.876, 0.877,0.890,0.893,0.891,0.892,0.890,0.893,0.891,0.892,0.890,0.893,0.891,0.892,0.890,0.893,0.891,0.892 ])

loss_of_mobile = pd.Series([0.2700,0.1823,0.1617,0.1478,0.1396,0.0913,0.0627,0.0525,0.0477,0.0438,0.0378,0.0371,0.0368,0.0361,0.0355,0.0337,0.0335,0.0330,0.0331,0.0329,0.0321,0.0321,0.0319,0.0319,0.0318,0.0315,])
ap_mobile = pd.Series([0.372,0.447,0.512,0.547,0.590,0.732,0.745,0.756, 0.773,0.787,0.817,0.815,0.820,0.817,0.828,0.828,0.830,0.830,0.829,0.824,0.831,0.832,0.833,0.833,0.832,0.833])
rc_mobile = pd.Series([0.544,0.603,0.626,0.652,0.680,0.786,0.797, 0.810,0.823,0.834,0.862,0.861,0.866,0.863,0.865,0.874,0.873,0.875,0.874, 0.870,0.876,0.877,0.877,0.877,0.877,0.878])


def draw_loss(x,y,names):
    x.plot()
    y.plot()
    plt.title(names)
    plt.grid()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.legend(('loss of resnet','loss of mobilenet50'), loc='best')
    plt.xlabel('num_epoch')
    plt.ylabel("loss_value")
    plt.show()

def draw_ap(x, y,names):
    x.plot()
    y.plot()
    plt.title(names)
    plt.grid()
    plt.xlabel('num_epoch')
    plt.ylabel("Average Precision (AP) @[ IoU=0.50:0.95 ]")
    plt.legend(('MobileNet backbone','ResNet50 backbone'),loc = 'best')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.title('fasterRCNN 在不同的骨干网络下的AP')
    plt.show()

def draw_rc(x, y,names):
    x.plot()
    y.plot()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.title(names)
    plt.grid()
    plt.legend(('ResNet50 as backbone','MobileNetv2 as backbone'),loc='best')
    plt.xlabel('num_epoch')
    plt.ylabel(" Average Recall(AR) @[ IoU=0.50:0.95]")
    plt.show()
# draw_ap(ap_mobile,ap_resnet,"Average Precision of MobileNetV2 and ResNet50_backbone_fastercnn")
# draw_rc(rc_resnet,rc_mobile,'RC曲线，不同骨干网络的fastercnn的召回率')
draw_loss(loss_of_resnet,loss_of_mobile,'训练过程的损失')