import torch
import torchvision
from torchvision import transforms
from network_files.faster_rcnn_framework import FasterRCNN, FastRCNNPredictor
from backbone.resnet50_fpn_model import resnet50_fpn_backbone
from network_files.rpn_function import AnchorsGenerator
from backbone.mobilenetv2_model import MobileNetV2
from draw_box_utils import draw_box
from PIL import Image
import json
import matplotlib.pyplot as plt
import numpy
import cv2
from PIL import Image
# 利用opencv显示多个镜头或窗口

def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    backbone = resnet50_fpn_backbone()
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)

    return model


def detect_hand(img1):
    global device
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # create model
    model = create_model(num_classes=12)  # 需要根据手势类别进行改变。
    # load train weights
    # train_weights = "./save_weights/model.pth"
    train_weights = "./save_hand_weights/resNetFpn-model-13.pth"
    model.load_state_dict(torch.load(train_weights)["model"])
    model.to(device)
    # read class_indict
    category_index = {}
    try:
        # json_file = open('./pascal_voc_classes.json', 'r')# voc 类别字典。
        json_file = open('./hand_classes.json', 'r')  # 手势文件字典。
        class_dict = json.load(json_file)
        category_index = {v: k for k, v in class_dict.items()}
    except Exception as e:
        print(e)
        exit(-1)
    # load image
    # original_img = Image.open("./test.jpg")#voc测试图片
    # original_img = Image.open("./ChuangyeguBusstop_Single_Good_color_2.jpg")  # 手势测试图片。
    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(img1)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    model.eval()
    with torch.no_grad():
        predictions = model(img.to(device))[0]
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        draw_box(img1,
                 predict_boxes,
                 predict_classes,
                 predict_scores,
                 category_index,
                 thresh=0.5,
                 line_thickness=5)
        # plt.imshow(img1)
        # plt.show()
        return img1


def show_video(video_file_path):
    '''读取视频路径文件，对视频进行双屏显示，右边显示处理测试后的效果'''
    ##选择摄像头
    # videoLeftUp = cv2.VideoCapture(0)
    # videoRightUp = cv2.VideoCapture(1)
    videoLeftUp = cv2.VideoCapture(video_file_path)
    width = (int(videoLeftUp.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = (int(videoLeftUp.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    while (videoLeftUp.isOpened()):
        retLeftUp, frameLeftUp = videoLeftUp.read()
        retRightUp, frameRightUp = videoLeftUp.read()

        frameLeftUp = cv2.resize(frameLeftUp, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
        # frameLeftUp = cv2.resize(frameLeftUp, (int(640), int(480)), interpolation=cv2.INTER_CUBIC)
        frameRightUp = cv2.resize(frameRightUp, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
        # frameRightUp = cv2.resize(frameRightUp, (int(640), int(480)), interpolation=cv2.INTER_CUBIC)
        frameRightUp = Image.fromarray(cv2.cvtColor(frameRightUp, cv2.COLOR_BGR2RGB))  # 将opencv 图像转换成PIL
        # 检测手势并且画出手势。
        frameRightUp = detect_hand(frameRightUp)
        # 图像转换回cv格式。
        frameRightUp = cv2.cvtColor(numpy.asarray(frameRightUp), cv2.COLOR_RGB2BGR)

        frameUp = numpy.hstack((frameLeftUp, frameRightUp))

        # {
        #     将frameUp图像进行保存为视频文件。
        #
        # }

        cv2.imshow('frame', frameUp)
        key = cv2.waitKey(60)
        if int(key) == 113:
            break
    videoLeftUp.release()
    cv2.destroyAllWindows()




def show_sigal_picture(picture_file):
    '''显示单个图片'''
    img = Image.open(picture_file)
    result = detect_hand(img)
    plt.imshow(result)
    plt.show()

###################船舰mobilenet模型#########################
def create_model_mobilenet(num_classes):
    # mobileNetv2+faster_RCNN
    backbone = MobileNetV2().features
    backbone.out_channels = 1280

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=[7, 7],
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model

def detect_hand_with_mob(img1):
    global device
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # create model
    model = create_model_mobilenet(num_classes=12)  # 需要根据手势类别进行改变。
    # load train weights
    # train_weights = "./save_weights/model.pth"
    train_weights = "./save_hand_weights/mobile-model-23.pth"
    model.load_state_dict(torch.load(train_weights)["model"])
    model.to(device)
    # read class_indict
    category_index = {}
    try:
        # json_file = open('./pascal_voc_classes.json', 'r')# voc 类别字典。
        json_file = open('./hand_classes.json', 'r')  # 手势文件字典。
        class_dict = json.load(json_file)
        category_index = {v: k for k, v in class_dict.items()}
    except Exception as e:
        print(e)
        exit(-1)
    # load image
    # original_img = Image.open("./test.jpg")#voc测试图片
    # original_img = Image.open("./ChuangyeguBusstop_Single_Good_color_2.jpg")  # 手势测试图片。
    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(img1)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    model.eval()
    with torch.no_grad():
        predictions = model(img.to(device))[0]
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        draw_box(img1,
                 predict_boxes,
                 predict_classes,
                 predict_scores,
                 category_index,
                 thresh=0.5,
                 line_thickness=5)
        # plt.imshow(img1)
        # plt.show()
        return img1



# picture_file = './x2.jpg'
# show_sigal_picture(picture_file)

# key = cv2.waitKey()
video_file_path = r'X:\jupyter_note_dontforget\contrl_with_handgesture\hand_vedio.mp4'
# video_file_path = r'X:\jupyter_note_dontforget\contrl_with_handgesture\hand_vedio2.mp4'

# show_video(video_file_path)
def show_sigal_picture_withmob(picture_file):
    '''显示单个图片'''
    img = Image.open(picture_file)
    result = detect_hand_with_mob(img)
    plt.imshow(result)
    plt.show()


picture_file = './pair.jpg'

import datetime
start = datetime.datetime.now()
show_sigal_picture(picture_file)
end = datetime.datetime.now()
print((end-start).seconds)