import torch
from torch import nn
from collections import OrderedDict
from network_files.rpn_function import AnchorsGenerator, RPNHead, RegionProposalNetwork
from network_files.roi_head import RoIHeads
from torchvision.ops import MultiScaleRoIAlign
from torch.jit.annotations import Tuple, List, Dict, Optional
from torch import Tensor
import torch.nn.functional as F
import warnings
from network_files.transform import GeneralizedRCNNTransform


class FasterRCNNBase(nn.Module):
    """
    Main class for Generalized R-CNN. R-CNN基础类，用来泛化定义fasterrcnn

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        # 该模块就是将backbone,rpn,roi_heads 进行组合，计算损失。
        super(FasterRCNNBase, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                训练的时候，返回box和reg的损失loss
                During testing, it returns list[BoxList] contains additional fields
                测试的时候，返回的是 预测框列表。以及对应的分类分数。
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for target in targets:         # 进一步判断传入的target的boxes参数是否符合规定
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                          boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], []) # ????/

        # original_image_sizes : List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:] # 最后两个维度。
            assert len(val) == 2  # 防止输入的是个一维向量
            original_image_sizes.append((val[0], val[1]))
        # original_image_sizes = [img.shape[-2:] for img in images]

        images, targets = self.transform(images, targets)  # 对图像进行预处理，将全部图像统一规格。
        features = self.backbone(images.tensors)  # 将图像输入backbone得到特征图
        if isinstance(features, torch.Tensor):  # 若只在一层特征层上预测，将feature放入有序字典中，并编号为‘0’
            features = OrderedDict([('0', features)])  # 若在多层特征层上预测，传入的就是一个有序字典

        # 将特征层以及标注target信息传入rpn中
        proposals, proposal_losses = self.rpn(images, features, targets)
        # 从debug结果可以看出，有900多个proposals, 而 proposal_losses 返回的则是 loss_objectness, loss_rpn_box_reg 损失，格式为字典。

        # 将rpn生成的数据以及标注target信息传入fast rcnn后半部分
        # 从debug 结果可以看出，detector_loos 也是一个字典，分别对应loss_classifier, loss_box_reg.
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        # 对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        # loss 字典：将之前生成的四个loss添加到里边，分别为： roi_heads部分损失：loss_classifier,loss_box_reg, rpn部分损失：loss_objectness,loss_rpn_box_reg.
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

        # if self.training:
        #     return losses
        #
        # return detections


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class FasterRCNN(FasterRCNNBase):
    """
    Implements Faster R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.
    模型的输入应该是一个张量列表，每个张量的形状都是[C, H, W]，每个张量对应一个
    图像，并且应该在0-1范围内。不同的图像可以有不同的大小。

    The behavior of the model changes depending if it is in training or evaluation mode.
    模型的行为取决于它是在培训模式还是在评估模式中。
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    在训练的模式下，模型期望输入 系列图片张量以及它们对应的标签。（以字典列表的形式形式）
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
          between 0 and H and 0 and W
          标签列表包含： boxes： [N，4] 形状，N表示object数量，4 表示左上，右下坐标，是ground-truth 对应的。
        - labels (Int64Tensor[N]): the class label for each ground-truth box
            labels： 每一个ground truth 框对应的类别标签。

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    在推断期间，模型只需要输入张量，并返回后处理的张量
    预测作为一个列表[Dict[张量]]，一个为每个输入图像
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between
          0 and H and 0 and W
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
            计算特征的模型模块，包含通道属性，表示输出特征图的通道数，所有的特征图的通道数应该相同。返回的应该是一个tensor或者有序字典。
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
            模型的输出类型数，包括背景。如果指定了box_predictor,那么该值应该为none。
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        最大和最小尺度，在图像被送入backbone之前。
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
            标准化的平均值和标准差。
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
            为一组特征图（不同尺度）生成anchor的模块，继承AnchorGenerator.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        从rpn计算对象的回归增量和对象的模块。
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        极大值抑制前要保留的proposals 数量。
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        再测试的过程中，要保留的proposals 数量。
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        训练过程中，极大值抑制后，保留的proposal个数。
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        测试过程中，极大值抑制后。。。。
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        用于过滤的极大值抑制的阈值。

        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN. 正样本阈值。

        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
            负样本阈值。

        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss  rpn 模块中，采样的anchors ，用来计算损失的数量。（并不是计算全部）

        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN（训练过程中正样本比例。）fraction.部分。

        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
            裁剪，resize ，模块，用来定位边界框。

        box_head (nn.Module): module that takes the cropped feature maps as input
        将裁剪的特征图进行输入的模块。

        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
            预测和回归模块。

        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
            分数阈值。

        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        预测模块的非极大值抑制。

        box_detections_per_img (int): maximum number of detections per image, for all classes.
        每张图像预测的边界框个数。

        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
            分类部分的最小iou阈值。正样本。大于这个值，都是正样本。

        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
            最大阈值，负样本。小于这个值都是负样本。

        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
            再分类阶段，每张图提取的边界框个数。

        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
            分类阶段，正样本比例。

        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
            编码和解码权重。

    """

    def __init__(self, backbone, num_classes=None,
                 # transform parameter
                 min_size=800, max_size=1344,      # 预处理resize时限制的最小尺寸与最大尺寸
                 image_mean=None, image_std=None,  # 预处理normalize时使用的均值和方差
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,    # rpn中在nms处理前保留的proposal数(根据score)
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,  # rpn中在nms处理后保留的proposal数
                 rpn_nms_thresh=0.7,  # rpn中进行nms处理时使用的iou阈值
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,  # rpn计算损失时，采集正负样本设置的阈值
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,  # rpn计算损失时采样的样本数，以及正样本占总样本的比例
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 # 移除低目标概率      fast rcnn中进行nms处理的阈值   对预测结果根据score排序取前100个目标
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,   # fast rcnn计算误差时，采集正负样本设置的阈值
                 box_batch_size_per_image=512, box_positive_fraction=0.25,  # fast rcnn计算误差时采样的样本数，以及正样本占所有样本的比例
                 bbox_reg_weights=None):
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels"
                "specifying the number of output channels  (assumed to be the"
                "same for all the levels"
            )

        assert isinstance(rpn_anchor_generator, (AnchorsGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

       # 不能同时为none
        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor "
                                 "is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        # 预测特征层的channels
        out_channels = backbone.out_channels

        # 若anchor生成器为空，则自动生成针对resnet50_fpn的anchor生成器
        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,)) #加逗号才是元组类型。
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorsGenerator(
                anchor_sizes, aspect_ratios
            )

        # 生成RPN通过滑动窗口预测网络部分
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        # 默认rpn_pre_nms_top_n_train = 2000, rpn_pre_nms_top_n_test = 1000,
        # 默认rpn_post_nms_top_n_train = 2000, rpn_post_nms_top_n_test = 1000,
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        # 定义整个RPN框架
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        #  Multi-scale RoIAlign pooling
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],  # 在哪些特征层进行roi pooling
                output_size=[7, 7],
                sampling_ratio=2)

        # fast RCNN中roi pooling后的展平处理两个全连接层部分
        if box_head is None:
            resolution = box_roi_pool.output_size[0]  # 默认等于7
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size
            )

        # 在box_head的输出上预测部分
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        # 将roi pooling, box_head以及box_predictor结合在一起
        roi_heads = RoIHeads(
            # box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        # 对数据进行标准化，缩放，打包成batch等处理部分
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        # 将以上定义的各个模块放入传入FasterRcnnBase中。
        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)

