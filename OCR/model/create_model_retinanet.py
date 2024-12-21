import torch

from pytorch_retinanet.loss import FocalLoss
from pytorch_retinanet.retinanet import RetinaNet
from pytorch_retinanet.encoder import DataEncoder

from braille_utils import label_tools

def create_model_retinanet(params, device):
    use_multiple_class_groups = params.data.get('class_as_6pt', False)
    num_classes = 1 if params.data.get_points else ([1]*6 if use_multiple_class_groups else 64)
    encoder = DataEncoder(**params.model_params.encoder_params)
    model = RetinaNet(num_layers=encoder.num_layers(), num_anchors=encoder.num_anchors(),
                      num_classes=num_classes,
                      num_fpn_layers=params.model_params.get('num_fpn_layers', 0)).to(device)
    retina_loss = FocalLoss(num_classes=num_classes, **params.model_params.get('loss_params', dict()))


    def detection_collate(batch):
        boxes = [torch.tensor(b[1][:, :4], dtype = torch.float32, device=device)
                 *torch.tensor(params.data.net_hw[::-1]*2, dtype = torch.float32, device=device) for b in batch]
        labels = [torch.tensor(b[1][:, 4], dtype = torch.long, device=device) for b in batch]
        if params.data.get_points:
            labels = [torch.tensor([0]*len(lb), dtype = torch.long, device=device) for lb in labels]
        elif use_multiple_class_groups:
            labels = [torch.tensor([[int(ch)-1 for ch in label_tools.int_to_label010(int_lbl.item())] for int_lbl in lb],
                                   dtype=torch.long, device=device) for lb in labels]

        original_images = [b[3] for b in batch if len(b)>3]

        imgs = [x[0] for x in batch]
        calc_cls_mask = torch.tensor([b[2].get('calc_cls', True) for b in batch],
                                dtype=torch.bool,
                                device=device)

        h, w = tuple(params.data.net_hw)
        num_imgs = len(batch)
        inputs = torch.zeros(num_imgs, 3, h, w).to(imgs[0])

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            labels_i = labels[i]
            if use_multiple_class_groups and len(labels_i.shape) != 2:
                labels_i = labels_i.reshape((0, len(num_classes)))
            loc_target, cls_target, max_ious = encoder.encode(boxes[i], labels_i, input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        if original_images: 
            return inputs, ( torch.stack(loc_targets), torch.stack(cls_targets), calc_cls_mask), original_images
        else:
            return inputs, (torch.stack(loc_targets), torch.stack(cls_targets), calc_cls_mask)

    class Loss:
        def __init__(self):
            self.encoder = encoder
            pass
        def __call__(self, pred, targets):
            loc_preds, cls_preds = pred
            loc_targets, cls_targets, calc_cls_mask = targets
            if calc_cls_mask.min():
                calc_cls_mask = None
            loss = retina_loss(loc_preds, loc_targets, cls_preds, cls_targets, cls_calc_mask=calc_cls_mask)
            return loss
        def get_dict(self, *kargs, **kwargs):
            return retina_loss.loss_dict
        def metric(self, key):
            def call(*kargs, **kwargs):
                return retina_loss.loss_dict[key]
            return call

    return model, detection_collate, Loss()

if __name__ == '__main__':
    pass
