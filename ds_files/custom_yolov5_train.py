class GAM(nn.Module):
    """
    Класс Global Attention Module
    """
    def __init__(self, in_channels, out_channels):
        super(GAM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Channel attention
        self.theta = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.psi = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_out = nn.Conv2d(out_channels, in_channels, kernel_size=1)  # to combine the output

    def forward(self, x):
        # Channel attention
        theta = self.theta(x)
        phi = self.phi(x)
        psi = self.psi(x)

        theta = theta.view(-1, self.out_channels, x.size(2) * x.size(3))
        phi = phi.view(-1, self.out_channels, x.size(2) * x.size(3)).permute(0, 2, 1)
        psi = psi.view(-1, self.out_channels, x.size(2) * x.size(3))

        attn = F.softmax(torch.bmm(theta, phi), -1)
        out = torch.bmm(psi, attn.permute(0, 2, 1))
        out = out.view(-1, self.out_channels, x.size(2), x.size(3))

        # Spatial attention
        out = self.conv_out(out)
        out = out + x  # skip connection
        return out


class DetectionModelGAM(BaseModel):
    """
    Класс DetectionModel с GAM после четвертого сверточного слоя
    """
    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)

        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])

        # добавляем GAM после выбранного сверточного слоя
        self.model[4] = nn.Sequential(self.model[4], GAM(in_channels=64, out_channels=64))

        self.names = [str(i) for i in range(self.yaml["nc"])]
        self.inplace = self.yaml.get("inplace", True)

        m = self.model[-1]
        if isinstance(m, (Detect, Segment)):
            s = 256
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()

        initialize_weights(self)
        self.info()
        LOGGER.info("")

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)
        return self._forward_once(x, profile, visualize)

    def _forward_augment(self, x):
        img_size = x.shape[-2]
        s = [1, 0.83, 0.67]
        f = [None, 3, None]
        y = []
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)
        return torch.cat(y, 1), None

    def _descale_pred(self, p, flips, scale, img_size):
        if self.inplace:
            p[..., :4] /= scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale
            if flips == 2:
                y = img_size[0] - y
            elif flips == 3:
                x = img_size[1] - x
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        nl = self.model[-1].nl
        g = sum(4**x for x in range(nl))
        e = 1
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))
        y[0] = y[0][:, :-i]
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))
        y[-1] = y[-1][:, i:]
        return y

    def _initialize_biases(self, cf=None):
        m = self.model[-1]
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

Model = DetectionModelGAM

def custom_yolov5_train(
        images_path_train: str,
        images_path_val: str,
        batch_size: int = 1,
        epochs: int = 1
):
    """
    Обучение yolov5 на custom data

    :code_assign: users
    :code_type: Пользовательские функции
    :imports: init_gui_dict, image_to_base64
    :launch: pytorch

    :param_block str images_path_train PATH_OD_TRAIN: путь до изображений для обучения
    :param_block str images_path_val PATH_OD_VAL: путь до изображений для валидации
    :param int batch_size: размер мини-батча
    :param int epochs: количество эпох

    :returns: gui_dict, weights, error
    :rtype: dict, str, str
    :semrtype: , YOLOv5,

    :packages:
    from datetime import datetime
    import json
    import os
    import yaml
    from yolov5 import train as yolov5_train_lib

    Параметры:
    images_path_train: путь до изображений для обучения
    images_path_val: путь до изображений для валидации
    batch_size: размер мини-батчей
    epochs: количество эпох
    """

    # инициализируем ошибку
    error = None

    # инициализируем папку сохранения обучения
    SAVE_DIR = '/var/training/'

    images_dict = {
        "classes": ["layer_damage", "furrow", "contour_blurring", "incomplete_spreading"]
        }

    # формируем словарь для yaml файла
    info_images = {'train': os.path.join(images_path_train, 'images/'), 'val': os.path.join(images_path_val, 'images/'),
                   'nc': len(images_dict['classes']), 'names': images_dict['classes']}

    # формируем индекс файла из текущего времени
    file_index = str(datetime.now().timestamp())
    # название файла
    yaml_file_name = file_index + '.yaml'
    # записываем пути и классы в yaml файл
    with open(yaml_file_name, 'w') as f:
        yaml.dump(info_images, f, sort_keys=False, default_flow_style=False)

    hyp_params = {
        'lr0': 0.001,  # initial learning rate (SGD=1E-2, Adam=1E-3)
        'lrf': 0.1,  # final OneCycleLR learning rate (lr0 * lrf)
        'momentum': 0.937,  # SGD momentum/Adam beta1
        'weight_decay': 0.0005,  # optimizer weight decay 5e-4
        'warmup_epochs': 3.0,  # warmup epochs (fractions ok)
        'warmup_momentum': 0.8,  # warmup initial momentum
        'warmup_bias_lr': 0.1,  # warmup initial bias lr
        'box': 0.5,  # box loss gain
        'cls': 0.01,  # cls loss gain
        'cls_pw': 1.0,  # cls BCELoss positive_weight
        'obj': 0.8,  # obj loss gain (scale with pixels)
        'obj_pw': 1.0,  # obj BCELoss positive_weight
        'iou_t': 0.5,  # IoU training threshold
        'anchor_t': 4.0,  # anchor-multiple threshold
        'fl_gamma': 1.5,  # focal loss gamma (efficientDet default gamma=1.5)
        'hsv_h': 0,  # image HSV-Hue augmentation (fraction)
        'hsv_s': 0,  # image HSV-Saturation augmentation (fraction)
        'hsv_v': 0,  # image HSV-Value augmentation (fraction)
        'degrees': 0.0,  # image rotation (+/- deg)
        'translate': 0,  # image translation (+/- fraction)
        'scale': 0.75,  # image scale (+/- gain)
        'shear': 0,  # image shear (+/- deg)
        'perspective': 0.0,  # image perspective (+/- fraction), range 0-0.001
        'flipud': 0.0,  # image flip up-down (probability)
        'fliplr': 0.0,  # image flip left-right (probability)
        'mosaic': 0.001,  # image mosaic (probability)
        'mixup': 0.0,  # image mixup (probability)
        'copy_paste': 0.0  # segment copy-paste (probability)
    }

    # формируем индекс файла из текущего времени
    file_index = str(datetime.now().timestamp())
    # название файла
    hyp = 'hyp_' + file_index + '.yaml'
    # записываем пути и классы в yaml файл
    with open(hyp, 'w') as f:
        yaml.dump(hyp_params, f, sort_keys=False, default_flow_style=False)

    try:
        # обучаем
        opt = yolov5_train_lib.run(imgsz=640,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   data=yaml_file_name,
                                   cache='ram',
                                   weights='yolov5s.pt',
                                   project=SAVE_DIR,
                                   name=file_index,
                                   freeze=[0])
    # ошибка
    except Exception as err:
        error = f'Ошибка обучения YOLOv5: {err}'
        return None, None, error

    try:
        # сохраняем историю обучения
        img_str_hist = image_to_base64(SAVE_DIR + file_index + '/results.png', "png")
    # ошибка
    except Exception as err:
        error = f'Ошибка сохранения истории обучения YOLOv5: {err}'
        return None, None, error

    try:
        # сохраняем confusion_matrix
        img_str_conf_matrix = image_to_base64(SAVE_DIR + file_index + '/confusion_matrix.png', "png")
    # ошибка
    except Exception as err:
        error = f'Ошибка сохранения матрицы ошибок YOLOv5: {err}'
        return None, None, error

    try:
        # сохраняем train_batch0
        img_str_train_batch0 = image_to_base64(SAVE_DIR + file_index + '/train_batch0.jpg', "jpeg")
    # ошибка
    except Exception as err:
        error = f'Ошибка сохранения фрагмента тренировочного датасета YOLOv5: {err}'
        return None, None, error

    try:
        # сохраняем val_batch0_pred
        img_str_val_batch0_pred = image_to_base64(SAVE_DIR + file_index + '/val_batch0_pred.jpg', "jpeg")
    # ошибка
    except Exception as err:
        error = f'Ошибка сохранения примеров валидации YOLOv5: {err}'
        return None, None, error

    try:
        # сохраняем val_batch0_labels
        img_str_val_batch0_labels = image_to_base64(SAVE_DIR + file_index + '/val_batch0_labels.jpg', "jpeg")
    # ошибка
    except Exception as err:
        error = f'Ошибка сохранения разметки валидационного датасета YOLOv5: {err}'
        return None, None, error

    # удаляем yaml файл
    os.remove(yaml_file_name)

    # сохраняем путь до весов
    weights = SAVE_DIR + file_index + '/weights/best.pt'

    # инициализируем словарь для gui
    gui_dict = init_gui_dict()
    # сохраняем модель
    gui_dict['text'].append({'title': 'Модель',
                             'value': 'YOLOv5(' + 'batch_size=' + str(opt.batch_size) + ', ' + 'epochs=' + str(
                                 opt.epochs) + ', ' + \
                                      'imgsz=' + str(opt.imgsz) + ', ' + 'optimizer=' + str(
                                 opt.optimizer) + ', ' + 'classes=' + str(images_dict['classes']) + ')'})
    # записываем историю обучения в gui_dict
    gui_dict['image'].append({'title': 'История обучения', 'value': img_str_hist})
    # записываем матрицу ошибок в gui_dict
    gui_dict['image'].append({'title': 'Матрица ошибок', 'value': img_str_conf_matrix})
    # записываем img_str_train_batch0 в gui_dict
    gui_dict['image'].append({'title': 'Фрагмент тренировочного датасета', 'value': img_str_train_batch0})
    # записываем img_str_val_batch0_labels в gui_dict
    gui_dict['image'].append({'title': 'Целевая разметка на валидационной выборке', 'value': img_str_val_batch0_labels})
    # записываем val_batch0_pred в gui_dict
    gui_dict['image'].append({'title': 'Итоговая разметка на валидационной выборке', 'value': img_str_val_batch0_pred})

    return gui_dict, weights, error

