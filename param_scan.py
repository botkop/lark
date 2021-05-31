import timm
import torch

from lark.config import Config
from lark.learner import Learner
from lark.ops import MixedSig2Spec


def default_cfg():
    return Config(
        sites=['SSW'],
        use_neptune=True,
        n_epochs=10,
        bs=32,
        n_samples_per_label=1000,
        lr=1e-3,
        model='tf_efficientnet_b0_ns',
        # scheduler='torch.optim.lr_scheduler.CosineAnnealingLR',
        scheduler='torch.optim.lr_scheduler.OneCycleLR',
        loss_fn='lark.ops.SigmoidFocalLossStar',
        use_pink_noise=0.5,
        use_recorded_noise=0.5,
        use_overlays=True,
        apply_filter=0.2,
    )


par_dict = dict(
    use_pink_noise=0.5,
    use_recorded_noise=0.5,
    use_overlays=True,
    apply_filter=0.5,
)


def apply_par(k):
    cfg = default_cfg()
    setattr(cfg, k, par_dict[k])
    return cfg


configs = [apply_par(k) for k in par_dict]

def make_model(cfg):
    prep = MixedSig2Spec(cfg)
    main_model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True)
    posp = torch.nn.Sequential(
        torch.nn.Linear(in_features=1280, out_features=512, bias=True),
        torch.nn.Dropout(p=0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=512, out_features=len(cfg.labels), bias=True),
    )
    main_model.classifier = posp
    model = torch.nn.Sequential(prep, main_model)
    model = model.cuda()

    # prep = MixedSig2Spec(cfg)
    # main_model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
    # posp = torch.nn.Sequential(
    #     torch.nn.Linear(in_features=2048, out_features=1024, bias=True),
    #     torch.nn.Dropout(p=0.2),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(in_features=1024, out_features=512, bias=True),
    #     torch.nn.Dropout(p=0.2),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(in_features=512, out_features=len(cfg.labels), bias=True),
    # )
    # main_model.fc = posp
    # model = torch.nn.Sequential(prep, main_model)
    # model = model.cuda()
    return model

# for cfg in configs:
#     model = make_model(cfg)
#     lrn = Learner("tf_efficientnet_b0_ns-param-scan", cfg, model)
#     lrn.learn()

cfg = default_cfg()
model = make_model(cfg)
lrn = Learner("tf_efficientnet_b0_ns-param-scan", cfg, model)
lrn.learn()

lrn.evaluate()
lrn.load_checkpoint('best')
lrn.evaluate()

