from lark.data import *
from lark.learner import Learner
from lark.ops import Sig2Spec, MixedSig2Spec

torch.cuda.set_device(0)
torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)

cfg = Config(
    n_workers=12,

    n_fft=3200,
    window_length=3200,
    n_mels=128,
    hop_length=800,

    use_pink_noise=0,
    use_recorded_noise=0,
    use_overlays=False,
    apply_filter=0,
    sites=['COR'],
    use_neptune=True,
    log_batch_metrics=False,
    n_epochs=150,
    bs=32,
    lr=1e-3,
    model='resnest50',
    scheduler='torch.optim.lr_scheduler.CosineAnnealingLR'
)

prep = MixedSig2Spec(cfg)
main_model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)

for param in main_model.parameters():
    param.requires_grad = False

for layer in [main_model.layer2, main_model.layer3, main_model.layer4, main_model.avgpool]:
    for param in layer.parameters():
        param.requires_grad = True


posp = torch.nn.Sequential(
    torch.nn.Linear(in_features=2048, out_features=1024, bias=True),
    torch.nn.Dropout(p=0.2),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=1024, out_features=512, bias=True),
    torch.nn.Dropout(p=0.2),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=512, out_features=len(cfg.labels), bias=True),
)
main_model.fc = posp
model = torch.nn.Sequential(prep, main_model)
model = model.cuda()

lrn = Learner("resnest50-vanilla-half-frozen", cfg, model)

lrn.learn()
