import torch
from lightning import Trainer
from lightning.pytorch.callbacks import GradientAccumulationScheduler, LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner import Tuner

from models import VQVAE
from models.mar import MAR
from utils.data import DataModule


def train():
    V = 4096
    Cvae = 32
    ch = 160
    share_quant_resi = 4
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae_local = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi,
                      v_patch_nums=patch_nums)
    depth = 16
    shared_aln = False
    attn_l2_norm = True
    flash_if_available = True
    fused_if_available = True
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth / 24
    mar = MAR(
        vae_local=vae_local,
        num_classes=1000, depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
    )
    '''

    vae_local = VQVAE()
    mar = MAR()
    '''
    vae_ckpt = './weights/vae_ch160v4096z32.pth'
    # TODO 将vae更改为pl——module
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    data = DataModule()
    # accumulator = GradientAccumulationScheduler(scheduling={0: 8, 4: 4, 8: 1})
    # TODO 更改Checkpoint保存策略
    checkpoint_callback = ModelCheckpoint(dirpath="./out",
                                          save_top_k=2,
                                          monitor="val_loss",
                                          save_last=False,
                                          every_n_epochs=50,
                                          save_on_train_epoch_end=True,
                                          filename="mar-{epoch:04d}")
    # 如果使用混合精度，则不需要更改渐变，因为渐变未缩放 在应用剪裁功能之前

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
                      # precision="16-mixed",
                      accelerator="gpu",
                      devices=1,
                      callbacks=[checkpoint_callback, lr_monitor],
                      accumulate_grad_batches=1,
                      gradient_clip_val=0.5,
                      gradient_clip_algorithm="value",
                      max_epochs=500,
                      default_root_dir="./out")

    # tuner = Tuner(trainer)
    # 自动查找学习率
    # tuner.lr_find(mar, datamodule=data)
    # 自动查找batch—size
    # tuner.scale_batch_size(mar, mode="power", datamodule=data)

    # trainer.fit(model=mar, ckpt_path='', datamodule=data)
    trainer.fit(model=mar, datamodule=data)


if __name__ == '__main__':
    train()
