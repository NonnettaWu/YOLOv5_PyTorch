## YOLOv5_PyTorch

Similar to YOLOv4.

With Google Colab provides a plant for training, the repository was creted to use `YOLOv5` model to detect objects. The most codes are copied from `Bubbliiiing`. 

His repository: https://github.com/bubbliiiing

In the repository, you can choose BackBone from `CSPdarknet`, `ConvNext` and `SwinTransformer`. And load their own pre_weights for training. After setting SwinTransformer as BackBone,  the problem that pre_weights loaded failedly was fixed.

```
pretrained_dict = {k.replace('layers.', 'backbone.layers.'): v for k, v in torch.load(model_path).items()}
pretrained_dict = {k.replace('patch_embed.', 'backbone.patch_embed.'): v for k, v in pretrained_dict.items()}
```

For contrasting with other models, imageSize is set to [3, 640, 640]. Besides, pre_weights have been used for training.You can do lots of settings in this Model with detailed exegesises.

The running environments has been supported in requirements.txt.

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple -y
```
 `YOLOv5 Structure` is below:

<img src="https://user-images.githubusercontent.com/86788385/208301572-3b5c6b1d-cbd7-418a-ae12-a681f0fa5776.png" width="800">

**Keep Learning!**
