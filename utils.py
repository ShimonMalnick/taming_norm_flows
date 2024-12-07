import math
import logging
import os
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Union, List, Dict
import numpy as np
from torchvision.transforms import Normalize, Compose, Resize, ToTensor, RandomHorizontalFlip
from datasets import CelebAPartial
from model import Glow
import torch
from easydict import EasyDict
import json
import torchvision.datasets as vision_datasets
from torch.utils.data import DataLoader, Subset
from arcface_model import Backbone
from torchvision.models import resnet50
from setup import CELEBA_ROOT, BASE_MODEL_PATH, ARCFACE_CKPT
# Constants

CELEBA_NUM_IDENTITIES = 10177
CELEBA_MALE_ATTR_IDX = 20
CELEBA_GLASSES_ATTR_IDX = 15
TEST_IDENTITIES = [10015, 1614, 1624, 2261, 3751, 3928, 4244, 4941, 5423, 6002, 6280, 6648, 6928, 7124, 7271, 8677,
                   9039, 9192, 9697, 9787]
OUT_OF_TRAINING_IDENTITIES = [56, 114, 192, 209, 235, 349, 365, 468, 499, 510]
FAIRFACE_MODEL_INPUT_DIM = 224

TIME_PER_ITER_TAME = 4.73  # seconds
TIME_PER_ITER_TRAIN = 1.93  # seconds



def get_args(**kwargs) -> EasyDict:
    parser = ArgumentParser(description="Taming normalizing flows", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch", help="batch size", type=int)
    parser.add_argument("--iter", help="maximum iterations", type=int)
    parser.add_argument(
        "--n_flow", help="number of flows in each block", type=int
    )
    parser.add_argument("--n_block", help="number of blocks", type=int)
    parser.add_argument(
        "--no_lu",
        action="store_true",
        help="use plain convolution instead of LU decomposed version",
        default=None
    )
    parser.add_argument(
        "--affine", action="store_true", help="use affine coupling instead of additive", default=None
    )
    parser.add_argument("--n_bits", help="number of bits", type=int)
    parser.add_argument("--lr", help="learning rate", type=float)
    parser.add_argument("--img_size", help="image size", type=int)
    parser.add_argument("--temp", help="temperature of sampling", type=float)
    parser.add_argument("--n_sample", help="number of samples", type=int)
    parser.add_argument("--ckpt_path", help='Path to checkpoint for model', default=BASE_MODEL_PATH)
    parser.add_argument("--dir_name",
                        help='directory name (will be created under experiments), to store the current experiment')
    parser.add_argument("--opt_path", help='Path to checkpoint for optimizer')
    parser.add_argument("--path", metavar="PATH", help="Path to image directory")
    parser.add_argument('--eval', action='store_true', help='Use for evaluating a model', default=None)
    parser.add_argument('--sample_name', help='Name of sample size in case of evaluation')
    parser.add_argument('--exp_name', help='Name experiment for saving dirs')
    parser.add_argument('--num_workers', help='Number of worker threads for dataloader', type=int)
    parser.add_argument('--training_labels', help='if training is only on some labels, specify them here, comma seperated')
    parser.add_argument('--config', '-c',
                        help='Name of json config file (optional) cmd will be overriden by file option')
    parser.add_argument('--devices', help='number of gpu devices to use', type=int)
    parser.add_argument('--log_every', help='run heavy logs every <log_every> iterations', type=int)
    if kwargs.get('forget', False):
        if kwargs.get('forget_attribute', False):
            parser.add_argument('--forget_attribute', help='which attribute to forget', type=int)
            parser.add_argument('--forget_additional_attribute', help='additional attribute to forget', type=int)
            parser.add_argument('--debias', type=int,
                                help='if 1, instead of forgetting the attribute, forget the opposite')
            parser.add_argument('--cls_every', help='classify random latents every <cls_every> iterations', type=int)
            parser.add_argument('--cls_n_samples', help='number of latents to randomly select for classification',
                                type=int)
        else:
            parser.add_argument('--forget_identity', help='Identity to forget', type=int)
            parser.add_argument('--forget_size', help='Number of images to forget', type=int)

        parser.add_argument('--save_every', help='number of steps between model and optimizer saving periods', type=int)
        parser.add_argument('--data_split', help='optional for data split, one of [train, val, test, all]')
        parser.add_argument('--forget_thresh', help='Threshold on forgetting procedure. when BPD > mu + std * thresh,'
                                                    ' the finetuning is stopped.', type=float)
        parser.add_argument('--forget_lr', help='Learning rate for the forget optimizer', type=float)
        parser.add_argument('--gamma', help='proportion between remembering and forcing the distributions proximity',
                            type=float)
        parser.add_argument('--alpha', help='if given, training will be done with loss updates of both forget and'
                                            ' remember data in every update, by using'
                                            ' alpha * forget_loss + (1 - alpha) * remember_loss', type=float)
        parser.add_argument('--loss', help='which loss function to use', choices=['reverse_kl', 'forward_kl', 'both'])
        parser.add_argument('--forget_loss_baseline', help='if 1, whether to use the forget loss as minus the NLL loss '
                                                     '(i.e. reducing the likelihood)', type=int)
    args = parser.parse_args()
    out_dict = EasyDict()
    if args.config:
        with open(args.config, "r") as in_j:
            config_dict = json.load(in_j)
        out_dict.update(config_dict)
    args_dict = vars(args)
    for k in args_dict:
        if args_dict[k] is not None or k not in out_dict:
            out_dict[k] = args_dict[k]

    if out_dict.path and 'celeba' in out_dict.path.lower():
        out_dict.path = CELEBA_ROOT

    return EasyDict(out_dict)


def save_dict_as_json(save_dict, save_path):
    with open(save_path, 'w') as out_j:
        json.dump(save_dict, out_j, indent=4)


def get_base_model_args():
    args_path = os.path.join(os.path.dirname(BASE_MODEL_PATH), "args.json")
    with open(args_path, 'r') as in_j:
        args = EasyDict(json.load(in_j))
        args.ckpt_path = BASE_MODEL_PATH
    return EasyDict(args)


def get_dataset(data_root_path, image_size, **kwargs):
    if 'transform' in kwargs:
        transform = kwargs['transform']
    else:
        transform = Compose(
            [
                Resize((image_size, image_size)),
                # CenterCrop(image_size),
                RandomHorizontalFlip(),
                ToTensor()])
    if 'celeba' in data_root_path.lower():
        assert 'data_split' in kwargs, 'data_split must be specified for celeba'
        if kwargs['data_split']:
            split = kwargs['data_split']
        ds = vision_datasets.CelebA(data_root_path, split, transform=transform, download=False, target_type='identity')
    else:
        ds = vision_datasets.ImageFolder(data_root_path, transform=transform)
    return ds


def get_dataloader(data_root_path, batch_size, image_size, num_workers=8, dataset=None, **kwargs) -> DataLoader:
    if dataset is None:
        dataset = get_dataset(data_root_path, image_size, **kwargs)
    return DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)


def sample_data(data_root_path, batch_size, image_size, num_workers=8, dataset=None, **kwargs):
    if dataset is None:
        dataset = get_dataset(data_root_path, image_size, **kwargs)
    if 'training_labels' in kwargs and kwargs['training_labels']:
        training_labels = [int(label) for label in kwargs['training_labels'].split(',')]
        # find the labels in the training set
        indices = []
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if label in training_labels:
                indices.append(idx)
        dataset = Subset(dataset, indices)
    logging.info(f"loading Data Set of size: {len(dataset)}")
    loader = get_dataloader(data_root_path, batch_size, image_size, num_workers, dataset=dataset)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = get_dataloader(data_root_path, batch_size, image_size, num_workers, dataset=dataset)
            loader = iter(loader)
            yield next(loader)


def compute_dataset_bpd(n_bins, img_size, model, device, dataset, reduce=True) -> Union[float, torch.Tensor]:
    """
    Computation of bits per dimension as done in Glow, meaning we:
    - compute the negative log likelihood of the data
    - add to the log likelihood the dequantization term -Mlog(a), where M=num_pixels, a = 1/n_bins
    - divide by log(2) for change base of the log used in the nll
    - divide by num_pixels
    :param reduce: if true, return a float representing the bpd of the whole data set. if False, return the
    individual scores, i.e. reduced_bpd = sum(unreduced_bpd) / batch_size
    :param n_bins: number of bins the data is quantized into.
    :param device: the device to run the computation on.
    :param model: The model to use for computing the nll.
    :param img_size: the images size (square of img_sizeXimg_size)
    :param dataset: data set for the data. the data is expected to be qunatized to n_bins.
    :return: bits per dimension
    """
    nll = 0.0
    total_images = len(dataset)
    for i in range(total_images):
        x, _ = dataset[i]
        x = x.to(device).unsqueeze(0)
        x += torch.rand_like(x) / n_bins
        with torch.no_grad():
            log_p, logdet, _ = model(x)
            logdet = logdet.mean()
        if reduce:
            nll -= torch.sum(log_p + logdet).item()
        else:
            cur_nll = - (log_p + logdet)
            if i == 0:
                nll = cur_nll
            else:
                nll = torch.cat((nll, cur_nll))
    M = img_size * img_size * 3
    if reduce:
        nll /= total_images
    bpd = (nll + (M * math.log(n_bins))) / (math.log(2) * M)
    return bpd


def compute_dataloader_bpd(n_bins, img_size, model, device, data_loader: DataLoader, reduce=True) \
        -> Union[float, torch.Tensor]:
    """
    Computation of bits per dimension as done in Glow, meaning we:
    - compute the negative log likelihood of the data
    - add to the log likelihood the dequantization term -Mlog(a), where M=num_pixels, a = 1/n_bins
    - divide by log(2) for change base of the log used in the nll
    - divide by num_pixels
    :param device: the device to run the computation on.
    :param model: The model to use for computing the nll.
    :param img_size: the images size (square of img_sizeXimg_size)
    :param n_bins: number of bins the data is quantized into.
    :param data_loader: data loader for the data. the data is expected to be qunatized to n_bins.
    :return: bits per dimension
    """
    nll = 0.0
    total_images = 0
    for idx, batch in enumerate(data_loader, 1):
        x, _ = batch
        x = x.to(device)
        x += torch.rand_like(x) / n_bins
        with torch.no_grad():
            log_p, logdet, _ = model(x)
            logdet = logdet.mean()
        if reduce:
            nll -= torch.sum(log_p + logdet).item()
        else:
            cur_nll = - (log_p + logdet)
            if total_images == 0:
                nll = cur_nll.detach().cpu()
            else:
                nll = torch.cat((nll, cur_nll.detach().cpu()))
        total_images += x.shape[0]
        logging.debug(f"finished batch: {idx}/{len(data_loader)}")
    M = img_size * img_size * 3
    if reduce:
        nll /= total_images
    bpd = (nll + (M * math.log(n_bins))) / (math.log(2) * M)
    return bpd


def load_model(args, device=None, training=False, device_ids=None, output_device=None) -> Union[
    Glow, torch.nn.DataParallel]:
    model = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)
    state_dict = torch.load(args.ckpt_path, map_location=lambda storage, loc: storage)
    first_key = list(state_dict.keys())[0]
    if first_key.startswith("module."):  # for network saved as Dataparallel accidentally
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if not training:
        logging.debug("device: {}".format(device))
        model = model.to(device)
        logging.debug("Loaded model on single device successfully")
        return model
    else:
        assert device is not None or device_ids is not None
        if device_ids is not None:
            device = device_ids[0]
        model_parallel = torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device)
        model_parallel = model_parallel.to(device)
        return model_parallel


def quantize_image(img, n_bits):
    """
    assuming the input is in [0, 1] of 8 bit images for each channel
    """
    if n_bits < 8:
        img = img * 255
        img = torch.floor(img / (2 ** (8 - n_bits)))
        img /= (2 ** n_bits)
    return img - 0.5


def save_model_optimizer(args, iter_num, model, optimizer, save_prefix='experiments', last=False,
                         save_optim=True, save_one=False) -> None:
    if last:
        model_id = "last"
    else:
        model_id = str(iter_num + 1).zfill(6)
    if save_one:
        model_id = ""
    torch.save(
        model.state_dict(), f'{save_prefix}/{args.exp_name}/checkpoints/model_{model_id}.pt'
    )
    with open(f'{save_prefix}/{args.exp_name}/checkpoints/iter_num.txt', 'a+') as f:
        f.write(f"{iter_num + 1}\n")
    if save_optim:
        torch.save(
            optimizer.state_dict(), f'{save_prefix}/{args.exp_name}/checkpoints/optim_{model_id}.pt'
        )


def gather_jsons(in_paths, keys_names, out_path, add_duplicate_names=False) -> None:
    d = {}
    for p, key_name in zip(in_paths, keys_names):
        with open(p, 'r') as in_j:
            data = json.load(in_j)
        if add_duplicate_names and key_name in d:
            key_name += '_addition'

        d[key_name] = data

    with open(out_path, 'w') as out_j:
        json.dump(d, out_j, indent=4)


def mean_float_jsons(jsons: List[str], save_path: str) -> dict:
    out = {}
    for json_f in jsons:
        with open(json_f, "r") as f:
            input_dict = json.load(f)
        for k in input_dict:
            if isinstance(input_dict[k], dict):
                if k not in out:
                    out[k] = {}
                for k2 in input_dict[k]:
                    if type(input_dict[k][k2]) == float:
                        if k2 not in out[k]:
                            out[k][k2] = []
                        out[k][k2].append(input_dict[k][k2])
                    else:
                        raise ValueError(
                            f"input_dict[{k}][{k2}] is not a float. currently supporting only jsons with 1 level of nesting that includes floats only")
            elif type(input_dict[k]) == float:
                if k not in out:
                    out[k] = []
                out[k].append(input_dict[k])
            else:
                raise ValueError(
                    f"input_dict[{k}] is not a float. currently supporting only jsons with 1 level of nesting that includes floats only")
    for k in out:
        if isinstance(out[k], dict):
            for k2 in out[k]:
                out[k][k2] = sum(out[k][k2]) / len(out[k][k2])
        else:
            out[k] = sum(out[k]) / len(out[k])
    if save_path:
        save_dict_as_json(out, save_path)
    return out


def load_arcface(device=None) -> Backbone:
    model = Backbone(50, 0.6, 'ir_se').to('cpu')
    model.requires_grad_(False)
    cls_ckpt = ARCFACE_CKPT
    model.load_state_dict(torch.load(cls_ckpt, map_location='cpu'))
    if device:
        model = model.to(device)
    return model.eval()


def load_arcface_transform():
    return Compose([Resize((112, 112)),
                    ToTensor(),
                    Normalize((0.5, 0.5, 0.5),
                              (0.5, 0.5,
                               0.5))])


def compute_cosine_similarity(e1, e2, mean=False) -> torch.Tensor:
    """
    Compute cosine similarity matrix (or reduced mean) of two arcface embedding matrices
    :param e1: First embedding's matrix of shape (N_1, E)
    :param e2: Second embedding's matrix of shape (N_2, E)
    :param mean: if true, return a mean of the outputs matrix, i.e. a scalar
    :return: matrix E = e_1 @ e_2.T of shape (N_1, N_2)
    """
    e1 = e1
    e2 = e2
    e = e1 @ e2.T
    if mean:
        return e.mean()
    return e


def get_num_params(model: torch.nn.Module) -> int:
    """
    Returns the number of trainable parameters in the given model
    :param model: torch model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_resnet_for_binary_cls(pretrained=True, all_layers_trainable=True, device=None, num_outputs=1) -> resnet50:
    """
    Loads a pretrained resenet50 apart from the last classification layer (fc) that is changed to a new untrained layer
    with output size 1
    :param num_outputs: number of outputs of the new last layer
    :param pretrained: whether to load the pretrained weights or not
    :param device: device to load the model to
    :param all_layers_trainable: if True, all layers are trainable, otherwise only the last layer is trainable
    :return:
    """
    model = resnet50(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = all_layers_trainable
    model.fc = torch.nn.Linear(2048, num_outputs)  # 2048 is the number of activations before the fc layer
    if device:
        model = model.to(device)
    return model


def get_resnet_50_normalization():
    """
    This is the normalization according to:
    https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights
    :return:
    """
    return Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def get_partial_dataset(transform, target_type='identity', **kwargs) -> CelebAPartial:
    """
    Used for datasets when performing censoring/forgetting. this function is used both to obtain a dataset containing
    only some identities/images, or to obtain a dataset containing all iamges in celeba apart from certain
    identities/images. See documnetaion of CelebAPArtial for more details.
    """
    ds = CelebAPartial(root=CELEBA_ROOT, transform=transform, target_type=target_type, **kwargs)
    logging.info(f"len of dataset: {len(ds)}")
    return ds


def get_default_forget_transform(img_size, n_bits) -> Compose:
    transform = Compose([Resize((img_size, img_size)),
                         ToTensor(),
                         lambda img: quantize_image(img, n_bits)])

    return transform


def forward_kl_univariate_gaussians(mu_p, sigma_p, mu_q, sigma_q):
    """
    Compute forward KL(P || Q) given parameters of univariate gaussian distributions, where P is the real distribution
    and Q is the approximated learned one.
    """
    return math.log(sigma_q / sigma_p) + ((sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2)) - 0.5


def reverse_kl_univariate_gaussians(mu_p, sigma_p, mu_q, sigma_q):
    """
    Compute forward KL(Q || P) given parameters of univariate gaussian distributions, where P is the real distribution
    and Q is the approximated learned one.
        """
    return math.log(sigma_p / sigma_q) + ((sigma_q ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_p ** 2)) - 0.5


def np_gaussian_pdf(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def data_parallel2normal_state_dict(dict_path: str, out_path: str):
    state_dict = torch.load(dict_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    torch.save(new_state_dict, out_path)


def identity2median_likelihood_images(identity, images_paths, num_images) -> List[str]:
    assert num_images <= 15
    if num_images > 1:
        return [images_paths[i] for i in range(num_images)]
    else:
        median_range = [0.41, 0.59]
        with open("data/test_identities_quantiles.json") as quantiles_json:
            identity_quantiles = json.load(quantiles_json)[str(identity)]
        for i in range(min(len(identity_quantiles), 15)):
            if median_range[0] <= identity_quantiles[i] <= median_range[1]:
                return [images_paths[i]]
    raise ValueError(f"No median image for given identity: {identity}")


def args2dset_params(args, ds_type, from_index=True, index_json=None) -> Dict:
    """
    Returns a dictionary of parameters to be passed to the dataset constructor.
    :param args: arguments determining the images/identities to forget/remember.
    :param ds_type: one of 'forget' or 'remember'
    :return: dictionary contating the parameters to be passed to the dataset constructor
    """
    assert ds_type in ['forget', 'remember', 'forget_ref'], \
        "ds_type must be one of 'forget' or 'remember' or 'forget_ref'"

    out = {"include_only_identities": None,
           "exclude_identities": None,
           "include_only_images": None,
           "exclude_images": None}
    if 'data_split' in args and args.data_split in ['train', 'all', 'valid']:
        out['split'] = args.data_split
    identity = args.forget_identity
    assert int(identity) in TEST_IDENTITIES + OUT_OF_TRAINING_IDENTITIES, "Identity not in test set identities"
    if int(identity) in OUT_OF_TRAINING_IDENTITIES:
        out['split'] = 'valid'
    assert args.forget_size, "must specify forget_size"
    max_forget_images = 15
    if index_json is None:
        with open("data/identities_index.json") as f:
            index = json.load(f)
    else:
        index = index_json
    identity_images_names = index[str(identity)]
    if ds_type == 'forget':
        out["include_only_images"] = identity2median_likelihood_images(identity, identity_images_names[:max_forget_images], args.forget_size)
    elif ds_type == 'remember':
        out["exclude_images"] = identity_images_names[max_forget_images:]
    elif ds_type == 'forget_ref':
        images_paths = identity_images_names[max_forget_images:]
        out["include_only_images"] = images_paths

    return out


def args2dataset(args, ds_type, transform):
    ds_params = args2dset_params(args, ds_type)
    ds = get_partial_dataset(transform=transform, **ds_params)
    return ds


def nll_to_sigma_normalized(nll: Union[torch.Tensor, float],
                            mean: Union[torch.Tensor, float],
                            std: Union[torch.Tensor, float], return_torch=True) -> Union[
    float, torch.Tensor, List[float]]:
    """
    """
    ret = (nll - mean) / std
    if return_torch:
        return ret
    else:
        return ret.item() if ret.nelement() == 1 else ret.tolist()


def set_all_seeds(seed=37):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
