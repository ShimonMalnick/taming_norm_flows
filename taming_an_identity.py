import math
import os
from math import log
from easydict import EasyDict
from model import Glow
import torch
import torch.nn.functional as F
from utils import get_args, load_model, save_dict_as_json, save_model_optimizer, compute_dataloader_bpd,\
    get_default_forget_transform, args2dataset, set_all_seeds, \
    forward_kl_univariate_gaussians, reverse_kl_univariate_gaussians
from torch.utils.data import DataLoader, Subset, Dataset
from easydict import EasyDict as edict
from typing import Iterator, Dict, Union, List, Tuple, Callable
from datasets import CelebAPartial, EqualProbSampler
# from evaluate import full_experiment_evaluation
import logging


def calc_regular_loss(log_p, logdet, image_size, n_bins, weights=None):
    """
    Calculate the loss as in normalizing flows, i.e. minimizing the negative loglikelihood which is log_p + logdet.
    The loss is normalized to BPD (bits per dimension). Weights are optional for wegihted loss.
    :param log_p: log probability of the model on inputs x.
    :param logdet: log determinant of the Jacobian of the flow.
    :param image_size: the size of the image, used to calculate the number of pixels.
    :param n_bins: qunatization level of the data
    :param weights: weights to apply to the loss, if None, no weights are applied.
    :return: (loss, average log_p, average logdet) in BPD units.
    """
    n_pixel = image_size * image_size * 3
    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    if weights is not None:
        assert weights.shape == loss.shape, "weights and loss must have the same shape, but got weights shape {} " \
                                            "and loss shape {}".format(weights.shape, loss.shape)
        loss = (-(loss * weights) / (log(2) * n_pixel)).sum()  # summing because weights are normalized to sum to 1
    else:
        loss = (-loss / (log(2) * n_pixel)).mean()
    return (
        loss,
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def make_forget_exp_dir(exp_name, exist_ok=False, dir_name="forget") -> str:
    """
    Creates a directory for the forget experiment under experiments/{dir_name}
    :param exp_name: desired name of the experiment.
    :param exist_ok: whether to raise an error if the directory already exists.
    :param dir_name: name of the base directory to create the experiment directory in.
    :return: path to the experiment directory.
    """
    base_path = os.path.join("experiments", dir_name, exp_name)
    os.makedirs(f'{base_path}/checkpoints', exist_ok=exist_ok)
    os.makedirs(f'{base_path}/logs', exist_ok=exist_ok)
    return os.path.join(dir_name, exp_name)


def get_data_iterator(ds: CelebAPartial, batch_size, num_workers=16) -> Iterator:
    """
    Creates an infinite data iterator for the given dataset.
    :param ds: the given dataset.
    :param batch_size: the batch size for the iterator.
    :param num_workers: num workers for the data loader.
    :return: an iterator over the dataset.
    """
    sampler = None
    shuffle = True
    if len(ds) < batch_size:
        sampler = EqualProbSampler(ds, batch_size)
        shuffle = None
    dl_iter = iter(DataLoader(ds, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, sampler=sampler))

    while True:
        try:
            yield next(dl_iter)

        except StopIteration:
            dl_iter = iter(
                DataLoader(ds, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, sampler=sampler))
            yield next(dl_iter)


def calc_batch_bpd(args, model, batch, reduce=True) -> Union[float, torch.Tensor]:
    """
    Calculates the batch-wise BPD of the model.
    :param reduce: if reduce is true, return a sclar value that is the mean of the batch-wise BPDs, if it is false,
    return the per example contriubtion to the BPD, meaning reduced_bpd = sum(unreduced_bpd) / batch_size
    :param args: arguments relevant to the model's parameters and input image size.
    :param model: model to calculate the BPD with.
    :param batch: batch to calculate the BPD of.
    :return: batch-wise BPD of the model.
    """
    n_bins = 2 ** args.n_bits
    M = args.img_size * args.img_size * 3
    with torch.no_grad():
        log_p, logdet, _ = model(batch + torch.rand_like(batch) / n_bins)
    if reduce:
        cur_nll = - torch.sum(log_p + logdet.mean()).item()
    else:
        cur_nll = - (log_p + logdet.mean())

    cur_bpd = (cur_nll + (M * math.log(n_bins))) / (math.log(2) * M)
    if reduce:
        cur_bpd /= batch.shape[0]
    return cur_bpd


def prob2bpd(prob, n_bins, n_pixels):
    """
    Converts a probability to BPD.
    :param prob: the probability to convert.
    :param n_bins: the qunaitzation level of the data.
    :param n_pixels: the number of pixels in the image.
    :return: the BPD of the probability.
    """
    return - prob / (math.log(2) * n_pixels) + math.log(n_bins) / math.log(2)


def get_log_p_parameters(n_bins, n_pixel, dist, device=None):
    """
    Calculates and returns the mean and std of a given probability distribution, normalized to BPD.
    :param n_bins: the qunaitzation level of the data.
    :param n_pixel: the number of pixels in the image.
    :param dist: the probability distribution to calculate the mean and std of.
    :param device: the device to put the mean and std on (optional).
    :return:mean, std of the distribution in BPD.
    """
    val = -log(n_bins) * n_pixel
    val += dist
    val = -val / (log(2) * n_pixel)

    mean, std = torch.mean(val), torch.std(val)
    if device is not None:
        return mean.to(device), std.to(device)
    return mean, std


def forget_alpha(args: edict, remember_iter: Iterator, forget_ds: Dataset, model: Union[Glow, torch.nn.DataParallel],
                 original_model: Glow,
                 training_devices: List[int],
                 original_model_device: torch.device,
                 optimizer: torch.optim.Optimizer,
                 remember_ds: Dataset,
                 forget_eval_data: Tuple[torch.Tensor, Dict]) -> Union[Glow, torch.nn.DataParallel]:
    """
    Performs the forget experiment with the given parameters.
    :param args: the arguments for the experiment.
    :param remember_iter: the iterator over the remember dataset.
    :param forget_ds: the dataset to forget.
    :param model: the model to forget with.
    :param original_model: the original model to compare the forgetting model to.
    :param training_devices: the devices to train the model on.
    :param original_model_device: the device to put the original model on.
    :param optimizer: the optimizer to use for the forget model.
    :param remember_ds: the dataset to remember.
    :param forget_eval_data: the data to evaluate the forget model on.
    :return: the forget model.
    """
    kl_loss_fn = get_kl_loss_fn(args.loss)
    main_device = torch.device(f"cuda:{training_devices[0]}")
    all_forget_images = torch.stack([x[0] for x in forget_ds]).to(main_device)
    n_bins = 2 ** args.n_bits
    n_pixels = args.img_size * args.img_size * 3
    avg_time = 0
    for i in range(args.iter):
        if args.forget_loss_baseline:
            log_p, log_det, _ = model(all_forget_images + torch.rand_like(all_forget_images) / n_bins)
            log_det = log_det.mean()
            forget_loss, _, _ = calc_regular_loss(log_p, log_det, args.img_size, n_bins,
                                                                     weights=None)
            forget_loss *= -1
            with torch.no_grad():
                distances = get_forget_distance_loss(n_bins, n_pixels, args.eval_mu, args.eval_std, args.forget_thresh,
                                                   all_forget_images, args.batch, model)
                if distances is None:
                    logging.info("breaking after {} iterations".format(i))
                    break
        else:
            forget_loss = get_forget_distance_loss(n_bins, n_pixels, args.eval_mu, args.eval_std, args.forget_thresh,
                                                   all_forget_images, args.batch, model)
        if forget_loss is None:
            logging.info("breaking after {} iterations".format(i))
            break

        remember_batch = next(remember_iter)[0].to(main_device)
        remember_batch += torch.rand_like(remember_batch) / n_bins
        with torch.no_grad():
            orig_p, orig_det, _ = original_model(remember_batch.to(original_model_device))
            orig_dist = orig_p + orig_det.mean()
            orig_mean, orig_std = get_log_p_parameters(n_bins, n_pixels, orig_dist, device=main_device)

        _, remember_loss = get_kl_and_remember_loss(args, kl_loss_fn, model, n_bins, n_pixels, orig_mean,
                                                          orig_std, remember_batch)

        loss = args.alpha * forget_loss + (1 - args.alpha) * remember_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        compute_step_stats(args, i, model, main_device, remember_ds, forget_eval_data)

        logging.info(f"Iter: {i + 1} Forget Loss: {forget_loss.item():.5f}; Remember Loss: {remember_loss.item():.5f}")

        if args.save_every is not None and (i + 1) % args.save_every == 0:
            save_model_optimizer(args, i, model.module, optimizer, save_optim=False)
    if args.save_every is not None:
        with open(os.path.join("experiments", args.exp_name, "timing.txt"), "a+") as f:
            f.write(f"Total avg time per iter[seconds] (timing mode={args.timing}): {avg_time}\n")
        save_model_optimizer(args, 0, model.module, optimizer, last=True, save_optim=False)

    return model


def get_kl_and_remember_loss(args, kl_loss_fn, model, n_bins, n_pixels, orig_mean, orig_std, remember_batch):
    """
    Calculates and return the KL loss and the remember loss.
    :param args:  the arguments for the experiment.
    :param kl_loss_fn:  the KL loss function to use.
    :param model:  the model to calculate the loss for.
    :param n_bins:  the qunaitzation level of the data.
    :param n_pixels:  the number of pixels in the image.
    :param orig_mean:  the mean of the original model.
    :param orig_std:  the std of the original model.
    :param remember_batch:  the batch to calculate the remember loss for.
    :return:  the KL loss and the remember loss.
    """
    remember_p, remember_det, _ = model(remember_batch)
    remember_det = remember_det.mean()
    regular_loss, _, _ = calc_regular_loss(remember_p, remember_det, args.img_size, n_bins,
                                                             weights=None)
    remember_dist = remember_p + remember_det
    remember_mean, remember_std = get_log_p_parameters(n_bins, n_pixels, remember_dist)
    kl_loss = kl_loss_fn(orig_mean, orig_std, remember_mean, remember_std)
    remember_loss = args.gamma * kl_loss + (1 - args.gamma) * regular_loss
    return kl_loss, remember_loss


def args2data_iter(args, ds_type, transform) -> Iterator:
    """
    Returns a data iterator for the dataset specified by ds_type.
    :param ds_len:
    :param transform: transform to be applied to the dataset.
    :param args: arguments determining the images/identities to forget/remember.
    :param ds_type: one of 'forget' or 'remember'
    :return: data iterator for the dataset specified by ds_type
    """
    ds = args2dataset(args, ds_type, transform)
    return get_data_iterator(ds, args.batch, num_workers=args.num_workers)


def compute_step_stats(args: EasyDict, step: int, model: torch.nn.Module, device, ds: Dataset,
                       forget_data: Tuple[torch.Tensor, Dict]) -> torch.Tensor:
    model.eval()
    forget_batch, forget_dict = forget_data
    forget_bpd = calc_batch_bpd(args, model, forget_batch, reduce=False).cpu()
    if not args.timing:
        forget_data = forget_bpd.view(-1).tolist()
        forget_dict["data"].append([step] + forget_data)
    cur_indices = torch.randperm(len(ds))[:1024]
    cur_ds = Subset(ds, cur_indices)
    cur_dl = DataLoader(cur_ds, batch_size=256, shuffle=False, num_workers=args.num_workers)
    if (step + 1) % args.log_every == 0:
        eval_bpd = compute_dataloader_bpd(2 ** args.n_bits, args.img_size, model, device, cur_dl, reduce=False).cpu()
        args.eval_mu = eval_bpd.mean().item()
        args.eval_std = eval_bpd.std().item()
        if not args.timing:
            logging.info(f"eval_mu: {args.eval_mu}, eval_std: {args.eval_std} for iteration {step}")

    model.train()

    return forget_bpd


def get_random_batch(ds: Dataset, batch_size, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randperm(len(ds))[:batch_size]
    batch = [ds[idx][0] for idx in indices]
    batch = torch.stack(batch).to(device)
    return batch, indices


def get_kl_loss_fn(loss_type: str) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Returns a KL divergence loss between two univariate Gaussians. Denoting P as the distribution we wish to approximate
    and Q as the distribution we use to approximate P, the loss is defined as:
    forward_kl: KL(P || Q)
    reverse_kl: KL(Q || P)
    both: KL(P || Q) + KL(Q || P)
    :param loss_type: a str containing the name of the loss type
    :return: a callable function that computes the loss. the function receives 4 parameters: mu_p, std_p, mu_q, std_q
    """
    if loss_type == 'forward_kl':
        return forward_kl_univariate_gaussians
    elif loss_type == 'reverse_kl':
        return reverse_kl_univariate_gaussians
    elif loss_type == 'both':
        return lambda mu_p, sigma_p, mu_q, sigma_q: \
            forward_kl_univariate_gaussians(mu_p, sigma_p, mu_q, sigma_q) + \
            reverse_kl_univariate_gaussians(mu_p, sigma_p, mu_q, sigma_q)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def get_forget_distance_loss(n_bins: int,
                             n_pixels: int,
                             mean: float,
                             std: float,
                             thresh: float,
                             forget_images: torch.Tensor,
                             batch_size: int,
                             model,
                             eps: float = None) -> Union[torch.Tensor, None]:
    """
    Returns the next batch of images to forget, along with the proportional weights for each example. the samples are
    drawn from the forget_images tensor, and the weights are computed according to the distance from the forget
    threshold,i.e. images that are further from the threshold get a higher weight (and are more likely to be forgotten).
    images above the threshold are filtered out. the final batch is samples (with replacement if needed) from the
    images under the threshold. In case there are none of those, the function returns None (and the forget process can
    be stopped).
    :param eps:
    :param thresh:
    :param std:
    :param mean:
    :param n_pixels:
    :param n_bins:
    :param forget_images: all the images to forget, assuming with no grad they can pass in one batch through the model.
     Images neeed to be on the same device as the model.
    :param batch_size:
    :param model:
    :return: None if no image needs to be forgotten, else a tuple of images to forget and corresponding weights.
    """
    if eps is None:
        eps = 0.15 * thresh
    with torch.no_grad():
        break_distance = compute_distance(forget_images, mean, model, n_bins, n_pixels, std, thresh)
        indices = torch.nonzero(torch.abs(break_distance) > (eps * std), as_tuple=True)[0]
        if indices.nelement() == 0:
            # means that all images are above the threshold
            return None
    sampling_indices = torch.randperm(forget_images.shape[0])[:batch_size]
    distance = compute_distance(forget_images[sampling_indices], mean, model, n_bins, n_pixels, std, thresh)
    loss = F.sigmoid(distance ** 2)
    return loss.mean()


def compute_distance(forget_images, mean, model, n_bins, n_pixels, std, thresh):
    log_p, logdet, _ = model(forget_images + torch.rand_like(forget_images) / n_bins)
    logdet = logdet.mean()
    cur_bpd = prob2bpd(log_p + logdet, n_bins, n_pixels)
    distance = (cur_bpd - (mean + std * thresh))
    return distance


def main():
    set_all_seeds(seed=37)
    logging.getLogger().setLevel(logging.INFO)
    args = get_args(forget=True)
    args.timing = False
    all_devices = list(range(torch.cuda.device_count()))
    train_devices = all_devices[:-1] if len(all_devices) > 1 else all_devices
    original_model_device = torch.device(f"cuda:{all_devices[-1]}")
    dir_name = "forget_identity_main" if not args.dir_name else args.dir_name
    args.exp_name = make_forget_exp_dir(args.exp_name, exist_ok=False, dir_name=dir_name)
    logging.info(args)
    model: torch.nn.DataParallel = load_model(args, training=True, device_ids=train_devices,
                                              output_device=train_devices[0])
    original_model: Glow = load_model(args, device=original_model_device, training=False)
    original_model.requires_grad_(False)
    transform = get_default_forget_transform(args.img_size, args.n_bits)
    forget_ds = args2dataset(args, "forget", transform)
    forget_ref_batch = torch.stack([forget_ds[idx][0] for idx in range(len(forget_ds))])
    forget_ref_data = (forget_ref_batch, {"columns": ["step"] + [f"idx {i}" for i in range(len(forget_ds))],
                                          "data": []})
    forget_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    remember_ds = args2dataset(args, ds_type='remember', transform=transform)
    if args.data_split == 'valid':
        # This id for  forgetting and remembering out of the training set, using the validation set,
        # and limiting the remember set to have a size of 1000
        remember_ds = torch.utils.data.Subset(remember_ds, range(1000))
    args["remember_ds_len"] = len(remember_ds)
    args["forget_ds_len"] = len(forget_ds)
    save_dict_as_json(args, f'experiments/{args.exp_name}/args.json')

    compute_step_stats(args, args.log_every - 1, model, None, remember_ds, forget_ref_data)
    remember_iter = get_data_iterator(remember_ds, args.batch, args.num_workers)
    logging.info("Starting forget alpha procedure")
    finetuned_model = forget_alpha(args, remember_iter, forget_ds, model,
                                   original_model, train_devices, original_model_device,
                                   forget_optimizer, remember_ds,
                                   forget_ref_data)
    # full_experiment_evaluation(f"experiments/{args.exp_name}", args, partial=10000, model=finetuned_model)


if __name__ == '__main__':
    main()
