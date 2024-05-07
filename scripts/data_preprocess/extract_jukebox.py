"""Extraction methods here."""
import argparse
import os
from glob import glob
from os.path import join, basename

import librosa as lr
import torch
import torch as t
import gc
import numpy as np
import sys

__all__ = ["load_audio", "extract"]

import wget
from accelerate import init_empty_weights
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.make_models import MODELS, make_vqvae, make_prior
from torch import nn
from tqdm import tqdm

JUKEBOX_SAMPLE_RATE = 16000
T = 8192  # max token of prior
CTX_WINDOW_LENGTH = 1048576
DEFAULT_DURATION = CTX_WINDOW_LENGTH / JUKEBOX_SAMPLE_RATE
REMOTE_PREFIX = "https://openaipublic.azureedge.net/jukebox/models/5b/"
VQVAE_RATE = T / DEFAULT_DURATION
CACHE_DIR = 'checkpoints/jukebox-5b-lyric'


def set_module_tensor_to_device(
        module: nn.Module, tensor_name: str, device, value=None
):
    """
    A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function).
    Args:
        module (`torch.nn.Module`): The module in which the tensor we want to move lives.
        param_name (`str`): The full name of the parameter/buffer.
        device (`int`, `str` or `torch.device`): The device on which to set the tensor.
        value (`torch.Tensor`, *optional*): The value of the tensor (useful when going from the meta device to any
            other device).
    """
    # Recurse if needed
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(
            f"{module} does not have a parameter or a buffer named {tensor_name}."
        )
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)

    if (
            old_value.device == torch.device("meta")
            and device not in ["meta", torch.device("meta")]
            and value is None
    ):
        raise ValueError(
            f"{tensor_name} is on the meta device, we need a `value` to put in on {device}."
        )

    with torch.no_grad():
        if value is None:
            new_value = old_value.to(device)
        elif isinstance(value, torch.Tensor):
            new_value = value.to(device)
        else:
            new_value = torch.tensor(value, device=device)

        if is_buffer:
            module._buffers[tensor_name] = new_value
        elif (
                value is not None
                or torch.device(device) != module._parameters[tensor_name].device
        ):
            param_cls = type(module._parameters[tensor_name])
            kwargs = module._parameters[tensor_name].__dict__
            new_value = param_cls(
                new_value, requires_grad=old_value.requires_grad, **kwargs
            ).to(device)
            module._parameters[tensor_name] = new_value


def get_checkpoint(local_path, remote_prefix):
    if not os.path.exists(local_path):
        remote_path = remote_prefix + local_path.split("/")[-1]

        # create this bar_progress method which is invoked automatically from wget
        def bar_progress(current, total, width=80):
            progress_message = "Downloading: %d%% [%d / %d] bytes" % (
                current / total * 100,
                current,
                total,
            )
            # Don't use print() as it will print in new line every time.
            sys.stdout.write("\r" + progress_message)
            sys.stdout.flush()

        wget.download(remote_path, local_path, bar=bar_progress)


def load_weights(model, weights_path, device):
    model_weights = torch.load(weights_path, map_location="cpu")

    # load_state_dict, basically
    for k in tqdm(model_weights["model"].keys()):
        set_module_tensor_to_device(model, k, device, value=model_weights["model"][k])

    model.to(device)

    del model_weights


def setup_models(cache_dir=None, remote_prefix=None, device=None, verbose=True):
    global VQVAE, TOP_PRIOR

    if cache_dir is None:
        cache_dir = CACHE_DIR

    if remote_prefix is None:
        remote_prefix = REMOTE_PREFIX

    if device is None:
        device = 'cuda'

    # caching preliminaries
    vqvae_cache_path = cache_dir + "/vqvae.pth.tar"
    prior_cache_path = cache_dir + "/prior_level_2.pth.tar"
    os.makedirs(cache_dir, exist_ok=True)

    # get the checkpoints downloaded if they haven't been already
    get_checkpoint(vqvae_cache_path, remote_prefix)
    get_checkpoint(prior_cache_path, remote_prefix)

    if verbose:
        print("Importing jukebox and associated packages...")

    # Set up VQVAE
    if verbose:
        print("Setting up the VQ-VAE...")

    model = "5b"
    hps = Hyperparams()
    hps.sr = JUKEBOX_SAMPLE_RATE
    hps.n_samples = 3 if model == "5b_lyrics" else 8
    hps.name = "samples"
    chunk_size = 16 if model == "5b_lyrics" else 32
    max_batch_size = 3 if model == "5b_lyrics" else 16
    hps.levels = 3
    hps.hop_fraction = [0.5, 0.5, 0.125]
    VQVAE, *priors = MODELS[model]

    hparams = setup_hparams(VQVAE, dict(sample_length=1048576))

    # don't actually load any weights in yet,
    # leave it for later. memory optimization
    with init_empty_weights():
        VQVAE = make_vqvae(hparams, "meta")

    # Set up language model
    if verbose:
        print("Setting up the top prior...")
    hparams = setup_hparams(priors[-1], dict())
    hparams.n_ctx=8192
    # don't actually load any weights in yet,
    # leave it for later. memory optimization
    with init_empty_weights():
        TOP_PRIOR = make_prior(hparams, VQVAE, "meta")

    # flips a bit that tells the model to return activations
    # instead of projecting to tokens and getting loss for
    # forward pass
    TOP_PRIOR.prior.only_encode = True

    if verbose:
        print("Loading the top prior weights into memory...")

    load_weights(TOP_PRIOR, prior_cache_path, device)

    gc.collect()
    torch.cuda.empty_cache()

    load_weights(VQVAE, vqvae_cache_path, device)

    return VQVAE, TOP_PRIOR


def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


def load_audio(fpath, offset=0.0, duration=None):
    if duration is not None:
        audio, _ = lr.load(
            fpath, sr=JUKEBOX_SAMPLE_RATE, offset=offset, duration=duration
        )
    else:
        audio, _ = lr.load(fpath, sr=JUKEBOX_SAMPLE_RATE, offset=offset)

    if audio.ndim == 1:
        audio = audio[np.newaxis]
    audio = audio.mean(axis=0)

    # normalize audio
    norm_factor = np.abs(audio).max()
    if norm_factor > 0:
        audio /= norm_factor

    return audio.flatten()


def get_z(vqvae, audio):
    # don't compute unnecessary discrete encodings
    if type(audio) == type([]):
        audio = np.array(audio)
    else:
        audio = audio[np.newaxis]

    audio = torch.from_numpy(audio[..., np.newaxis]).to(device='cuda')

    zs = vqvae.encode(audio)
    # get the last level of VQ-VAE tokens
    z = zs[-1]

    return z


def get_cond(top_prior):
    # model only accepts sample length conditioning of
    # >60 seconds, so we use 62
    sample_length_in_seconds = 200

    HPS_SAMPLE_LENGTH = (int(sample_length_in_seconds * JUKEBOX_SAMPLE_RATE) // top_prior.raw_to_tokens
                        ) * top_prior.raw_to_tokens
    # NOTE: the 'lyrics' parameter is required, which is why it is included,
    # but it doesn't actually change anything about the `x_cond`, `y_cond`,
    # nor the `prime` variables. The `prime` variable is supposed to represent
    # the lyrics, but the LM prior we're using does not condition on lyrics,
    # so it's just an empty tensor.
    metas = [
                dict(
                    artist="unknown",
                    genre="unknown",
                    total_length=HPS_SAMPLE_LENGTH,
                    offset=0,
                    lyrics="""placeholder lyrics""",
                ),
            ] * 8

    labels = [None, None, top_prior.labeller.get_batch_labels(metas, 'cuda')]

    x_cond, y_cond, prime = top_prior.get_cond(None, top_prior.get_y(labels[-1], 0))

    x_cond = x_cond[0][np.newaxis, ...].to('cuda')
    y_cond = y_cond[0][np.newaxis, ...].to('cuda')

    return x_cond, y_cond


def downsample(representation, target_rate=30, method=None):
    if method is None:
        method = "librosa_fft"

    if method == "librosa_kaiser":
        resampled_reps = lr.resample(
            np.asfortranarray(representation.T), 8192 / DEFAULT_DURATION, target_rate
        ).T
    elif method in ["librosa_fft", "librosa_scipy"]:
        resampled_reps = lr.resample(
            np.asfortranarray(representation.T),
            orig_sr=8192/ DEFAULT_DURATION,
            target_sr=target_rate,
            res_type="fft",
        ).T
    elif method == "mean":
        raise NotImplementedError

    return resampled_reps


def roll(x, n):
    return t.cat((x[:, -n:], x[:, :-n]), dim=1)


def get_activations_custom(
        x, x_cond, y_cond, layers_to_extract=None, fp16=False, fp16_out=False
):
    # this function is adapted from:
    # https://github.com/openai/jukebox/blob/08efbbc1d4ed1a3cef96e08a931944c8b4d63bb3/jukebox/prior/autoregressive.py#L116

    # custom jukemir stuff
    if layers_to_extract is None:
        layers_to_extract = [36]

    input_seq_length = x.shape[1]

    # chop x_cond if input is short
    x_cond = x_cond[:, :input_seq_length]

    # Preprocess.
    with t.no_grad():
        x = TOP_PRIOR.prior.preprocess(x)

    N, D = x.shape
    # assert isinstance(x, t.cuda.LongTensor)
    assert isinstance(x, t.cuda.LongTensor) or isinstance(x, t.LongTensor)
    assert (0 <= x).all() and (x < TOP_PRIOR.prior.bins).all()

    if TOP_PRIOR.prior.y_cond:
        assert y_cond is not None
        assert y_cond.shape == (N, 1, TOP_PRIOR.prior.width)
    else:
        assert y_cond is None

    if TOP_PRIOR.prior.x_cond:
        assert x_cond is not None
        assert x_cond.shape == (N, D, TOP_PRIOR.prior.width) or x_cond.shape == (
            N,
            1,
            TOP_PRIOR.prior.width,
        ), f"{x_cond.shape} != {(N, D, TOP_PRIOR.prior.width)} nor {(N, 1, TOP_PRIOR.prior.width)}. Did you pass the correct --sample_length?"
    else:
        assert x_cond is None
        x_cond = t.zeros((N, 1, TOP_PRIOR.prior.width), device=x.device, dtype=t.float)

    x_t = x  # Target
    # self.x_emb is just a straightforward embedding, no trickery here
    x = TOP_PRIOR.prior.x_emb(x)  # X emb
    # this is to be able to fit in a start token/conditioning info: just shift to the right by 1
    x = roll(x, 1)  # Shift by 1, and fill in start token
    # self.y_cond == True always, so we just use y_cond here
    if TOP_PRIOR.prior.y_cond:
        x[:, 0] = y_cond.view(N, TOP_PRIOR.prior.width)
    else:
        x[:, 0] = TOP_PRIOR.prior.start_token
    # for some reason, p=0.0, so the dropout stuff does absolutely nothing
    x = (
            TOP_PRIOR.prior.x_emb_dropout(x)
            + TOP_PRIOR.prior.pos_emb_dropout(TOP_PRIOR.prior.pos_emb())[:input_seq_length]
            + x_cond
    )  # Pos emb and dropout

    layers = TOP_PRIOR.prior.transformer._attn_mods

    reps = {}

    if fp16:
        x = x.half()

    for i, l in enumerate(layers):
        # to be able to take in shorter clips, we set sample to True,
        # but as a consequence the forward function becomes stateful
        # and its state changes when we apply a layer (attention layer
        # stores k/v's to cache), so we need to clear its cache religiously
        l.attn.del_cache()

        x = l(x, encoder_kv=None, sample=True)

        l.attn.del_cache()

        if i + 1 in layers_to_extract:
            reps[i + 1] = np.array(x.squeeze().cpu())

            # break if this is the last one we care about
            if layers_to_extract.index(i + 1) == len(layers_to_extract) - 1:
                break

    return reps


# important, gradient info takes up too much space,
# causes CUDA OOM
@torch.no_grad()
def extract(
        audio=None,
        fpath=None,
        vqvae=None,
        prior=None,
        meanpool=False,
        layers=None,
        offset=0.0,
        duration=None,
        # downsampling frame-wise reps
        downsample_target_rate=None,
        downsample_method=None,
        # for speed-saving
        fp16=False,
        # for space-saving
        fp16_out=False,
        force_empty_cache=True,

):
    # main function that runs extraction end-to-end.

    if layers is None:
        layers = [36]  # by default

    if audio is None:
        assert fpath is not None

        if type(fpath) == type([]):
            audio = [
                load_audio(path, offset=offset, duration=duration) for path in fpath
            ]
            bsize = len(fpath)
        else:
            audio = load_audio(fpath, offset=offset, duration=duration)
            bsize = 1

    elif fpath is None:
        assert audio is not None

        if type(audio) == type([]):
            bsize = len(audio)
        else:
            bsize = 1
    input_secs = len(audio) / JUKEBOX_SAMPLE_RATE
    if force_empty_cache:
        empty_cache()

    # run vq-vae on the audio to get discretized audio
    z = get_z(VQVAE, audio)

    if force_empty_cache:
        empty_cache()

    x_cond, y_cond = get_cond(TOP_PRIOR)

    # avoid raising asserts
    x_cond, y_cond = x_cond.repeat(bsize, 1, 1), y_cond.repeat(bsize, 1, 1)

    if force_empty_cache:
        empty_cache()

    # get the activations from the LM
    acts = get_activations_custom(
        z, x_cond, y_cond, layers_to_extract=layers, fp16=fp16, fp16_out=fp16_out
    )

    if force_empty_cache:
        empty_cache()

    # postprocessing
    if downsample_target_rate is not None:
        for num in acts.keys():
            if bsize == 1:
                acts[num] = downsample(
                    acts[num],
                    target_rate=downsample_target_rate,
                    method=downsample_method,
                )
            else:
                acts[num] = np.array(
                    [
                        downsample(
                            act,
                            target_rate=downsample_target_rate,
                            method=downsample_method,
                        )
                        for act in acts[num]
                    ]
                )

    if meanpool:
        acts = {num: act.mean(axis=0) for num, act in acts.items()}

    if not fp16_out:
        acts = {num: act.astype(np.float32) for num, act in acts.items()}
    output_secs = len(acts[66]) / downsample_target_rate
    assert int(abs(input_secs - output_secs)) < 1, f'input sec: {input_secs}, output sec:{output_secs}'
    return acts[66]


if __name__ == '__main__':
    args = argparse.ArgumentParser('extract jukebox features with jukebox-5b prior_2 layer')
    args.add_argument('--sound_root', default='data/aist_plusplus_final/raw_music')
    args.add_argument('--save_root', default='data/aist_plusplus_final/jukebox')
    args.add_argument('--postfix',default='wav')
    args = args.parse_args()
    os.makedirs(args.save_root, exist_ok=True)
    VQVAE, TOP_PRIOR = setup_models()
    for path in tqdm(glob(join(args.sound_root, f'*.{args.postfix}'))):
        save_path = join(args.save_root, basename(path).replace(f'.{args.postfix}', '.npy'))
        # if exists(save_path):
        #     continue
        embeds = extract(fpath=path, layers=[66], downsample_target_rate=20, fp16=True, fp16_out=True)
        np.save(save_path, embeds)
