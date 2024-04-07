import gradio as gr

import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
import numpy as np
from tqdm import tqdm
import csv
import logging
from speechbrain.core import AMPConfig
from torch.utils.data import Dataset,DataLoader


# Define training procedure
class Separation(sb.Brain):
    def compute_forward(self, mix, targets, stage, noise=None):
        """Forward computations from the mixture to the separated signals."""

        # Unpack lists and put tensors in the right device
        mix = mix
        mix = mix.to(self.device)

        # Convert targets to tensor
        targets = torch.cat(
            [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
            dim=-1,
        ).to(self.device)

        # Add speech distortions
        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                if self.hparams.use_speedperturb:
                    mix, targets = self.add_speed_perturb(targets)

                    mix = targets.sum(-1)

                # if self.hparams.use_wavedrop:
                #     mix = self.hparams.drop_chunk(mix)
                #     mix = self.hparams.drop_freq(mix)

                if self.hparams.limit_training_signal_len:
                    mix, targets = self.cut_signals(mix, targets)

        # Separation
        mix_w = self.hparams.Encoder(mix)
        est_mask = self.hparams.MaskNet(mix_w)
        mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source, targets

    def compute_objectives(self, predictions, targets):
        """Computes the sinr loss"""
        return self.hparams.loss(targets, predictions)

    def fit_batch(self, batch):
        """Trains one batch"""
        amp = AMPConfig.from_name(self.precision)
        should_step = (self.step % self.grad_accumulation_factor) == 0

        # Unpacking batch list
        mixture = batch["mixed_audio"]
        targets = [batch["source_1"], batch["source_2"]]

        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        with self.no_sync(not should_step):
            if self.use_amp:
                with torch.autocast(
                    dtype=amp.dtype, device_type=torch.device(self.device).type
                ):
                    predictions, targets = self.compute_forward(
                        mixture, targets, sb.Stage.TRAIN
                    )
                    loss = self.compute_objectives(predictions, targets)

                    # hard threshold the easy dataitems
                    if self.hparams.threshold_byloss:
                        th = self.hparams.threshold
                        loss = loss[loss > th]
                        if loss.nelement() > 0:
                            loss = loss.mean()
                    else:
                        loss = loss.mean()

                if (
                    loss.nelement() > 0 and loss < self.hparams.loss_upper_lim
                ):  # the fix for computational problems
                    self.scaler.scale(loss).backward()
                    if self.hparams.clip_grad_norm >= 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.modules.parameters(),
                            self.hparams.clip_grad_norm,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.nonfinite_count += 1
                    logger.info(
                        "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                            self.nonfinite_count
                        )
                    )
                    loss.data = torch.tensor(0).to(self.device)
            else:
                predictions, targets = self.compute_forward(
                    mixture, targets, sb.Stage.TRAIN
                )
                loss = self.compute_objectives(predictions, targets)

                if self.hparams.threshold_byloss:
                    th = self.hparams.threshold
                    loss = loss[loss > th]
                    if loss.nelement() > 0:
                        loss = loss.mean()
                else:
                    loss = loss.mean()

                if (
                    loss.nelement() > 0 and loss < self.hparams.loss_upper_lim
                ):  # the fix for computational problems
                    loss.backward()
                    if self.hparams.clip_grad_norm >= 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.modules.parameters(),
                            self.hparams.clip_grad_norm,
                        )
                    self.optimizer.step()
                else:
                    self.nonfinite_count += 1
                    logger.info(
                        "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                            self.nonfinite_count
                        )
                    )
                    loss.data = torch.tensor(0).to(self.device)
        self.optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        # snt_id = batch.id
        mixture = batch["mixed_audio"].squeeze(0)
        targets = [batch["source_1"], batch["source_2"]]
        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        with torch.no_grad():
            predictions, targets = self.compute_forward(mixture, targets, stage)
            loss = self.compute_objectives(predictions, targets)

        return loss.mean().detach()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"si-snr": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            # Learning rate annealing
            if isinstance(
                self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                current_lr = self.hparams.optimizer.optim.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"si-snr": stage_stats["si-snr"]}, min_keys=["si-snr"]
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def add_speed_perturb(self, targets):
        """Adds speed perturbation and random_shift to the input signals"""

        min_len = -1
        recombine = False

        if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
            # Performing speed change (independently on each source)
            new_targets = []
            recombine = True

            for i in range(targets.shape[-1]):
                new_target = self.hparams.speed_perturb(targets[:, :, i])
                new_targets.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                else:
                    if new_target.shape[-1] < min_len:
                        min_len = new_target.shape[-1]

            if self.hparams.use_rand_shift:
                # Performing random_shift (independently on each source)
                recombine = True
                for i in range(targets.shape[-1]):
                    rand_shift = torch.randint(
                        self.hparams.min_shift, self.hparams.max_shift, (1,)
                    )
                    new_targets[i] = new_targets[i].to(self.device)
                    new_targets[i] = torch.roll(
                        new_targets[i], shifts=(rand_shift[0],), dims=1
                    )

            # Re-combination
            if recombine:
                if self.hparams.use_speedperturb:
                    targets = torch.zeros(
                        targets.shape[0],
                        min_len,
                        targets.shape[-1],
                        device=targets.device,
                        dtype=torch.float,
                    )
                for i, new_target in enumerate(new_targets):
                    targets[:, :, i] = new_targets[i][:, 0:min_len]

        mix = targets.sum(-1)
        return mix, targets

    def cut_signals(self, mixture, targets):
        """This function selects a random segment of a given length within the mixture.
        The corresponding targets are selected accordingly"""
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        targets = targets[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        mixture = mixture[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return mixture, targets

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def save_results(self, test_data):
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file"""

        # This package is required for SDR computation
        from mir_eval.separation import bss_eval_sources

        # Create folders where to store audio
        save_file = os.path.join(self.hparams.output_folder, "test_results.csv")

        # Variable init
        all_sdrs = []
        all_sdrs_i = []
        all_sisnrs = []
        all_sisnrs_i = []
        csv_columns = ["sdr", "sdr_i", "si-snr", "si-snr_i"]

        test_loader = sb.dataio.dataloader.make_dataloader(
            test_data, **self.hparams.dataloader_opts
        )

        with open(save_file, "w") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):
                    # Apply Separation
                    mixture =  batch["mixed_audio"].squeeze(0)
                    targets = [batch["source_1"], batch["source_2"]]
                    if self.hparams.num_spks == 3:
                        targets.append(batch.s3_sig)

                    with torch.no_grad():
                        predictions, targets = self.compute_forward(
                            mixture, targets, sb.Stage.TEST
                        )

                    # Compute SI-SNR
                    sisnr = self.compute_objectives(predictions, targets)

                    # Compute SI-SNR improvement
                    mixture_signal = torch.stack(
                        [mixture] * self.hparams.num_spks, dim=-1
                    )
                    mixture_signal = mixture_signal.to(targets.device)
                    sisnr_baseline = self.compute_objectives(
                        mixture_signal, targets
                    )
                    sisnr_i = sisnr - sisnr_baseline

                    # Compute SDR
                    sdr, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        predictions[0].t().detach().cpu().numpy(),
                    )

                    sdr_baseline, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        mixture_signal[0].t().detach().cpu().numpy(),
                    )

                    sdr_i = sdr.mean() - sdr_baseline.mean()

                    # Saving on a csv file
                    row = {
                        "sdr": sdr.mean(),
                        "sdr_i": sdr_i,
                        "si-snr": -sisnr.item(),
                        "si-snr_i": -sisnr_i.item(),
                    }
                    writer.writerow(row)

                    # Metric Accumulation
                    all_sdrs.append(sdr.mean())
                    all_sdrs_i.append(sdr_i.mean())
                    all_sisnrs.append(-sisnr.item())
                    all_sisnrs_i.append(-sisnr_i.item())

                row = {
                    "sdr": np.array(all_sdrs).mean(),
                    "sdr_i": np.array(all_sdrs_i).mean(),
                    "si-snr": np.array(all_sisnrs).mean(),
                    "si-snr_i": np.array(all_sisnrs_i).mean(),
                }
                writer.writerow(row)

        logger.info("Mean SISNR is {}".format(np.array(all_sisnrs).mean()))
        logger.info("Mean SISNRi is {}".format(np.array(all_sisnrs_i).mean()))
        logger.info("Mean SDR is {}".format(np.array(all_sdrs).mean()))
        logger.info("Mean SDRi is {}".format(np.array(all_sdrs_i).mean()))

class LibriMixDataset(Dataset):
    def __init__(self, mix_files, s1_dir, s2_dir, transform=None):
        self.mix_both_dir = mix_both_dir
        self.s1_dir = s1_dir
        self.s2_dir = s2_dir
        self.mix_files = mix_files
        self.transform = transform

    def __len__(self):
        return len(self.mix_files)

    def __getitem__(self, idx):
        mix_file = self.mix_files[idx]
        mix_path = os.path.join(self.mix_both_dir, mix_file)
        s1_path = os.path.join(self.s1_dir, mix_file)
        s2_path = os.path.join(self.s2_dir, mix_file)

        # Load mixed audio and sources
        mix_audio, _ = torchaudio.load(mix_path)
        s1_audio, _ = torchaudio.load(s1_path)
        s2_audio, _ = torchaudio.load(s2_path)

        return {'mixed_audio': mix_audio, 'source_1': s1_audio, 'source_2': s2_audio,'mixed_audio_path':mix_path}

def collate_fn(batch):
        # Get maximum length among all audio files in the batch
        max_len = max(max(sample['mixed_audio'].size(1), sample['source_1'].size(1), sample['source_2'].size(1)) for sample in batch)

        # Pad each audio file in the batch to match the maximum length
        padded_batch = []
        for sample in batch:
            padded_mixed_audio = torch.nn.functional.pad(sample['mixed_audio'], (0, max_len - sample['mixed_audio'].size(1)))
            padded_source_1 = torch.nn.functional.pad(sample['source_1'], (0, max_len - sample['source_1'].size(1)))
            padded_source_2 = torch.nn.functional.pad(sample['source_2'], (0, max_len - sample['source_2'].size(1)))
            padded_batch.append({'mixed_audio': padded_mixed_audio, 'source_1': padded_source_1, 'source_2': padded_source_2 , 'mixed_audio_path':sample['mixed_audio_path']})

        return padded_batch


if __name__ == "__main__":

    model_path = r"Trained_model\trained_separator.pth"
    

    model_state = torch.load(model_path)

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    
    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # # for key, module in separator.modules.items():
    # #     if key in model_state:
    # #         module.load_state_dict(model_state[key])
    # #     else:
    # #         print(f"Warning: Module {key} not found in the loaded model state.")



    # # Initialize ddp (useful only for multi-GPU DDP training)
    # sb.utils.distributed.ddp_init_group(run_opts)

    # # Logger info
    # logger = logging.getLogger(__name__)

    # # Create experiment directory
    # sb.create_experiment_directory(
    #     experiment_directory=hparams["output_folder"],
    #     hyperparams_to_save=hparams_file,
    #     overrides=overrides,
    # )

    # # Update precision to bf16 if the device is CPU and precision is fp16
    # if run_opts.get("device") == "cpu" and hparams.get("precision") == "fp16":
    #     hparams["precision"] = "bf16"

    # # Check if wsj0_tr is set with dynamic mixing
    # if hparams["dynamic_mixing"] and not os.path.exists(
    #     hparams["base_folder_dm"]
    # ):
    #     raise ValueError(
    #         "Please, specify a valid base_folder_dm folder when using dynamic mixing"
    #     )

    # mix_both_dir = "/content/wav8k/max/test/mix_both"
    # s1_dir =  "/content/wav8k/max/test/s1"
    # s2_dir = "/content/wav8k/max/test/s2"

    # file_names = os.listdir(mix_both_dir)

    # split_idx = int(0.7 * len(file_names))

    # train_file_names = file_names[:split_idx]
    # train_file_names = train_file_names[:9]
    # test_file_names = file_names[split_idx:]

    # # Create datasets
    # train_dataset = LibriMixDataset(train_file_names, s1_dir, s2_dir)
    # test_dataset = LibriMixDataset(test_file_names, s1_dir, s2_dir)

    # # dataset = LibriMixDataset(mix_both_dir, s1_dir, s2_dir)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

    # # Training
    # separator.fit(
    #     separator.hparams.epoch_counter,
    #     train_dataset,
    #     train_loader_kwargs=hparams["dataloader_opts"],
    # )

    # # Eval
    # separator.evaluate(test_dataset, min_key="si-snr")
    # separator.save_results(test_dataset)

    # model_state = {key: module.state_dict() for key, module in separator.modules.items()}
    # torch.save(model_state, os.path.join(hparams["output_folder"], "trained_separator.pth"))



    # # re-initialize the parameters if we don't use a pretrained model
    # if "pretrained_separator" not in hparams:
    #     for module in separator.modules.values():
    #         separator.reset_layer_recursively(module)

    # def separate_audio(input_audio):
    # # Perform separation using your trained model
    # # Assuming you have a function called `separate_audio` in your `Separation` class
    #     separated_audio,_ = separator(input_audio)
    #     return separated_audio

    # # # Load your trained model
    # # separator = Separation.load_model("<path_to_your_trained_model>")

    # # Create a Gradio interface
    # iface = gr.Interface(
    #     fn=separate_audio,
    #     inputs="microphone",
    #     outputs="audio",
    #     title="Audio Separation",
    #     description="Separate audio input into individual sources.",
    #     interpretation="audio",
    # )

    # # Launch the interface
    # iface.launch()        


