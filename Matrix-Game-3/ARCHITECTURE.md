# Matrix-Game 3.0 Architecture

## Scope

- This note covers `Matrix-Game-3/`, the Matrix-Game 3.0 implementation.
- The repository root also contains older `Matrix-Game-1/` and `Matrix-Game-2/` releases, but they are separate code trees.
- Matrix-Game 3.0 is a Python inference repo for real-time, streaming, action-conditioned video generation from an initial image and text prompt.

## System Structure

- `generate.py` is the public CLI entry point.
- `pipeline/` orchestrates end-to-end inference, model loading, autoregressive generation, memory selection, VAE decoding, and video output.
- `wan/` contains the adapted Wan model stack: DiT backbone, text encoder, VAE, attention, quantization, schedulers, configs, and distributed helpers.
- `utils/` prepares image/action/camera conditioning and renders output videos with control overlays.
- `demo_images/` and `assets/` provide sample inputs and presentation assets.

## Main Components

| Component | Key Files | Responsibility |
| --- | --- | --- |
| CLI | `generate.py` | Parses arguments, initializes distributed execution, loads config, chooses the standard or interactive pipeline, and starts generation. |
| Inference orchestration | `pipeline/inference_pipeline.py`, `pipeline/inference_interactive_pipeline.py` | Loads T5, DiT, and VAE; builds conditions; runs segment-by-segment latent generation; decodes and saves output. |
| Async VAE | `pipeline/vae_worker.py`, `pipeline/vae_config.py` | Selects VAE checkpoints and optionally decodes latent chunks in a separate process/GPU. |
| DiT model runtime | `wan/modules/model.py` | Main `WanModel` diffusion transformer with text, action, camera, memory, and int8 hooks. |
| Action control | `wan/modules/action_module.py`, `utils/conditions.py` | Encodes mouse/keyboard controls and provides default non-interactive action sequences. |
| VAE runtime | `wan/modules/vae2_2.py` | Encodes the initial image and stream-decodes generated latents into video frames. |
| Text encoding | `wan/modules/t5.py`, `wan/modules/tokenizers.py` | Wraps UMT5 text encoding and tokenization. |
| Attention and position encoding | `wan/modules/attention.py`, `wan/modules/posemb_layers.py` | Routes Flash Attention/SDPA and provides rotary positional embeddings. |
| Config and scheduling | `wan/configs/`, `wan/utils/fm_solvers*.py` | Defines model defaults, supported size keys, and flow-matching schedulers. |
| Distributed/performance | `wan/distributed/`, `wan/triton_kernels.py` | Supports FSDP, Ulysses-style sequence parallelism, and quantized kernels. |
| Conditioning utilities | `utils/utils.py`, `utils/cam_utils.py`, `utils/transform.py` | Preprocesses images, rolls actions into poses, builds Plucker camera embeddings, and selects memory frames. |
| Output rendering | `utils/visualize.py` | Overlays keyboard/mouse state and exports MP4 videos. |

## Runtime Flow

1. User runs `torchrun ... generate.py` with `--ckpt_dir`, `--image`, `--prompt`, output options, and optional distributed/performance flags.
2. `generate.py` validates args, initializes `torch.distributed` when needed, loads `WAN_CONFIGS["matrix_game3"]`, and instantiates a pipeline.
3. The pipeline loads T5 from `models_t5_umt5-xxl-enc-bf16.pth`/`google/umt5-xxl`, DiT from `base_model/` or `base_distilled_model/`, and VAE from `Wan2.2_VAE.pth`, `MG-LightVAE.pth`, or `MG-LightVAE_v2.pth`.
4. The pipeline prepares text embeddings, initial image latents, mouse/keyboard actions, camera poses, Plucker embeddings, and memory conditions.
5. Generation runs autoregressively: the first segment produces 57 frames, later segments add 40 frames each, and older latent frames are selected by camera FOV overlap for long-horizon memory.
6. For each diffusion timestep, `WanModel` predicts noise from latent, text, action, camera, and memory conditions; the UniPC scheduler updates latents.
7. Final latents are decoded with streaming VAE, optionally by `vae_worker.py`, then `process_video()` overlays controls and writes MP4 output.

## End-to-End Generation Flow

### 1. Request Entry

| Step | File / Function | What Happens |
| --- | --- | --- |
| Parse CLI | `generate.py::_parse_args()` | Collects model path, prompt, image, output settings, sampling steps, VAE mode, int8, Flash Attention, FSDP, sequence parallelism, and `--interactive`. |
| Validate inputs | `generate.py::_validate_args()` | Requires `--ckpt_dir`, `--prompt`, and `--image`; fills sampler defaults from `WAN_CONFIGS["matrix_game3"]`; disables FSDP on single-GPU runs. |
| Start runtime | `generate.py::generate()` | Creates the output directory, sets seed through `utils.misc.set_seed()`, initializes NCCL when launched with multiple processes, and opens the input image. |
| Select pipeline | `generate.py::generate()` | Uses `pipeline.inference_pipeline.MatrixGame3Pipeline` by default or `pipeline.inference_interactive_pipeline.MatrixGame3Pipeline` when `--interactive` is passed. |

### 2. Pipeline Construction

`MatrixGame3Pipeline.__init__()` is the main assembly point in both pipeline files.

| Component | Constructor Path | Role in the Flow |
| --- | --- | --- |
| Text encoder | `wan.modules.t5.T5EncoderModel` | Loads UMT5 checkpoint/tokenizer and converts prompt text into context embeddings. |
| DiT backbone | `wan.modules.model.WanModel.load_config()` and `WanModel.from_pretrained()` | Loads either `base_model/` or `base_distilled_model/` and becomes the denoising model called every scheduler step. |
| Optional quantization | `wan.modules.model.convert_model_to_int8()` | Replaces selected projection layers with int8 paths when `--use_int8` is enabled. |
| Distributed setup | `MatrixGame3Pipeline._configure_model()` | Applies sequence-parallel forward overrides (`sp_dit_forward`, `sp_attn_forward`) and/or FSDP sharding via `wan.distributed.fsdp.shard_model()`. |
| VAE | `pipeline.vae_config.load_vae()` | Loads Wan VAE or MG-LightVAE for image encoding and latent-to-video decoding. |
| Async decoder | `pipeline.vae_worker.start_vae_worker_process()` | Starts a separate VAE process when `--use_async_vae` is enabled, so diffusion can queue latents while decoding runs independently. |

### 3. Input Conditioning

The generation loop starts in `MatrixGame3Pipeline.generate()`.

| Data | Standard Pipeline | Interactive Pipeline |
| --- | --- | --- |
| Image tensor | `utils.utils.get_data()` preprocesses the PIL image through `utils.transform.get_video_transform()`. | The image is resized and normalized directly in `pipeline/inference_interactive_pipeline.py`. |
| Text condition | `self.text_encoder([text], device=self.device)` creates positive context; `self.text_encoder([self.config.sample_neg_prompt], ...)` creates negative context for base-model CFG. | Same. |
| Actions | `utils.conditions.Bench_actions_universal()` creates random benchmark keyboard/mouse sequences. | `get_current_action()` prompts the user each segment for camera and movement keys. |
| Camera poses | `utils.utils.compute_all_poses_from_actions()` rolls actions into positions/pitch/yaw, then `utils.utils.get_extrinsics()` builds camera matrices. | Same pose logic, but appended segment by segment from user input. |
| Camera embeddings | `utils.utils.build_plucker_from_c2ws()` converts camera trajectories into Plucker embeddings used by `WanModel`. | Same. |
| Initial latent | `self.vae.encode([current_image[0]])` encodes the first image into `img_cond`, then broadcasts it to other ranks if distributed. | Same. |

### 4. Autoregressive Segment Loop

Generation is chunked so long videos can reuse previous latent context.

| Stage | Key Code | Meaning |
| --- | --- | --- |
| Segment schedule | `clip_frame = 56`, `first_clip_frame = 57`, `past_frame = 16` in `MatrixGame3Pipeline.generate()` | First clip covers 57 frames; later clips advance by 40 frames because 16 frames overlap with prior context. |
| Frame count | `num_frames = first_clip_frame + (num_iterations - 1) * 40` | `--num_iterations` controls output duration. |
| Current window | `current_start_frame_idx`, `current_end_frame_idx` | Defines which action/camera frames condition the current latent chunk. |
| Target latent range | local `get_latent_idx()` helper | Maps frame indices into VAE-compressed latent time indices. |
| Plucker chunk | `build_plucker_from_c2ws(...)` | Builds camera conditioning for the current chunk at latent resolution. |

### 5. Long-Horizon Memory

For the first clip, memory inputs are `None`. For later clips, the pipeline selects prior latent frames and feeds them back into the DiT.

| Step | File / Function | What Moves Forward |
| --- | --- | --- |
| Pick memory candidates | `utils.cam_utils.select_memory_idx_fov()` | Finds earlier frames whose camera frustums overlap with the current view. |
| Build memory camera condition | `utils.utils.build_plucker_from_pose()` | Creates Plucker embeddings for selected memory poses. |
| Gather latent memory | `x_memory = src[:, :, latent_idx]` in `MatrixGame3Pipeline.generate()` | Extracts selected previous latents from `all_latents_list`. |
| Add memory metadata | `conditions_full` | Passes `x_memory`, `timestep_memory`, memory action placeholders, `memory_latent_idx`, and `predict_latent_idx` into `WanModel.forward()`. |
| Use memory inside DiT | `wan.modules.model.WanModel.forward()` | Concatenates memory latents before current latents, embeds Plucker camera data, applies action/text conditioning, and returns only the newly predicted latent portion. |

### 6. Denoising Loop

Each segment is denoised with `wan.utils.fm_solvers_unipc.FlowUniPCMultistepScheduler`.

| Step | File / Function | What Happens |
| --- | --- | --- |
| Create scheduler | `FlowUniPCMultistepScheduler()` | Builds timestep schedule with `set_timesteps(num_inference_steps, shift=shift)`. |
| Initialize latent noise | `torch.randn(...)` in `MatrixGame3Pipeline.generate()` | Creates the latent tensor for the current segment and pins the first latent slice to `img_cond`. |
| Build conditions | `conditions_full`, `conditions_null` | Full conditions include prompt/actions/camera/memory; null conditions use the negative prompt and disabled controls for classifier-free guidance. |
| Predict noise | `self.model(**model_kwargs)` | Calls `WanModel.forward()` with latent, timestep, sequence length, text, action, camera, and memory inputs. |
| CFG path | `if use_base_model:` | Base model runs full and null passes, then combines them with `guide_scale`; distilled model uses the single full-condition prediction. |
| Update latent | `test_scheduler.step(noise_pred, t, latents, return_dict=False)[0]` | Moves latents one diffusion step closer to decoded video. |
| Preserve image condition | `latents = torch.cat([img_cond, latents[:, :, img_cond.shape[2]:]], dim=2)` | Keeps the initial image latent fixed across denoising steps. |

### 7. Decoding and Output

After a segment finishes, `denoised_pred` is decoded or queued.

| Mode | Path | Output Behavior |
| --- | --- | --- |
| Synchronous VAE | `self.vae.stream_decode(...)` in the pipeline | Rank 0 decodes each segment, stores decoded tensors in `all_videos_list`, concatenates them at the end, and writes `{save_name}.mp4`. |
| Async VAE | `pipeline.vae_worker.run_vae_worker_queue()` | Rank 0 sends CPU latents through `vae_latent_queue`; the worker decodes chunks, optionally writes per-iteration interactive clips, concatenates all videos, and writes final MP4. |
| Video rendering | `utils.visualize.process_video()` | Draws keyboard keys and mouse cursor from the action arrays, then exports the MP4 via Diffusers `export_to_video()`. |

### 8. Data Movement Summary

| Object | Producer | Consumer |
| --- | --- | --- |
| CLI args | `generate.py::_parse_args()` | `generate.py::generate()` and `MatrixGame3Pipeline.__init__/generate()` |
| PIL image | `Image.open(args.image)` | VAE encoder and image preprocessing utilities |
| Prompt text | CLI `--prompt` | `T5EncoderModel.__call__()` |
| Action tensors | `Bench_actions_universal()` or `get_current_action()` | Action module and output overlay renderer |
| Camera extrinsics | `compute_all_poses_from_actions()` plus `get_extrinsics()` | Plucker builders and memory selection |
| Plucker embeddings | `build_plucker_from_c2ws()` / `build_plucker_from_pose()` | `WanModel.forward()` camera conditioning |
| Current latents | Scheduler loop | `WanModel.forward()`, then VAE decoder |
| Memory latents | `all_latents_list` | Later `WanModel.forward()` calls as `x_memory` |
| Decoded frames | `Wan2_2_VAE.stream_decode()` | `utils.visualize.process_video()` |
| MP4 file | `process_video()` | Written under `--output_dir` as `{save_name}.mp4` and, in interactive sync/async paths, optional per-iteration clips. |

## Key Entry Points

- `Matrix-Game-3/generate.py`: main command-line entry point.
- `Matrix-Game-3/test.sh`: reference multi-GPU command using FSDP, Ulysses sequence parallelism, int8 DiT, interactive mode, compiled LightVAE, and async VAE.
- `Matrix-Game-3/pipeline/inference_pipeline.py`: non-interactive generation with generated/random action sequences.
- `Matrix-Game-3/pipeline/inference_interactive_pipeline.py`: console-driven interactive action input.
- `Matrix-Game-3/pipeline/vae_worker.py`: async VAE process entry point.

## Important Folders

- `pipeline/`: orchestration layer and VAE process management.
- `wan/modules/`: core neural network modules.
- `wan/configs/`: runtime/model configuration registry.
- `wan/distributed/`: multi-GPU FSDP and sequence-parallel support.
- `wan/utils/`: schedulers and generic media helpers inherited/adapted from Wan.
- `utils/`: Matrix-Game-specific action, camera, preprocessing, and visualization utilities.
- `demo_images/`: image and prompt examples for inference.
- `assets/images/`: overlay assets used in rendered videos.
- `assets/pdf/`: technical report.

## Critical Files To Check First

- `README.md`: setup, model download, and example commands.
- `generate.py`: CLI arguments and top-level control flow.
- `pipeline/inference_pipeline.py`: default generation loop and memory logic.
- `pipeline/inference_interactive_pipeline.py`: interactive variant and action collection.
- `pipeline/vae_config.py`: VAE checkpoint selection.
- `wan/configs/config.py`: model dimensions and sampler defaults.
- `wan/modules/model.py`: DiT forward contract and condition names.
- `wan/modules/action_module.py`: keyboard/mouse conditioning.
- `wan/modules/vae2_2.py`: streaming VAE behavior.
- `utils/utils.py`: action-to-pose conversion and Plucker condition construction.
- `utils/cam_utils.py`: memory-frame selection and camera geometry.
