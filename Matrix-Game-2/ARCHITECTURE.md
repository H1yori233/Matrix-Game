# Matrix-Game 2.0 Architecture

## Scope

- This note covers `Matrix-Game-2/`, the Matrix-Game 2.0 implementation.
- The repository root also contains older/newer `Matrix-Game-1/` and `Matrix-Game-3/` releases, but they are separate code trees.
- Matrix-Game 2.0 is a Python inference repo for real-time, streaming, action-conditioned image-to-world video generation from an initial image.
- Unlike Matrix-Game 3.0, the 2.0 inference path does not use a text prompt. It conditions generation on the first image, CLIP visual features, keyboard inputs, and optionally mouse/camera inputs.

## System Structure

- `inference.py` is the public batch-style CLI for generating videos with random benchmark action trajectories.
- `inference_streaming.py` is the public console-interactive CLI for generating videos with user-provided actions and images.
- `pipeline/` contains the block-causal inference loops, including standard and streaming variants.
- `wan/` contains the adapted Wan/SkyReels model stack: causal DiT backbone, action modules, attention, VAE wrappers, schedulers, configs, and distributed helpers.
- `utils/` provides diffusion wrappers, action trajectory synthesis, scheduler utilities, seeding, and video visualization.
- `demo_utils/` provides the optimized streaming VAE decoder, VAE cache definitions, TensorRT helpers, and memory/offload helpers.
- `configs/` selects inference mode, distilled model architecture, local attention window size, action dimensions, and denoising steps.
- `demo_images/` and `assets/` provide sample inputs, control-overlay assets, demo videos, and the report PDF.

## Main Components

| Component | Key Files | Responsibility |
| --- | --- | --- |
| Random-action CLI | `inference.py` | Parses arguments, loads config/checkpoints, prepares one input image, synthesizes benchmark actions, runs causal inference, and writes MP4 outputs. |
| Interactive CLI | `inference_streaming.py` | Repeatedly asks for an image path and per-block keyboard/mouse actions, streams decoded chunks, and writes current/final MP4 outputs. |
| Inference orchestration | `pipeline/causal_inference.py` | Implements `CausalInferencePipeline` and `CausalInferenceStreamingPipeline`, blockwise denoising, KV-cache management, streaming VAE decode, and output chunking. |
| Diffusion wrapper | `utils/wan_wrapper.py` | Builds `CausalWanModel`, owns the flow-matching scheduler, converts flow predictions to x0, and forwards latent/action/image conditions into the model. |
| Causal DiT runtime | `wan/modules/causal_model.py` | Main image-to-video transformer with blockwise causal self-attention, local KV caches, image cross-attention, action conditioning, and unpatchified latent output. |
| Action control | `wan/modules/action_module.py`, `utils/conditions.py` | Encodes keyboard/mouse controls and creates random benchmark trajectories for universal, GTA driving, and TempleRun modes. |
| VAE and CLIP encoding | `wan/vae/wanx_vae.py` | Loads Wan VAE and OpenCLIP/XLM-R visual encoder; encodes the initial image video condition and CLIP visual context. |
| Streaming VAE decode | `demo_utils/vae_block3.py`, `demo_utils/constant.py` | Loads only the Wan VAE decoder weights into a cache-aware decoder and decodes latent blocks incrementally. |
| Scheduling | `utils/scheduler.py` | Implements the flow-matching sigma/timestep schedule, `add_noise()`, `step()`, and conversion helpers. |
| Rendering | `utils/visualize.py` | Draws keyboard and optional mouse overlays and exports MP4 videos. |
| Configs | `configs/inference_yaml/*.yaml`, `configs/distilled_model/*/config.json` | Define inference mode, denoising steps, frames per block, timestep shift, action module shape, model dimensions, and local attention size. |
| Wan support modules | `wan/modules/model.py`, `wan/modules/attention.py`, `wan/modules/vae.py` | Provide shared Wan transformer blocks, Flash Attention/SDPA routing, and VAE building blocks used by causal/inference components. |

## Runtime Flow

1. User runs `python inference.py ...` for random benchmark actions or `python inference_streaming.py ...` for console-driven actions.
2. The entry script loads an OmegaConf inference YAML, initializes CUDA/bfloat16 runtime state, builds `WanDiffusionWrapper`, and creates either `CausalInferencePipeline` or `CausalInferenceStreamingPipeline`.
3. `WanDiffusionWrapper` loads `CausalWanModel.from_config()` from the selected `configs/distilled_model/{mode}/config.json` and attaches a `FlowMatchScheduler` with `timestep_shift`.
4. The script loads the distilled DiT checkpoint from `--checkpoint_path` when provided, loads Wan VAE/CLIP from `--pretrained_model_path`, and loads/compiles a standalone VAE decoder from `Wan2.1_VAE.pth`.
5. The input image is center-cropped/resized to 352x640, normalized, padded into an image-conditioning video, encoded to 16-channel latents at 44x80, and paired with a first-frame mask as `cond_concat`.
6. The same image is encoded by CLIP as `visual_context`; keyboard/mouse action arrays are generated from `utils.conditions` or updated interactively through `get_current_action()`.
7. The pipeline samples latent noise shaped `[B, 16, latent_frames, 44, 80]`, splits it into blocks of `num_frame_per_block` latent frames, denoises each block using the configured few-step distilled schedule, and updates model caches with clean context.
8. Each denoised latent block is decoded immediately by the cache-aware VAE decoder, accumulated, then rendered through `process_video()` with keyboard and optional mouse overlays.

## End-to-End Generation Flow

### 1. Request Entry

| Step | File / Function | What Happens |
| --- | --- | --- |
| Parse random-action CLI | `inference.py::parse_args()` | Collects config path, distilled checkpoint, input image path, output folder, latent frame count, seed, and Wan VAE/CLIP folder. |
| Parse streaming CLI | `inference_streaming.py::parse_args()` | Collects config path, distilled checkpoint, output folder, max latent frame count, seed, and Wan VAE/CLIP folder. |
| Start runtime | `main()` in both entry files | Sets the random seed through `utils.misc.set_seed()`, creates the output directory, and constructs `InteractiveGameInference`. |
| Load mode | `InteractiveGameInference._init_config()` and `generate_videos()` | Reads `mode` from the YAML. Supported modes are `universal`, `gta_drive`, and `templerun`. |
| Choose pipeline | `InteractiveGameInference._init_models()` | Uses `CausalInferencePipeline` for `inference.py` and `CausalInferenceStreamingPipeline` for `inference_streaming.py`. |

### 2. Model Construction

`InteractiveGameInference._init_models()` is the main assembly point in both entry scripts.

| Component | Constructor Path | Role in the Flow |
| --- | --- | --- |
| Causal DiT | `utils.wan_wrapper.WanDiffusionWrapper` | Instantiates `CausalWanModel` from the configured distilled model JSON and exposes the model through a flow-prediction-to-x0 wrapper. |
| Flow scheduler | `WanDiffusionWrapper.get_scheduler()` | Builds a `FlowMatchScheduler`, registers conversion helpers, and supplies `add_noise()` for the multi-step block denoising loop. |
| Distilled checkpoint | `safetensors.torch.load_file()` | Loads `--checkpoint_path` into `pipeline.generator` when provided. |
| VAE decoder | `demo_utils.vae_block3.VAEDecoderWrapper` | Extracts decoder and `conv2` weights from `Wan2.1_VAE.pth`, moves to fp16 CUDA, disables gradients, and compiles the decoder. |
| VAE encoder and CLIP | `wan.vae.wanx_vae.get_wanx_vae_wrapper()` | Loads the full Wan VAE for encoding and CLIP visual encoder for image context. |
| Causal pipeline | `pipeline.causal_inference.CausalInferencePipeline` | Owns the generator, scheduler, decoder, per-layer caches, block size, and denoising loop. |
| Streaming pipeline | `pipeline.causal_inference.CausalInferenceStreamingPipeline` | Adds per-block user action replacement and writes current-video previews during generation. |

### 3. Input Conditioning

The generation setup happens in `InteractiveGameInference.generate_videos()`.

| Data | Random-Action Pipeline | Streaming Pipeline |
| --- | --- | --- |
| Image source | Loads `--img_path` with `diffusers.utils.load_image()`. | Prompts for an image path at the console and retries until it loads. |
| Image tensor | Center-crops to 352x640, converts to tensor, and normalizes to `[-1, 1]`. | Same preprocessing path. |
| Image latent condition | Pads the single image to a video of `1 + 4 * (latent_frames - 1)` pixel frames and encodes it with tiled Wan VAE. | Same, using `--max_num_output_frames` as the latent-frame budget. |
| Masked concat condition | Creates `mask_cond`, clears all but the first latent frame, and concatenates `mask_cond[:, :4]` with `img_cond` to form `cond_concat`. | Same. |
| Visual context | Calls `self.vae.clip.encode_video(image)` and passes the result as `visual_context`. | Same. |
| Actions | `Bench_actions_universal()`, `Bench_actions_gta_drive()`, or `Bench_actions_templerun()` creates full random action arrays. | Starts from benchmark action arrays, then overwrites each generated block with `get_current_action()` input. |
| Noise | Samples `torch.randn([1, 16, latent_frames, 44, 80])`. | Samples `torch.randn([1, 16, max_latent_frames, 44, 80])`. |

### 4. Action Modes

| Mode | Config | Action Channels | Behavior |
| --- | --- | --- | --- |
| `universal` | `configs/inference_yaml/inference_universal.yaml` | Keyboard: 4 (`W/S/A/D`), mouse: 2 | General first-person movement and camera look. |
| `gta_drive` | `configs/inference_yaml/inference_gta_drive.yaml` | Keyboard: 2 (`forward/back`), mouse: 2 | Driving-style forward/backward and left/right steering/camera controls. |
| `templerun` | `configs/inference_yaml/inference_templerun.yaml` | Keyboard: 7, no mouse | Runner actions: no-op, jump, slide, turn left/right, lane left/right. |

The distilled model JSON for each mode configures the matching action dimensions. Universal and TempleRun use `local_attn_size = 6`; GTA driving uses `local_attn_size = 4`.

### 5. Block-Causal Inference Loop

Generation is performed in latent-frame blocks, not as a full bidirectional video.

| Stage | Key Code | Meaning |
| --- | --- | --- |
| Block size | `num_frame_per_block: 3` in inference YAMLs | Each model step generates three latent frames at a time. |
| Pixel-frame mapping | `num_frames = (latent_frames - 1) * 4 + 1` | Keyboard/mouse arrays live at pixel-frame resolution; Wan VAE compresses time by 4. |
| Token geometry | `frame_seq_length = 880` | One latent frame at 44x80 becomes 880 spatial tokens after 2x2 patching. |
| Cache reset | `self.kv_cache1 = self.kv_cache_keyboard = self.kv_cache_mouse = self.crossattn_cache = None` | Each inference call starts with empty transformer/action/cross-attention caches. |
| Causal masks | `CausalWanModel._prepare_blockwise_causal_attn_mask*()` | Creates FlexAttention masks so each block attends only to current and earlier local-window context. |
| Current conditions | `cond_current()` | Slices `cond_concat`, `keyboard_cond`, and optional `mouse_cond` up to the current block; streaming mode can overwrite the current block's actions. |

### 6. KV, Action, and Cross-Attention Caches

The pipeline keeps separate caches because video tokens, mouse tokens, keyboard tokens, and image cross-attention have different shapes.

| Cache | Initializer | Shape / Scope | Purpose |
| --- | --- | --- | --- |
| Video self-attention | `_initialize_kv_cache()` | Per transformer block, `[B, local_attn_size * 880, 12, 128]` | Stores local-window K/V for latent video tokens. |
| Mouse action attention | `_initialize_kv_cache_mouse_and_keyboard()` | Per transformer block, `[B * 880, local_attn_size, 16, 64]` | Stores mouse/action K/V per spatial token. Not used in TempleRun. |
| Keyboard action attention | `_initialize_kv_cache_mouse_and_keyboard()` | Per transformer block, `[B, local_attn_size, 16, 64]` | Stores keyboard K/V and repeats across spatial tokens when RoPE keyboard mode is enabled. |
| Image cross-attention | `_initialize_crossattn_cache()` | Per transformer block, `[B, 257, 12, 128]` | Caches CLIP image-context K/V so it does not need to be recomputed each block. |

After a block is denoised, the pipeline reruns the generator at `context_noise` with the clean `denoised_pred`. That pass advances the caches so future blocks can attend to generated history without recomputing the full prefix.

### 7. Denoising Loop

Each latent block uses a short distilled schedule from the YAML config.

| Step | File / Function | What Happens |
| --- | --- | --- |
| Prepare timesteps | `CausalInferencePipeline.__init__()` | Reads `denoising_step_list`; if `warp_denoising_step` is true, maps those ids through the 1000-step flow scheduler. |
| Slice current noise | `CausalInferencePipeline.inference()` | Selects the current `[B, 16, num_frame_per_block, 44, 80]` latent noise block. |
| Predict x0 | `WanDiffusionWrapper.forward()` | Calls `CausalWanModel.forward()` and converts flow prediction to clean latent `pred_x0`. |
| Re-noise between steps | `self.scheduler.add_noise()` | For all but the final denoising step, re-corrupts the x0 prediction at the next configured timestep. |
| Record clean block | `output[:, :, current_start_frame:...] = denoised_pred` | Stores the final x0 prediction in the full latent output tensor. |
| Update context | `self.generator(... timestep=context_timestep ...)` | Runs the clean block through the model to advance self-attention, action, and cross-attention caches. |
| Decode block | `self.vae_decoder(denoised_pred.transpose(1, 2).half(), *vae_cache)` | Converts the generated latent block to RGB frames and updates the streaming VAE cache. |

Universal and GTA configs use three denoising ids: `1000`, `666`, and `333`. TempleRun uses four ids: `1000`, `750`, `500`, and `250`.

### 8. CausalWanModel Forward Contract

`CausalWanModel` is an image-to-video DiT with action injection.

| Input | Producer | Consumer |
| --- | --- | --- |
| `x` | Current latent noise block | Concatenated with `cond_concat`, patchified by `patch_embedding`, and denoised. |
| `t` | Pipeline timestep tensor | Encoded by sinusoidal/time MLPs and used to modulate transformer blocks and the output head. |
| `cond_concat` | Wan VAE image latent plus first-frame mask | Concatenated with `x`; combined channels produce the configured `in_dim = 36`. |
| `visual_context` | CLIP image encoder | Projected by `img_emb` and consumed by image cross-attention in each block. |
| `mouse_cond` | Condition utilities or `get_current_action()` | Consumed by `ActionModule` when the mode enables mouse. |
| `keyboard_cond` | Condition utilities or `get_current_action()` | Consumed by `ActionModule` in all modes. |
| `kv_cache*` | Pipeline cache initializers | Enables incremental causal inference over generated history. |
| `current_start` | Pipeline block pointer | Aligns RoPE and cache indices with the absolute latent-frame position. |

The distilled model configs inject `ActionModule` into blocks `0` through `14`. The foundation config includes action modules in all 30 transformer blocks.

### 9. Decoding and Output

| Mode | Path | Output Behavior |
| --- | --- | --- |
| Random-action inference | `CausalInferencePipeline.inference()` then `inference.py::generate_videos()` | Decodes every block, concatenates all video chunks, writes `demo.mp4` without overlays and `demo_icon.mp4` with overlays. |
| Streaming inference | `CausalInferenceStreamingPipeline.inference()` | Decodes every block, writes `{image_name}_current.mp4` after each block, then writes `{image_name}.mp4` and `{image_name}_icon.mp4` at the end. |
| Video rendering | `utils.visualize.process_video()` | Draws mode-specific key state, draws mouse cursor for universal mode when requested, and exports MP4 at 12 FPS. |

## Data Movement Summary

| Object | Producer | Consumer |
| --- | --- | --- |
| CLI args | `parse_args()` | `InteractiveGameInference` and output setup |
| YAML config | `OmegaConf.load(args.config_path)` | Pipeline, scheduler, model config, action mode selection |
| Distilled checkpoint | `safetensors.torch.load_file(args.checkpoint_path)` | `pipeline.generator.load_state_dict()` |
| PIL image | `load_image()` | Image preprocessing, Wan VAE encoder, CLIP encoder |
| Image tensor | `torchvision.transforms.v2` pipeline | `self.vae.encode()` and `self.vae.clip.encode_video()` |
| `img_cond` | `WanxVAEWrapper.encode()` | `cond_concat` and CausalWanModel input channels |
| `visual_context` | CLIP visual encoder | `CausalWanModel.img_emb()` and cross-attention |
| `keyboard_cond` | `utils.conditions` or `get_current_action()` | `ActionModule` and `utils.visualize.process_video()` |
| `mouse_cond` | `utils.conditions` or `get_current_action()` | `ActionModule` and mouse overlay renderer |
| Current latent block | Noise sampler and scheduler loop | `WanDiffusionWrapper.forward()` |
| Clean latent block | `WanDiffusionWrapper` x0 output | VAE decoder and future cache-update pass |
| KV/action/cross-attention caches | Pipeline cache initializers and model blocks | Later denoising blocks |
| VAE feature cache | `ZERO_VAE_CACHE` and `VAEDecoderWrapper` | Streaming VAE decoder across blocks |
| Decoded RGB chunks | `VAEDecoderWrapper.forward()` | Concatenation and MP4 rendering |
| MP4 files | `process_video()` | Written under `--output_folder` |

## Key Entry Points

- `Matrix-Game-2/inference.py`: main command-line entry point for random benchmark action trajectories.
- `Matrix-Game-2/inference_streaming.py`: console-interactive image/action entry point.
- `Matrix-Game-2/pipeline/causal_inference.py`: standard and streaming block-causal inference loops.
- `Matrix-Game-2/utils/wan_wrapper.py`: CausalWanModel and scheduler wrapper.
- `Matrix-Game-2/wan/modules/causal_model.py`: causal DiT implementation and cache-aware forward path.
- `Matrix-Game-2/wan/modules/action_module.py`: keyboard/mouse action injection.
- `Matrix-Game-2/demo_utils/vae_block3.py`: compiled streaming VAE decoder.
- `Matrix-Game-2/wan/vae/wanx_vae.py`: full Wan VAE and CLIP wrapper for image conditioning.

## Important Folders

- `configs/inference_yaml/`: per-mode inference settings, timestep schedules, and block size.
- `configs/distilled_model/`: per-mode CausalWanModel architecture JSONs.
- `configs/foundation_model/`: foundation CausalWanModel architecture JSON.
- `pipeline/`: orchestration layer for causal inference and interactive streaming inference.
- `wan/modules/`: core neural network modules, including causal transformer, action module, attention, tokenizer, CLIP, T5, and VAE components.
- `wan/vae/`: Wan VAE wrapper and source modules used to encode images and CLIP visual context.
- `utils/`: action generation, scheduler, wrapper, visualization, and miscellaneous runtime helpers.
- `demo_utils/`: streaming VAE decoder, VAE cache constants, TensorRT conversion utilities, and memory helpers.
- `demo_images/`: sample images for universal, GTA driving, and TempleRun modes.
- `assets/images/`: overlay assets used in rendered videos.
- `assets/videos/`: demo video.
- `assets/pdf/`: technical report.

## Critical Files To Check First

- `README.md`: setup, checkpoint download, and example commands.
- `inference.py`: random-action CLI and full image/action/latent preparation.
- `inference_streaming.py`: console-driven image and action input loop.
- `pipeline/causal_inference.py`: blockwise generation, cache initialization, denoising loop, and streaming output behavior.
- `utils/wan_wrapper.py`: DiT wrapper, scheduler ownership, and flow-to-x0 conversion.
- `utils/conditions.py`: mode-specific random benchmark action generation.
- `utils/visualize.py`: keyboard/mouse overlay and MP4 export.
- `utils/scheduler.py`: flow-matching timestep/sigma implementation.
- `wan/modules/causal_model.py`: CausalWanModel forward path, blockwise causal masks, and cache-aware transformer blocks.
- `wan/modules/action_module.py`: mouse/keyboard attention modules and action cache behavior.
- `wan/vae/wanx_vae.py`: VAE and CLIP model loading for image conditioning.
- `demo_utils/vae_block3.py`: cache-aware streaming decoder used during inference.
- `demo_utils/constant.py`: VAE cache tensor layout.
- `configs/inference_yaml/*.yaml`: inference schedules and mode selection.
- `configs/distilled_model/*/config.json`: mode-specific model/action dimensions and local attention size.
