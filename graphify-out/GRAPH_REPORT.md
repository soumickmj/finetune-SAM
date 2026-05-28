# Graph Report - /group/glastonbury/soumick/codebase/finetune-SAM  (2026-05-28)

## Corpus Check
- 115 files · ~86,858 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 865 nodes · 1777 edges · 80 communities (28 shown, 52 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 268 edges (avg confidence: 0.54)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_SAM Adapter and Normalisation Layers|SAM Adapter and Normalisation Layers]]
- [[_COMMUNITY_EfficientNet Backbone|EfficientNet Backbone]]
- [[_COMMUNITY_Distributed Training Pipeline|Distributed Training Pipeline]]
- [[_COMMUNITY_Automatic Mask Generation|Automatic Mask Generation]]
- [[_COMMUNITY_SAM Encoder-Decoder Architecture|SAM Encoder-Decoder Architecture]]
- [[_COMMUNITY_UNet Backbone|UNet Backbone]]
- [[_COMMUNITY_Tag-Based Encoder-Decoder|Tag-Based Encoder-Decoder]]
- [[_COMMUNITY_Segmentation Loss Functions|Segmentation Loss Functions]]
- [[_COMMUNITY_Prompt-Based Mask Prediction|Prompt-Based Mask Prediction]]
- [[_COMMUNITY_Auxiliary Losses and Utilities|Auxiliary Losses and Utilities]]
- [[_COMMUNITY_Medical Imaging Datasets|Medical Imaging Datasets]]
- [[_COMMUNITY_Project Documentation and Config|Project Documentation and Config]]
- [[_COMMUNITY_ResNet Backbone Blocks|ResNet Backbone Blocks]]
- [[_COMMUNITY_SE-ResNet Squeeze-Excitation Blocks|SE-ResNet Squeeze-Excitation Blocks]]
- [[_COMMUNITY_UNet Components|UNet Components]]
- [[_COMMUNITY_GAN Discriminator|GAN Discriminator]]
- [[_COMMUNITY_Evaluation and Metrics|Evaluation and Metrics]]
- [[_COMMUNITY_LoRA Fine-tuning Layers|LoRA Fine-tuning Layers]]
- [[_COMMUNITY_SAM Model and ONNX Export|SAM Model and ONNX Export]]
- [[_COMMUNITY_Running Statistics|Running Statistics]]
- [[_COMMUNITY_Variational Autoencoder|Variational Autoencoder]]
- [[_COMMUNITY_VGG Backbone|VGG Backbone]]
- [[_COMMUNITY_Implicit Neural Network|Implicit Neural Network]]
- [[_COMMUNITY_LR Scheduling and Module Hooks|LR Scheduling and Module Hooks]]
- [[_COMMUNITY_SqueezeNet Backbone|SqueezeNet Backbone]]
- [[_COMMUNITY_Segmentation Evaluation|Segmentation Evaluation]]
- [[_COMMUNITY_DDP GPU Demo Script|DDP GPU Demo Script]]
- [[_COMMUNITY_Single GPU Box Demo Script|Single GPU Box Demo Script]]
- [[_COMMUNITY_Single GPU Demo Script|Single GPU Demo Script]]
- [[_COMMUNITY_LoRA Mobile Demo Script|LoRA Mobile Demo Script]]
- [[_COMMUNITY_Plasma Cells Training Script|Plasma Cells Training Script]]
- [[_COMMUNITY_UKBB Abdomen Training Script|UKBB Abdomen Training Script]]
- [[_COMMUNITY_UKBB Abdomen Exec Script|UKBB Abdomen Exec Script]]
- [[_COMMUNITY_Validation Demo Script|Validation Demo Script]]
- [[_COMMUNITY_Mobile Adapter Demo Script|Mobile Adapter Demo Script]]
- [[_COMMUNITY_Plasma Cells Array Script|Plasma Cells Array Script]]
- [[_COMMUNITY_IBD Array Training Script|IBD Array Training Script]]
- [[_COMMUNITY_Smile Dataset Script|Smile Dataset Script]]
- [[_COMMUNITY_VS Code Launch Config|VS Code Launch Config]]
- [[_COMMUNITY_Plasma Cells Crop Script|Plasma Cells Crop Script]]
- [[_COMMUNITY_UKBB Abd 204 All Script|UKBB Abd 204 All Script]]
- [[_COMMUNITY_UKBB Abd 204 Liver Script|UKBB Abd 204 Liver Script]]
- [[_COMMUNITY_UKBB Abd 254 Liver Script|UKBB Abd 254 Liver Script]]
- [[_COMMUNITY_UKBB Abd 259 All Script|UKBB Abd 259 All Script]]
- [[_COMMUNITY_UKBB Abd 259 No-liver Script|UKBB Abd 259 No-liver Script]]
- [[_COMMUNITY_UKBB Abd 260 Kidney Script|UKBB Abd 260 Kidney Script]]
- [[_COMMUNITY_UKBB Abd Exec Script|UKBB Abd Exec Script]]
- [[_COMMUNITY_Plasma Cells Box Script|Plasma Cells Box Script]]
- [[_COMMUNITY_Plasma Cells Exec Script|Plasma Cells Exec Script]]
- [[_COMMUNITY_Plasma Cells V2 Script|Plasma Cells V2 Script]]
- [[_COMMUNITY_IBD Difficult Array Script|IBD Difficult Array Script]]
- [[_COMMUNITY_IBD Easy Array Script|IBD Easy Array Script]]
- [[_COMMUNITY_IBD Array Script|IBD Array Script]]
- [[_COMMUNITY_Smile Exec Script|Smile Exec Script]]
- [[_COMMUNITY_UKBB Abd 204 Aorta Script|UKBB Abd 204 Aorta Script]]
- [[_COMMUNITY_UKBB Abd 204 Spleen Script|UKBB Abd 204 Spleen Script]]
- [[_COMMUNITY_UKBB Abd 254 All Script|UKBB Abd 254 All Script]]
- [[_COMMUNITY_UKBB Abd 254 Aorta Script|UKBB Abd 254 Aorta Script]]
- [[_COMMUNITY_UKBB Abd 254 Spleen Script|UKBB Abd 254 Spleen Script]]
- [[_COMMUNITY_UKBB Abd 259 Aorta Script|UKBB Abd 259 Aorta Script]]
- [[_COMMUNITY_UKBB Abd 259 Kidney Script|UKBB Abd 259 Kidney Script]]
- [[_COMMUNITY_UKBB Abd 259 Pancreas Script|UKBB Abd 259 Pancreas Script]]
- [[_COMMUNITY_UKBB Abd 259 Spleen Script|UKBB Abd 259 Spleen Script]]
- [[_COMMUNITY_UKBB Abd 260 All Script|UKBB Abd 260 All Script]]
- [[_COMMUNITY_UKBB Abd 260 Aorta Script|UKBB Abd 260 Aorta Script]]
- [[_COMMUNITY_UKBB Abd 260 No-liver Script|UKBB Abd 260 No-liver Script]]
- [[_COMMUNITY_UKBB Abd 260 Pancreas Script|UKBB Abd 260 Pancreas Script]]
- [[_COMMUNITY_UKBB Abd 260 Spleen Script|UKBB Abd 260 Spleen Script]]
- [[_COMMUNITY_UKBB Abd All Array Script|UKBB Abd All Array Script]]
- [[_COMMUNITY_Plasma Cells Infer Script|Plasma Cells Infer Script]]
- [[_COMMUNITY_Plasma Cells Infer Edit Script|Plasma Cells Infer Edit Script]]
- [[_COMMUNITY_Arg Parsing Test Script|Arg Parsing Test Script]]
- [[_COMMUNITY_VS Code Python Settings|VS Code Python Settings]]
- [[_COMMUNITY_VS Code Workspace Settings|VS Code Workspace Settings]]

## God Nodes (most connected - your core abstractions)
1. `Adapter` - 49 edges
2. `LayerNorm2d` - 49 edges
3. `MLPBlock` - 40 edges
4. `ResizeLongestSide` - 32 edges
5. `SamPredictor` - 29 edges
6. `PromptEncoder` - 25 edges
7. `MaskDecoder` - 22 edges
8. `DiceLoss` - 21 edges
9. `Public_dataset` - 20 edges
10. `ImageEncoderViT` - 20 edges

## Surprising Connections (you probably didn't know these)
- `Finetune SAM README` --references--> `Fine-tuning Strategy Overview by Dataset Availability (v9)`  [EXTRACTED]
  README.md → finetune_strategy_v9.png
- `Python Requirements (finetune-SAM)` --semantically_similar_to--> `Conda Environment (finetune-SAM)`  [INFERRED] [semantically similar]
  requirements.txt → environment.yml
- `Fine-tuning Strategy Overview by Dataset Availability (v9)` --references--> `SSL-SAM Pretrained Weights`  [INFERRED]
  finetune_strategy_v9.png → pretrained_weights/SSLSAM/source.txt
- `WarmUpLR` --uses--> `Discriminator`  [INFERRED]
  utils/utils.py → models/discriminator.py
- `WarmUpLR` --uses--> `ResizeLongestSide`  [INFERRED]
  utils/utils.py → models/sam/utils/transforms.py

## Hyperedges (group relationships)
- **SAM Variant Foundation Models for Medical Imaging** — weights_SAM, weights_medSAM, weights_mobileSAM, weights_sslSAM, weights_pathoSAM, weights_medicoSAM, weights_mriFoundation [INFERRED 0.88]
- **PEFT and Vanilla Fine-tuning Strategy Space (18 Combinations)** — method_adapter_finetune, method_lora_finetune, method_vanilla_finetune, img_finetuneCombinationV3 [EXTRACTED 0.95]
- **Setup 3 Self-supervised Pretraining then Supervised Fine-tuning Pipeline** — weights_sslSAM, img_finetuneStrategyV9, readme_finetuneSAM [INFERRED 0.82]

## Communities (80 total, 52 thin omitted)

### Community 0 - "SAM Adapter and Normalisation Layers"
Cohesion: 0.05
Nodes (49): Adapter, LayerNorm2d, MLPBlock, add_decomposed_rel_pos(), Attention, Block, closest_numbers(), get_rel_pos() (+41 more)

### Community 1 - "EfficientNet Backbone"
Cohesion: 0.06
Nodes (26): EfficientNet, MBConvBlock, MBConvBlock_freeze, EfficientNet, MBConvBlock, BlockDecoder, Conv2dDynamicSamePadding, Conv2dStaticSamePadding (+18 more)

### Community 2 - "Distributed Training Pipeline"
Cohesion: 0.08
Nodes (31): cleanup(), model_basic(), setup(), train_model(), train_model(), train_model(), main(), main() (+23 more)

### Community 3 - "Automatic Mask Generation"
Cohesion: 0.08
Nodes (41): ItemsView, MaskData, Any, float, int, ndarray, Sam, str (+33 more)

### Community 4 - "SAM Encoder-Decoder Architecture"
Cohesion: 0.09
Nodes (29): ImageEncoderViT, MaskDecoder, ImageEncoderViT, MaskDecoder, PatchEmbed, PositionEmbeddingRandom, PromptAutoEncoder, PromptEncoder (+21 more)

### Community 5 - "UNet Backbone"
Cohesion: 0.09
Nodes (17): Stage, BasicBlock, Bottleneck, conv3x3(), ResNet, resnet101(), resnet152(), resnet18() (+9 more)

### Community 6 - "Tag-Based Encoder-Decoder"
Cohesion: 0.08
Nodes (15): Decoder, Encoder, AnyAttention, apply_pos(), FullRelPos, Mlp, SimpleReasoning, PatchEmbed (+7 more)

### Community 7 - "Segmentation Loss Functions"
Cohesion: 0.11
Nodes (14): _AbstractDiceLoss, BCEDiceLoss, compute_per_channel_dice(), _create_loss(), DiceLoss, flatten(), GeneralizedDiceLoss, get_loss_criterion() (+6 more)

### Community 8 - "Prompt-Based Mask Prediction"
Cohesion: 0.12
Nodes (15): MLP, SmallDecoder, FeedForward, MultiHeadAttention, PatchEmbedding, ResidualBlock, TransformerEncoder, TransformerEncoderBlock (+7 more)

### Community 9 - "Auxiliary Losses and Utilities"
Cohesion: 0.13
Nodes (8): cppn(), gene_out(), get_network(), get_siren(), para_image(), raw_out(), siren(), to_valid_out()

### Community 10 - "Medical Imaging Datasets"
Cohesion: 0.16
Nodes (8): Dataset, Public_H5dataset, recursive_read_h5(), get_first_prompt(), get_top_boxes(), MaskToBoxSimple(), random_sum_to(), read_h5_data()

### Community 11 - "Project Documentation and Config"
Cohesion: 0.22
Nodes (17): UKBB F20204 Liver Imaging Dataset, Conda Environment (finetune-SAM), Fine-tuning Architecture Combination Diagram (v3), Fine-tuning Strategy Overview by Dataset Availability (v9), Adapter Fine-tuning (PEFT), LoRA Fine-tuning (PEFT), Vanilla Fine-tuning, Finetune SAM README (+9 more)

### Community 12 - "ResNet Backbone Blocks"
Cohesion: 0.19
Nodes (8): BasicBlock, BottleNeck, ResNet, resnet101(), resnet152(), resnet18(), resnet34(), resnet50()

### Community 13 - "SE-ResNet Squeeze-Excitation Blocks"
Cohesion: 0.19
Nodes (8): BasicResidualSEBlock, BottleneckResidualSEBlock, SEResNet, seresnet101(), seresnet152(), seresnet18(), seresnet34(), seresnet50()

### Community 14 - "UNet Components"
Cohesion: 0.22
Nodes (4): DoubleConv, Down, OutConv, Up

### Community 15 - "GAN Discriminator"
Cohesion: 0.23
Nodes (9): BinaryIO, Discriminator, Text, make_grid(), bool, int, str, Tensor (+1 more)

### Community 16 - "Evaluation and Metrics"
Cohesion: 0.21
Nodes (11): calculate_gradient_penalty(), DiceCoeff, export(), gram_matrix(), hook_model(), inverse_normalize(), pre_d(), render_vis() (+3 more)

### Community 17 - "LoRA Fine-tuning Layers"
Cohesion: 0.23
Nodes (5): _LoRA_qkv, _LoRA_qkv_proj, int, Module, Sam

### Community 18 - "SAM Model and ONNX Export"
Cohesion: 0.33
Nodes (5): bool, int, Sam, Tensor, SamOnnxModel

### Community 21 - "VGG Backbone"
Cohesion: 0.44
Nodes (6): make_layers(), VGG, vgg11_bn(), vgg13_bn(), vgg16_bn(), vgg19_bn()

### Community 23 - "LR Scheduling and Module Hooks"
Cohesion: 0.25
Nodes (3): _LRScheduler, ModuleHook, WarmUpLR

### Community 25 - "Segmentation Evaluation"
Cohesion: 0.33
Nodes (5): array, CompositeActivation, dice_coeff(), eval_seg(), iou()

## Knowledge Gaps
- **70 isolated node(s):** `exec_traintest_singlegpu_AlexPlasmaCells.sh script`, `exec_traintest_singlegpu_UKBAbd_array_260_pancreas.sh script`, `train_singlegpu_demo_adapter_mobile.sh script`, `CUDA_VISIBLE_DEVICES`, `exec_traintest_singlegpu_UKBAbd_array_260_aorta.sh script` (+65 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **52 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Block` connect `SAM Adapter and Normalisation Layers` to `EfficientNet Backbone`, `UNet Backbone`, `ResNet Backbone Blocks`, `SE-ResNet Squeeze-Excitation Blocks`?**
  _High betweenness centrality (0.251) - this node is a cross-community bridge._
- **Why does `LayerNorm2d` connect `SAM Adapter and Normalisation Layers` to `Prompt-Based Mask Prediction`, `SAM Encoder-Decoder Architecture`?**
  _High betweenness centrality (0.143) - this node is a cross-community bridge._
- **Why does `_build_sam()` connect `SAM Encoder-Decoder Architecture` to `SAM Adapter and Normalisation Layers`, `Distributed Training Pipeline`?**
  _High betweenness centrality (0.142) - this node is a cross-community bridge._
- **Are the 39 inferred relationships involving `Adapter` (e.g. with `Attention` and `Block`) actually correct?**
  _`Adapter` has 39 INFERRED edges - model-reasoned connections that need verification._
- **Are the 42 inferred relationships involving `LayerNorm2d` (e.g. with `Attention` and `Block`) actually correct?**
  _`LayerNorm2d` has 42 INFERRED edges - model-reasoned connections that need verification._
- **Are the 32 inferred relationships involving `MLPBlock` (e.g. with `Attention` and `Block`) actually correct?**
  _`MLPBlock` has 32 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `ResizeLongestSide` (e.g. with `array` and `BinaryIO`) actually correct?**
  _`ResizeLongestSide` has 13 INFERRED edges - model-reasoned connections that need verification._