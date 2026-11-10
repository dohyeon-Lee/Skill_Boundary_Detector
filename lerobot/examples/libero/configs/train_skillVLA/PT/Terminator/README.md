# Terminator training

Select one name or a list in `terminator_train_config.yaml`, then run
`bash submit_train.sh`. A list trains independent models concurrently on one GPU;
batch size and DataLoader workers apply to each model.

All models use Skill, current Top/Wrist images and current proprio. Each input
profile supports all four encoder variants below. The image projection, fusion
layers and termination head always train.

| Prefix | Additional input |
| --- | --- |
| Term_Token_1 | skill-start proprio + skill-end Top UV |
| Term_Token_2 | skill-start proprio |
| Term_Token_3 | skill-end Top UV |
| Term_Token_4 | none |

| Suffix | Backbone | Train backbone |
| --- | --- | --- |
| _dinoS_freeze | models/dinov3-vits16 | no |
| _dinoL_freeze | models/dinov3-vitl16 | no |
| _dinoS_unfreeze | models/dinov3-vits16 | yes |
| _dinoL_unfreeze | models/dinov3-vitl16 | yes |

For example, `Term_Token_1_dinoL_unfreeze` trains DINO-L along with the
Terminator. The default YAML selects all eight S-freeze/L-freeze models. Backbone
paths and trainability are saved in each checkpoint and restored for FT and
evaluation. Missing config/weights are detected before submission; encoder
output width is inferred from its model config.

The shared PT config selects the dataset. Each model uses its canonical skill
boundaries, Gaussian termination targets (`sigma=1`), positive weight 2, and
additional termination-positive supervision for 10 frames after each skill ends.
Global transition jitter is disabled for Terminators. Each frame is encoded
once; the same frozen image features serve current and past skill conditions.

UV variants add uniform noise of +/-0.01 metres per XYZ axis before projection
through the recorded Top camera; finite out-of-view coordinates are clipped to
the image boundary. Evaluation disables goal noise.

New names correspond to the previous `_both_proprio` indices 3, 1, 4, 5 in that
order. Historical checkpoint names, files and saved input contracts are preserved
for evaluation and FT. Retired architecture names cannot start new PT training.
