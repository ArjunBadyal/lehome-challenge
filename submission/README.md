# LeHome Challenge Submission

## Evaluate The Policy

The submitted policy is hosted as a public Docker image:

```text
docker.io/arjunbadyal/lehome-submission:v7
```

Image digest:

```text
sha256:4d4e9d8e1e0dddbb8578fe8dc719aee04d5b054f4532367870bde3306ba9b66a
```

No credentials are required.

Pull and run the policy server:

```bash
docker pull docker.io/arjunbadyal/lehome-submission:v7
docker run --rm -p 8080:8080 docker.io/arjunbadyal/lehome-submission:v7
```

In another terminal, evaluate with the challenge Docker policy interface, for example:

```bash
python -m scripts.eval \
    --policy_type docker \
    --garment_type top_short \
    --num_episodes 5 \
    --max_steps 600 \
    --enable_cameras \
    --device cpu \
    --headless
```

Change `--garment_type` to evaluate the other categories:

```text
top_long
top_short
pant_long
pant_short
```

The server implements the standard HTTP policy interface:

```text
POST /reset
POST /infer
```

`/reset` clears episode state. `/infer` returns one 12-dimensional robot action.

## Method

This submission uses ACT imitation policies with a visual garment classifier and a small action stabilizer.

At the start of each episode, a ResNet-18 classifier predicts the garment category from the public top-down RGB observation. The policy then selects the corresponding ACT policy:

```text
Top-Short:  outputs/train/act_top_short_aug/checkpoints/055000/pretrained_model
Top-Long:   outputs/train/act_top_long_aug/checkpoints/090000/pretrained_model
Pant-Long:  outputs/train/act_pant_long_balanced_weak_tiny/checkpoints/090250/pretrained_model
Pant-Short: outputs/train/act_pant_short_aug/checkpoints/060000/pretrained_model
```

A unified ACT policy trained on the combined four-category dataset is included as a fallback if classification or specialist loading fails.

After the ACT action is produced, the stabilizer smooths consecutive actions by limiting how much each joint command can change between policy steps. For Top-Long and Pant-Short, it also delays unsafe gripper opening while the arms are moving quickly.

The classifier, ACT policies, fallback policy, and stabilizer use only the public observation dictionary. The submission does not read garment filenames, simulator-provided category labels, checkpoint particle positions, success-condition margins, rewards, task metrics, or any other simulator-internal state at inference time.

## Docker Image Contents

The Docker image contains the inference runtime, policy code, checkpoints, classifier, and required dataset metadata:

```text
/app/
  server.py
  policy.py
  policies/
    classifier/
    specialists/
    unified/
    datasets/
```

The image does not include training datasets, experiment logs, or research scripts. Dataset `meta/` directories are included because LeRobot requires them to load policy metadata.

## Rollout Results

Self-reported rollout results are provided in:

```text
rollout_results.txt
```

They were measured with:

```text
--max_steps 600
--num_episodes 5
--device cpu
--enable_cameras
```

The final packaged result was:

```text
153 / 240 = 63.75%
```
