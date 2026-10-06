# ISeeShip Dataset

ISeeShip is a multi-task ship visual-intelligence benchmark covering classification, detection, and open-set recognition.

## Image Download

The public images are hosted at [Ship30/dataset](https://github.com/lxixim/Ship30/tree/main/dataset).

```bash
git clone https://github.com/lxixim/Ship30.git /path/to/Ship30
```

Images are organized by category under `/path/to/Ship30/dataset`:

```text
dataset/
  aircraft_carrier/
  container_ship/
  LNG_tanker/
  ...
```

Keep the original class names and capitalization. The classification label list is available in [ships.txt](../classification/val_data/ships.txt).

## Record Format

Data-construction tools use an image, a task prompt, and a target response:

```json
{
  "image": "/path/to/image.jpg",
  "problem": "Identify the ship category.",
  "solution": "<think>Visual feature description</think><answer>container_ship</answer>"
}
```

The example shows the schema. Depending on the converter, records are stored as JSONL or Hugging Face datasets. Use the prepared dataset format expected by the training entry point.

For detection, training answers use `Position: [x1, y1, x2, y2]` scaled to 0-1000 and `Confidence` in 0-1. COCO evaluation uses pixel-coordinate `[x, y, width, height]` boxes with image/category IDs.

## Preparation Tools

| Tool | Purpose |
| --- | --- |
| `ship_data_build/xml2coco_ship.py` | Convert supplied VOC annotations to COCO |
| `ship_data_build/voc2rft_detection.py` | Convert COCO annotations to category-conditioned training records |
| `ship_data_build/create_multi_shot_datasets.py` | Prepare support sets using image root, shot count, and seed |
| `ship_data_build/build_openset_dataset_with_think.py` | Construct open-set records using the configured class partitions |
| `ship_data_build/generate_cot_for_dataset.py` | Generate reference reasoning text |
| `dataset/build_dataset.ipynb` | Notebook-based data processing |

Set input/output paths before running a construction utility. Use the same support image IDs and evaluation lists across methods in a controlled comparison, and keep training and evaluation memberships separate.

## Paper Settings

The main benchmark uses 4-shot and 8-shot adaptation. The four-shot controlled classification experiment uses 120 support images and 2,173 evaluation images; the controlled detection experiment uses 56 support images and 3,886 category-conditioned evaluation queries. The main seed is 100; the repeated controlled comparisons use seeds 42, 43, and 44.
