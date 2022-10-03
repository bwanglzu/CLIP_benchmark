"""Console script for clip_benchmark."""
import argparse
import sys
import json
import torch
import open_clip

from clip_benchmark.datasets.builder import build_dataset, get_dataset_collate_fn
from clip_benchmark.metrics import zeroshot_retrieval

from torch.utils.data import default_collate

from .rebuild import image_encoder, text_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="cifar10",
                        help="Dataset to use for the benchmark")
    parser.add_argument('--split', type=str, default="test",
                        help="Dataset split to use")
    parser.add_argument('--pretrained', type=str, default="laion400m_e32",
                        help="Model checkpoint name to use from OpenCLIP")
    parser.add_argument('--task', type=str, default="zeroshot_retrieval",
                        choices=["zeroshot_classification", "zeroshot_retrieval"])
    parser.add_argument('--amp', default=True, action="store_true",
                        help="whether to use mixed precision")
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--recall_k', default=[5], type=int,
                        help="for retrieval, select the k for Recall@K metric. ",
                        nargs="+", )
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dataset_root', default="root", type=str,
                        help="dataset root folder where the datasets are downloaded.")
    parser.add_argument('--annotation_file', default="", type=str,
                        help="text annotation file for retrieval datasets. Only needed  for when `--task` is `zeroshot_retrieval`.")
    parser.add_argument('--output', default="result.json", type=str,
                        help="output file where to dump the metrics")
    parser.add_argument('--verbose', default=False, action="store_true",
                        help="verbose mode")
    args = parser.parse_args()
    run(args)


def run(args):
    """Console script for clip_benchmark."""
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, transform = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    image_encoder = image_encoder.to(args.device)
    text_encoder = text_encoder.to(args.device)
    dataset = build_dataset(
        dataset_name=args.dataset,
        root=args.dataset_root,
        transform=transform,
        split=args.split,
        annotation_file=args.annotation_file,
        download=True,
    )
    collate_fn = get_dataset_collate_fn(args.dataset)
    if args.verbose:
        print(f"Dataset size: {len(dataset)}")
        print(f"Dataset split: {args.split}")
        print(f"Dataset classes: {dataset.classes}")
        print(f"Dataset number of classes: {len(dataset.classes)}")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    metrics = zeroshot_retrieval.evaluate(
        image_encoder,
        text_encoder,
        dataloader,
        open_clip.tokenizer.tokenize,
        recall_k_list=args.recall_k,
        device=args.device,
        amp=args.amp
    )
    dump = {
        "dataset": args.dataset,
        "model": args.model,
        "pretrained": args.pretrained,
        "task": args.task,
        "metrics": metrics
    }
    with open(args.output, "w") as f:
        json.dump(dump, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
