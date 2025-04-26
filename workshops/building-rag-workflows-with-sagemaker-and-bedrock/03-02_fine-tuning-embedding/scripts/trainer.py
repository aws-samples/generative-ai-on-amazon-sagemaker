import os
import json
import torch
import argparse
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers

def load_data(train_file, test_file):
    train_dataset = load_dataset("json", data_dir=train_file, split="train")
    test_dataset = load_dataset("json", data_dir=test_file, split="test")
    corpus_dataset = concatenate_datasets([train_dataset, test_dataset])
    return train_dataset, test_dataset, corpus_dataset

def prepare_ir_evaluator(test_dataset, corpus_dataset, matryoshka_dimensions):
    corpus = dict(zip(corpus_dataset["id"], corpus_dataset["context"]))
    queries = dict(zip(test_dataset["id"], test_dataset["question"]))
    relevant_docs = {q_id: [q_id] for q_id in queries}

    matryoshka_evaluators = []
    for dim in matryoshka_dimensions:
        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,
            score_functions={"cosine": cos_sim},
        )
        matryoshka_evaluators.append(evaluator)

    return SequentialEvaluator(matryoshka_evaluators)

def main(args):
    print("Loading datasets...")
    train_dataset, test_dataset, corpus_dataset = load_data(args.train_data, args.validation_data)

    base_model_id_safe = args.model_name.replace("/", "_")
    output_dir = f"{args.model_output}/{base_model_id_safe}_ds={len(train_dataset)}_bs={args.batch_size}_e={args.epochs}"

    print("Loading model...")
    model = SentenceTransformer(
        args.model_name,
        model_kwargs={"attn_implementation": "eager"},
        trust_remote_code=True
    )

    print("Preparing loss function...")
    model_dim = model.get_sentence_embedding_dimension()
    matryoshka_dimensions = [dim for dim in [768, 512, 384, 256, 128, 64] if dim <= model_dim]
    # matryoshka_dimensions = [768, 512, 256, 128, 64]
    inner_train_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(model, inner_train_loss, matryoshka_dims=matryoshka_dimensions)

    print("Configuring evaluator...")
    evaluator = prepare_ir_evaluator(test_dataset, corpus_dataset, matryoshka_dimensions)

    print("Setting training arguments...")
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=0.1,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        tf32=True,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_dim_128_cosine_ndcg@10",
    )

    print("Starting training...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset.select_columns(["question", "context"]),
        loss=train_loss,
        evaluator=evaluator,
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--validation_data", type=str, default="/opt/ml/input/data/validation")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_output", type=str, default="/opt/ml/model")
    args = parser.parse_args()

    main(args)
