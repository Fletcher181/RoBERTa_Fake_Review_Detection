# STEP 21: IMPROVED — Multi-seed evaluation for statistical reliability
# Run the full pipeline across 3 seeds and report mean ± std
# This is required for any research paper claim about improvement
#
# NOTE: This cell re-trains from scratch for each seed.
#       On GPU it takes ~2–3x the original training time.
#       Comment out if you only need a single run.

from transformers import RobertaConfig, RobertaForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score

MULTI_SEEDS = [42, 123, 456]
seed_results = []

for run_seed in MULTI_SEEDS:
    print(f"\n{'='*50}")
    print(f"Running seed {run_seed}...")
    print('='*50)

    # Reproducibility
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    torch.cuda.manual_seed_all(run_seed)
    set_seed(run_seed)

    # Fresh splits with this seed
    tr_texts, tmp_texts, tr_labels, tmp_labels = train_test_split(
        df["review"], df["label"],
        test_size=0.30, stratify=df["label"], random_state=run_seed
    )
    v_texts, te_texts, v_labels, te_labels = train_test_split(
        tmp_texts, tmp_labels,
        test_size=0.50, stratify=tmp_labels, random_state=run_seed
    )

    tr_enc = tokenizer(list(tr_texts), truncation=True, padding=True, max_length=MAX_LENGTH)
    v_enc  = tokenizer(list(v_texts),  truncation=True, padding=True, max_length=MAX_LENGTH)
    te_enc = tokenizer(list(te_texts), truncation=True, padding=True, max_length=MAX_LENGTH)

    tr_ds = ReviewDataset(tr_enc, tr_labels)
    v_ds  = ReviewDataset(v_enc,  v_labels)
    te_ds = ReviewDataset(te_enc, te_labels)

    cw_np = compute_class_weight("balanced", classes=np.unique(tr_labels), y=list(tr_labels))
    cw    = torch.tensor(cw_np, dtype=torch.float)

    run_config = RobertaConfig.from_pretrained(
        "roberta-base", num_labels=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        classifier_dropout=0.1
    )
    run_model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", config=run_config, ignore_mismatched_sizes=True
    )

    run_args = TrainingArguments(
        output_dir=f"./proposed_seed_{run_seed}",
        eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="f1", greater_is_better=True,
        learning_rate=1e-5, lr_scheduler_type="cosine", warmup_ratio=0.1,
        per_device_train_batch_size=16, per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=10, weight_decay=0.01,
        fp16=USE_FP16, logging_strategy="epoch",
        report_to="none", seed=run_seed, data_seed=run_seed,
    )

    run_trainer = WeightedTrainer(
        model=run_model, args=run_args,
        train_dataset=tr_ds, eval_dataset=v_ds,
        compute_metrics=compute_metrics,
        class_weights=cw,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    run_trainer.train()

    run_preds  = run_trainer.predict(te_ds)
    run_y_pred = run_preds.predictions.argmax(axis=1)
    run_y_true = run_preds.label_ids

    seed_results.append({
        "seed":      run_seed,
        "accuracy":  accuracy_score(run_y_true, run_y_pred),
        "f1":        f1_score(run_y_true, run_y_pred, average="binary"),
    })
    print(f"  Seed {run_seed} — Acc: {seed_results[-1]['accuracy']:.4f}  F1: {seed_results[-1]['f1']:.4f}")

# Summary
accs = [r["accuracy"] for r in seed_results]
f1s  = [r["f1"]       for r in seed_results]

print(f"\n{'='*50}")
print(f"Multi-seed Summary ({len(MULTI_SEEDS)} runs)")
print(f"{'='*50}")
print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"F1 Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"\n(Report these numbers in your paper, not single-run values)")



# STEP 22: Save best model and tokenizer
import os

save_path = "./proposed_model_final"
os.makedirs(save_path, exist_ok=True)

trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

print(f"Model saved to: {save_path}")
print(f"Files: {os.listdir(save_path)}")