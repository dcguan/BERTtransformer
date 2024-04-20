# BERTtransformer
CS 224N project: BERT transformer model to perform 3 different tasks of sentiment classification, paraphrase prediction, and semantic similarity 
1. Implemented the minBERT transformer model from scratch to perform 3 NLP tasks: sentiment classification, semantic similarity, and paraphrase detection.
2. Built a custom implementation of LoRA, a parameter-efficient alternative to fine-tuning, and applied it on top of frozen minBERT parameters. 
3. Experimented with many LoRA configurations and identified a set of hyperparameters which outperformed baseline minBERT model after PEFT with LoRA.
