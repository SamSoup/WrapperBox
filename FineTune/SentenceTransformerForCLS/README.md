## Using Retrieval Models for Classification

Retrieval models like [SFR2](https://huggingface.co/Salesforce/SFR-Embedding-2_R) are quite useful and can be directly interagrated with wrapper box inference (e.g., KNN). To obtain a comparative inference baseline, we can implement / fine-tune a traditional classification head on top of these models in order to quickly identify what a nerual baseline of performance would look like. Note that the original retrieval model is completely frozen and therefore computation is relative cheap (e.g., only the CLS head is updated).

Turns out, there is already something like [SetFit](https://huggingface.co/docs/setfit/en/how_to/classification_heads) that tries to do this. Any `SetFit` model consists of a `SentenceTransformer` embedding model and a specified classification head (`logistic`, `differentiable`, or `custom`). I did not like their fine-tuning code, however, as they try to combine both the fine-tuning for the underlying embedding encoder + the classification head, which results in less flexability when I (most likely) only want to fine-tune the head. So I wrote it a bit more manual style. There was also a bug in `setfit` version `1.0.3` dated July 16th, 2024, where the embedding is always trained (the `end_to_end` param does not work as intended).

## Wandb Integration

When running on remote clusters, be sure to first 
`export WANDB_API_KEY=4cc6af1f2dda7fbbe9cb01789f4d7f81f5f75d4c`