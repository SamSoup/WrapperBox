from torch.utils.data import Dataset


class PromptWithLabelsDataset(Dataset):
    def __init__(self, texts, labels, prompt: str):
        self.texts = texts
        self.labels = labels
        self.prompt = prompt

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.prompt.format(input=self.texts[idx]),
            "labels": self.labels[idx],
        }


class PromptDataset(Dataset):
    def __init__(self, texts, prompt):
        self.texts = texts
        self.prompt = prompt

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.prompt.format(input=self.texts[idx])}
