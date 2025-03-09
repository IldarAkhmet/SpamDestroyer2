class BertTextPreprocess:
    def __init__(self, tokenizer=None, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def preprocess(self, text):

        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
            }