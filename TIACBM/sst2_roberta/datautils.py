import os
import re
from functools import partial
import datasets

class BertDataProcessor():
    def __init__(self, hf_dset, hf_tokenizer, max_length, text_col='sentence', label_col='label', lines_delimiter='\n', sentiword=None):
        self.hf_tokenizer = hf_tokenizer
        self._max_length = max_length
        self.sentiwordtree = sentiword
        self.hf_dset = hf_dset
        self.text_col = text_col
        self.label_col = label_col

    def map(self, **kwargs) -> datasets.arrow_dataset.Dataset:
        num_proc = kwargs.pop('num_proc', os.cpu_count())
        num_proc = 2
        cache_file_name = kwargs.pop('cache_file_name', None)
        if cache_file_name is not None:
            if not cache_file_name.endswith('.arrow'):
                cache_file_name += '.arrow'
            if '/' not in cache_file_name:
                cache_dir = os.path.abspath(os.path.dirname(self.hf_dset.cache_files[0]['filename']))
                cache_file_name = os.path.join(cache_dir, cache_file_name)

        return self.hf_dset.map(
            function=self,
            batched=True,
            cache_file_name=cache_file_name,
            remove_columns=self.hf_dset.column_names,
            disable_nullable=True,
            writer_batch_size=10**4,
            input_columns=['sentence','label'],
            num_proc=num_proc,
            **kwargs
        )

    def __call__(self, texts, labels):
        new_example = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'word_indices': [],
            'labels': []
        }
        for text, label in zip(texts, labels):  # For every doc with its label
            example = self.process_text(text)
            if example:
                for k, v in example.items():
                    new_example[k].append(v)
                new_example['labels'].append(label)  # Append label after the example is created
        return new_example

    def process_text(self, text):
        """Tokenizes the text and calculates word indices based on the sentiment dictionary."""
        # Tokenize the text
        encoding = self.hf_tokenizer.encode_plus(text, max_length=self._max_length, padding='max_length'
                                                 , truncation=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        token_type_ids = encoding['token_type_ids']
        tokens = self.hf_tokenizer.convert_ids_to_tokens(input_ids)

        # Calculate word indices
        word_indices = {}
        for word in self.sentiwordtree:
            indices = self.get_word_token_indices(tokens, word)
            if indices:
                word_indices[word] = indices

        # Create the example with input_ids, attention_mask, token_type_ids, and word_indices
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'word_indices': word_indices
        }

    def get_word_token_indices(self, tokens, word):
        """Finds the indices of the tokens corresponding to a specific word."""
        word_tokens = self.hf_tokenizer.tokenize(word)
        word_len = len(word_tokens)
        indices = []
        for i in range(len(tokens) - word_len + 1):
            if tokens[i:i + word_len] == word_tokens:
                indices.append(list(range(i, i + word_len)))
        return indices