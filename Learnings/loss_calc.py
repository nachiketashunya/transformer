"""
Scenario:

Imagine you're training a machine translation model translating English sentences to French.
You have a batch size (bs) of 2, meaning you're feeding the model 2 English sentences at once.
Each sentence has a sequence length (seq_len) of 5 words.
Your vocabulary size (vocab_size) is 10,000 words.
Original Shapes:

proj_output: (bs, seq_len, vocab_size) = (2, 5, 10,000)
This represents the model's predicted probability distribution for each word in each sentence (2 sentences, each with 5 words, and each word has probabilities for 10,000 vocabulary words).
label: (bs, seq_len) = (2, 5)
This represents the actual French translation for each word in each sentence (2 sentences, each with 5 target words).
Reshaping with .view(-1, vocab_size):

Flattening: The -1 in the .view function tells PyTorch to infer the first dimension automatically based on the remaining elements. In this case, it will flatten the first two dimensions (batch size and sequence length) into a single dimension.
Combining Predictions and Labels: This essentially combines the predictions and labels for all words across all sentences in the batch into a single list.
Reshaped Shapes:

Reshaped proj_output: (bs * seq_len, vocab_size) = (10, 10,000)
This flattens the batch dimension and sequence length dimension of the original output, resulting in a tensor with 10 rows (2 sentences * 5 words each). Each row represents the predicted probability distribution for a single word (across all sentences and positions).
Reshaped label: (bs * seq_len) = (10)
Similarly, this flattens the label tensor, resulting in a list of 10 elements (2 sentences * 5 words each), where each element is the corresponding target word index from the vocabulary.
Loss Calculation:

With both reshaped tensors, the loss function (loss_fn) can efficiently compare the model's predictions (each row in the reshaped proj_output) with the corresponding target word index (each element in the reshaped label) and calculate the overall loss for the entire batch.

"""