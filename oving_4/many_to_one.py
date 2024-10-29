import torch
import torch.nn as nn

# Define a Long Short-Term Memory (LSTM) model to map character sequences to emojis
class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size, emoji_encoding_size):
        super(LongShortTermMemoryModel, self).__init__()

        # LSTM layer with input size as encoding_size and hidden state size of 128
        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        # Dense (fully connected) layer to map the LSTM output to the emoji encoding space
        self.dense = nn.Linear(128, emoji_encoding_size)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 1, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

if __name__ == '__main__':
    SEARCH_WORD = ' ct ' # The word to test with the mode

    print("The word is {}:".format(SEARCH_WORD))

    # Input: One-hot encodings for each character
    char_enc = [
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # ' ' 0 (space)
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'h' 1
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'a' 2
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 't' 3
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'r' 4
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],  # 'c' 5
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  # 'f' 6
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 'l' 7
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # 'm' 8
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],  # 'p' 9
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],  # 's' 10
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],  # 'o' 11
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]   # 'n' 12
    ]
    char_encoding_size = len(char_enc) # Length of character encoding (13)
    index_to_char = [' ', 'h', 'a', 't', 'r',
                     'c', 'f', 'l', 'm', 'p', 's', 'o', 'n']

    # Output
    emojis = {
        'hat': '\U0001F3A9', # 🎩
        'cat': '\U0001F408', # 🐈
        'rat': '\U0001F400', # 🐀
        'flat': '\U0001F3E2', # 🏢
        'matt': '\U0001F468', # 👨
        'cap': '\U0001F9E2', # 🧢
        'son': '\U0001F466' # 👦
    }

    # One-hot encoding for each emoji
    emoji_enc = [
        [1., 0., 0., 0., 0., 0., 0.],  # 'hat' 0
        [0., 1., 0., 0., 0., 0., 0.],  # 'rat' 1
        [0., 0., 1., 0., 0., 0., 0.],  # 'cat' 2
        [0., 0., 0., 1., 0., 0., 0.],  # 'flat' 3
        [0., 0., 0., 0., 1., 0., 0.],  # 'matt' 4
        [0., 0., 0., 0., 0., 1., 0.],  # 'cap' 5
        [0., 0., 0., 0., 0., 0., 1.]   # 'son' 6
    ]

    emoji_encoding_size = len(emoji_enc)
    index_to_emoji = [emojis['hat'], emojis['rat'], emojis['cat'],
                      emojis['flat'], emojis['matt'], emojis['cap'], emojis['son']]

    # Training data: sequences of one-hot encoded characters
    x_train = torch.tensor([
        [[char_enc[1]], [char_enc[2]], [char_enc[3]], [char_enc[0]]], # 'hat '
        [[char_enc[4]], [char_enc[2]], [char_enc[3]], [char_enc[0]]], # 'rat '
        [[char_enc[5]], [char_enc[2]], [char_enc[3]], [char_enc[0]]], # 'cat '
        [[char_enc[6]], [char_enc[7]], [char_enc[2]], [char_enc[3]]], # 'flat'
        [[char_enc[8]], [char_enc[2]], [char_enc[3]], [char_enc[3]]], # 'matt'
        [[char_enc[5]], [char_enc[2]], [char_enc[9]], [char_enc[0]]], # 'cap '
        [[char_enc[10]], [char_enc[11]], [char_enc[12]], [char_enc[0]]]])  # 'son '

    # Corresponding emoji encodings as labels
    y_train = torch.tensor([
        [emoji_enc[0], emoji_enc[0], emoji_enc[0], emoji_enc[0]], # 'hat'
        [emoji_enc[1], emoji_enc[1], emoji_enc[1], emoji_enc[1]], # 'rat'
        [emoji_enc[2], emoji_enc[2], emoji_enc[2], emoji_enc[2]], # 'cat'
        [emoji_enc[3], emoji_enc[3], emoji_enc[3], emoji_enc[3]], # 'flat'
        [emoji_enc[4], emoji_enc[4], emoji_enc[4], emoji_enc[4]], # 'matt'
        [emoji_enc[5], emoji_enc[5], emoji_enc[5], emoji_enc[5]], # 'cap'
        [emoji_enc[6], emoji_enc[6], emoji_enc[6], emoji_enc[6]]]) # 'son'

    model = LongShortTermMemoryModel(char_encoding_size, emoji_encoding_size)

    # Function to generate an emoji based on the input string
    def generate(string):
        model.reset()
        for i in range(len(string)): # For each character in the input string
            char_index = index_to_char.index(string[i]) # Get index of the character
            y = model.f(torch.tensor([[char_enc[char_index]]])) # Get model prediction
            if i == len(string) - 1: # If it's the last character, print the predicted emoji
                print(index_to_emoji[y.argmax(1)]) # Get the predicted emoji from the index

    # Training loop using RMSprop optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
    for epoch in range(500):
        for i in range(x_train.size()[0]):
            model.reset()
            model.loss(x_train[i], y_train[i]).backward()
            optimizer.step()
            optimizer.zero_grad()

        # Generate emoji for the input SEARCH_WORD every 10 epochs
        if epoch % 10 == 9:
            generate(SEARCH_WORD)