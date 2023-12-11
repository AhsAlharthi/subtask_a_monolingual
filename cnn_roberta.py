import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score





# Load your training data from the JSONL file
def load_data_from_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line.strip())
            data.append(entry)
    return data


# Define the command-line arguments using argparse
parser = argparse.ArgumentParser(description='Train a CNN model on RoBERTa embeddings.')
parser.add_argument('--training_data', required=True, help='Path to the training data JSONL file')
parser.add_argument('--dev_data', required=True, help='Path to the dev data JSONL file')
parser.add_argument('--output_file', required=True, help='Path to the output JSONL file for validation results')

# Parse the command-line arguments
args = parser.parse_args()

# Load training data from the specified file
train_data = load_data_from_jsonl(args.training_data)
val_data = load_data_from_jsonl(args.dev_data)


# Load your dev data from the JSONL file
# val_data = load_data_from_jsonl('subtaskA_dev_monolingual.jsonl')  # Replace 'dev_set.jsonl' with your actual dev set file name

# Tokenize the text using the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Tokenize the training data
# train_data = load_data_from_jsonl('subtaskA_train_monolingual.jsonl')
# train_data = load_data_from_jsonl('10pct_balanced_subset.jsonl')

train_encodings = tokenizer([entry['text'] for entry in train_data], truncation=True, padding=True, return_tensors='pt')

# Tokenize the dev data
val_encodings = tokenizer([entry['text'] for entry in val_data], truncation=True, padding=True, return_tensors='pt')

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Prepare DataLoader for training and validation sets
train_labels = [entry['label'] for entry in train_data]
val_labels = [entry['label'] for entry in val_data]

train_dataset = CustomDataset(train_encodings, train_labels)
val_dataset = CustomDataset(val_encodings, val_labels)

batch_size = 12
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Instantiate RoBERTa model and move it to GPU
roberta_model = RobertaModel.from_pretrained('roberta-base')

# Freeze RoBERTa model parameters
for param in roberta_model.parameters():
    param.requires_grad = False

roberta_model = roberta_model.to('cuda')


class CNNModel(nn.Module):
    def __init__(self, in_channels=768, out_classes=2):
        super(CNNModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        # Residual blocks
        self.res_block1 = self._make_residual_block(256, 256)
        self.res_block2 = self._make_residual_block(256, 256)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(128, out_classes)

    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Input: [batch_size, channels, sequence_length]

        # Convolutional layers
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))

        # Residual blocks
        x = x + self.res_block1(x)
        x = x + self.res_block2(x)

        # Global average pooling
        x = x.mean(dim=2)

        # Fully connected layers
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)

        return x





# Instantiate CNN model and move it to GPU
in_channels = 768  # Adjust based on RoBERTa output
out_channels = 128  # Adjust based on your architecture
cnn_model = CNNModel().to('cuda')

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

import torch.optim.lr_scheduler as lr_scheduler

# Training loop
num_epochs = 3
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.005, total_iters=num_epochs)

for epoch in range(num_epochs):
    cnn_model.train()
    total_loss = 0.0

    for batch in train_dataloader:
        inputs = batch['input_ids'].to('cuda')
        labels = batch['labels'].to('cuda')

        roberta_output = roberta_model(input_ids=inputs, attention_mask=batch['attention_mask'].to('cuda')).last_hidden_state
        # print(f'Dimensions after RoBERTa output: {roberta_output.shape}')
        
        cnn_input = roberta_output[:, -2, :].unsqueeze(2)
        # print(f'Dimensions after selecting second-to-last layer: {cnn_input.shape}')
        
        cnn_output = cnn_model(cnn_input)
        # print(f'Dimensions after passing through CNN: {cnn_output.shape}')
        output = cnn_output.squeeze()
        # print(f'Dimensions after passing through CNN: {cnn_output.shape}')

        # Compute the loss
        loss = criterion(output, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate the total loss for this epoch
        total_loss += loss.item()
    
    scheduler.step()
    # Calculate the average loss for this epoch
    average_loss = total_loss / len(train_dataloader)

    # Print the average loss for monitoring
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}')



import json

# Validation
cnn_model.eval()
val_predictions = []
results_list = []  # Define results_list here

with torch.no_grad():
    for batch_id, batch in enumerate(val_dataloader):
        inputs = batch['input_ids'].to('cuda')
        labels = batch['labels'].to('cuda')

        roberta_output = roberta_model(input_ids=inputs, attention_mask=batch['attention_mask'].to('cuda')).last_hidden_state
        cnn_input = roberta_output[:, -2, :].unsqueeze(2)
        cnn_output = cnn_model(cnn_input)
        output = cnn_output.squeeze()

        _, preds = torch.max(output, dim=1)
        val_predictions.extend(preds.cpu().numpy())

        for i, pred in enumerate(preds.cpu().numpy()):
            results_list.append({"id": batch_id * val_dataloader.batch_size + i, "label": int(pred)})

val_accuracy = accuracy_score(val_labels, val_predictions)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Save results to JSONL file
output_file_path = args.output_file
with open(output_file_path, 'w') as jsonl_file:
    for result in results_list:
        jsonl_file.write(json.dumps(result) + '\n')

print(f'Validation results saved to {output_file_path}')
