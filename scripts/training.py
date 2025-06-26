# goal: train neural network model to find the optimal internal params 
# that minimize the difference between predictions and ideal data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



# define configuration params
DATASET_FILE = 'data/dataset_2q_d5.npz'  # path to the dataset file
MODEL_OUTPUT_FILE = 'models/model0.pth'
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# define the neural network model
class ErrorMitigationNet(nn.Module):
    def __init__(self, input_shape):
        super(ErrorMitigationNet, self).__init__()
        self.layer1 = nn.Linear(input_shape, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, input_shape)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        # note: sofmax on the output layer returns a probability distribution instead of a number
        x = torch.softmax(self.layer3(x), dim=1)
        return x

# prepare data
def load_data(file_path, batch_size, test_size = 0.2):
    print(f"Loading data from{ file_path}...")
    data = np.load(file_path)
    X = data['x']
    y = data['y']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

    # simple conversion to torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)  
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    # create TensorDataset and DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print("Data loaded successfully.")
    return train_loader, val_loader, X_train.shape[1] # note: last element is the num_features

def plot_history(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/training_history_pytorch.png')
    print("Training history plot saved as figures/training_history_pytorch.png")

# main
if __name__ == "__main__":
    # load data
    train_loader, val_loader, num_features = load_data(DATASET_FILE, BATCH_SIZE) 
    model = ErrorMitigationNet(num_features)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    print("started training...")

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs) #forward pass
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # val pass
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                running_val_loss += loss.item()
            
            epoch_val_loss = running_val_loss / len(val_loader)
            val_losses.append(epoch_val_loss)
            print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        print("training finished.")

        torch.save(model.state_dict(), MODEL_OUTPUT_FILE)
        print(f"Model saved to {MODEL_OUTPUT_FILE}")

        plot_history(train_losses, val_losses)


