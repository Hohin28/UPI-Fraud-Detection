import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# 1. SETUP: Define the Model Structure
class AnomalyAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, input_dim), nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 2. LOAD DATA: Load your dataset (e.g., 'transactions.csv')
# IMPORTANT: Use ONLY rows that are NOT Fraud (Class = 0)
print("Loading data...")
# df = pd.read_csv("your_dataset.csv") 
# normal_data = df[df['Class'] == 0].drop(['Class'], axis=1).values

# --- FOR DEMO: Generating Fake Normal Data (Replace this with above lines) ---
input_dim = 30  # MATCH THIS TO YOUR COLUMNS
normal_data = np.random.rand(1000, input_dim).astype(np.float32) # Fake data 0-1
# -------------------------------------------------------------------------

# 3. TRAIN: Teach the model
tensor_data = torch.tensor(normal_data)
model = AnomalyAutoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training Anomaly Detector...")
for epoch in range(50):
    output = model(tensor_data)
    loss = criterion(output, tensor_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 4. SAVE: Create the weights file
torch.save(model.state_dict(), "anomaly_weights.pth")
print("SUCCESS: 'anomaly_weights.pth' has been created!")
