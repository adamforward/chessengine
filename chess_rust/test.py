import torch

model = torch.load(
    "/Users/adamforward/Desktop/chess/chess_rust/src/chess_cnn_checkpoint_epoch_15.pt",
    map_location=torch.device('cpu')
)
print(model)
