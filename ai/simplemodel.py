import torch

# Define a simple model


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# Instantiate and script the model
model = SimpleModel()
scripted_model = torch.jit.script(model)

# Save the scripted model
torch.jit.save(scripted_model,
               "/Users/adamforward/Desktop/chess/chess_rust/src/simple_model_scripted.pt")
