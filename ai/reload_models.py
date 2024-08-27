import torch

# Print the PyTorch version
print(f"Using PyTorch version: {torch.__version__}")

# Paths to the models
model_paths = [
    "/Users/adamforward/Desktop/chess/chess_rust/src/chess_cnn_checkpoint_epoch_15.pt",
    "/Users/adamforward/Desktop/chess/chess_rust/src/chess_cnn_model_final_3x3x3x3_f16.pt",
    "/Users/adamforward/Desktop/chess/chess_rust/src/chess_cnn_model_final_3xpool2xconv_f16.pt",
    "/Users/adamforward/Desktop/chess/chess_rust/src/chess_cnn_model_final_f16_3_convs.pt",
    "/Users/adamforward/Desktop/chess/chess_rust/src/chess_cnn_model_final_f16.pt"
]
model_paths = ["/Users/adamforward/Desktop/chess/chess_rust/src/model4.pt"]

# Function to load, set to eval mode, and re-save the model or state_dict as CPU


def reload_and_save_model_or_state_dict(model_path):
    try:
        # Load the model or state_dict on the CPU
        model_data = torch.load(model_path, map_location=torch.device('cpu'))

        if isinstance(model_data, torch.nn.Module):
            # If it's a model, set it to eval mode
            model_data.eval()
        else:
            # Otherwise, assume it's a state_dict
            print(f"Loaded a state_dict from {model_path}")

        # Re-save the model or state_dict to the same path
        torch.save(model_data, model_path)
        print(
            f"Successfully re-saved the model or state_dict at {model_path} as a CPU model")
    except Exception as e:
        print(
            f"Failed to process the model or state_dict at {model_path}: {e}")


# Process each model path
for model_path in model_paths:
    reload_and_save_model_or_state_dict(model_path)
