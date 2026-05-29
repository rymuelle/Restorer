import os
import torch
import mlflow.pytorch
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

RUN_ID = "c4d46b5deaca451a9adcff35c00b081f"  
DEVICE = "cuda"
model_uri = f"runs:/{RUN_ID}/model"
run = mlflow.get_run(RUN_ID)
CONFIG = run.data.params

output_filename = f"trace/{run.data.tags.get("mlflow.runName")}.pt"

IN_CHANNELS = 6
CROP_SIZE = 256

def main():
    print(f"Fetching model from MLflow tracking server: {model_uri}")
    
    try:
        model = mlflow.pytorch.load_model(model_uri)
        model = model.to(DEVICE)
        try:
            model.residual
        except:
            model.residual = False
        model.eval() 
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model from MLflow: {e}")
        return

    print(f"Creating dummy tensor of shape (1, {IN_CHANNELS}, {CROP_SIZE}, {CROP_SIZE}) on {DEVICE}")
    dummy_input = torch.rand(1, IN_CHANNELS, CROP_SIZE, CROP_SIZE, device=DEVICE)

    print("Compiling model via torch.jit.trace...")
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input)
            
        torch.jit.save(traced_model, output_filename)
        print(f"Compilation complete! Traced model saved as: '{output_filename}'")
    
    except Exception as e:
        print(f"Tracing compilation failed: {e}")
   
    # Sign model
    private_key_path = "keys/model_signing_private_key.pem"
    private_key = serialization.load_pem_private_key(
        open(private_key_path, "rb").read(),
        password=None,
    )
    output_filename_path = Path(output_filename)
    model_bytes = output_filename_path.read_bytes()
    signature = private_key.sign(
        model_bytes,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )

    sig_path = Path(output_filename + ".sig")
    sig_path.write_bytes(signature)

if __name__ == "__main__":
    main()