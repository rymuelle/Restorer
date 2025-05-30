import torch
from ConvNeXtUNet.convnextv2_unet import convnextv2unet_atto

def test_main_model_loads():
    """
    Test that the model can be loaded.
    """
    print("\nRunning test_main_model_loads...")
    model = convnextv2unet_atto()
    print(model)
    print("Model loaded.")

def test_model_prediction_with_valid_data():
    """
    Test that the model can process valid random data.
    """
    print("\nRunning test_model_prediction_with_valid_data...")
    model = convnextv2unet_atto() 
    test_data = torch.rand(1, 3, 32, 32)
    result = model(test_data)
    assert result.shape == test_data.shape
    print("Prediction test passed.")