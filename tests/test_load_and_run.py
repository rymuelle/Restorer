import torch
from Restorer.Restorer import Restorer


def test_model_prediction_with_valid_data():
    """
    Test that the model can process valid random data.
    """
    print("\nRunning test_model_prediction_with_valid_data...")
    width = 60
    enc_blks = [2, 2, 4, 6]
    middle_blk_num = 10
    dec_blks = [2, 2, 2, 2]

    model = Restorer(
        in_channels=4,
        out_channels=4,
        width=width,
        middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blks,
        dec_blk_nums=dec_blks,
        cond_input=1,
        cond_output=128,
    )
    B, C, H, W = 1, 4, 128, 128
    test_data = torch.rand(B, C, H, W)
    conditioning = torch.rand(B, 1)
    result = model(test_data, conditioning)
    assert result.shape == test_data.shape
    print("Prediction test passed.")
