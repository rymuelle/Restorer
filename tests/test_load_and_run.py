import torch
from Restorer.Restorer import Restorer


def test_model_prediction_with_valid_data():
    """
    Test that the model can process valid random data.
    """
    print("\nRunning test_model_prediction_with_valid_data...")
    width = 60
    enc_blks = [1, 1, 1, 0]
    vit_blks = [0, 0, 0, 1]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    expand_dims = 2
    cond_output = 32

    model = Restorer(
        in_channels=4,
        out_channels=4,
        width=width,
        middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blks,
        vit_blk_nums=vit_blks,
        dec_blk_nums=dec_blks,
        cond_input=1,
        expand_dims=expand_dims,
        cond_output=cond_output,
    )
    B, C, H, W = 1, 4, 128, 128
    test_data = torch.rand(B, C, H, W)
    conditioning = torch.rand(B, 1)
    result = model(test_data, conditioning)
    assert result.shape == test_data.shape
    print("Prediction test passed.")
