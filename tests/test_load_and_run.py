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
    GCE_CONVS_nums = [3,3,2,2]

    model = Restorer(in_channels=4, out_channels=3 * 2 ** 2, width=width, middle_blk_num=middle_blk_num,
                    enc_blk_nums=enc_blks, dec_blk_nums=dec_blks, GCE_CONVS_nums=GCE_CONVS_nums,
                    cond_input = 1, cond_output=128)
    test_data = torch.rand(1, 3, 32, 32)
    result = model(torch.rand(1, 4, 128, 128), torch.rand(1, 1))
    assert result.shape == test_data.shape
    print("Prediction test passed.")
