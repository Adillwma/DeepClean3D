from DC3D_V3.Helper_files.Tensorboard_helpers import TensorBoardLogger


path = "A:/Users/Ada/Desktop/AE2D_encode_20251103-201607/"

if __name__ == "__main__":
    # Create logger (auto-launches TensorBoard)
    logger = TensorBoardLogger(logdir=path, rootdir=path, port=6006)
    logger.launch_dashboard()

    # User input (Enter to close)
    input("Press 'Enter' on your keyboard to close the dashboard...")

    logger.close()

