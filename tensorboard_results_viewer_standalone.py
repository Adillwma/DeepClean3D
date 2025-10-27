from DC3D_V3.Helper_files.Tensorboard_helpers import TensorBoardLogger

if __name__ == "__main__":
    # Create logger (auto-launches TensorBoard)
    logger = TensorBoardLogger(logdir="tensorboard_logs_dir/", port=6006)
    logger.launch_dashboard()

    # User input (Enter to close)
    input("Press 'Enter' on your keyboard to close the dashboard...")

    logger.close()

