class Config:
    # Paths
    CONTENT_PATH = '/content/gdrive/MyDrive/DIP_project/crowd.png'
    STYLE_PATH = 'none'
    SAVE_PATH = './outputs/'

    # Hyperparameters
    IMG_HEIGHT = 512
    IMG_WIDTH = 512
    MAX_STEP = 400
    LAMBDA_PATCH = 9000
    CONTENT_WEIGHT = 150
    STYLE_WEIGHT = 20000000
    LAMBDA_DIR = 500

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
