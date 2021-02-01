class Config(object):
    # imgDirPath = '/media/gdp/date/MCG_project/MCG_qrst'
    imgDirPath = 'sample_dataset/samples_remove_info'
    labelDirPath = "sample_dataset/gt.csv"

    # Weight path or "none"
    weightFile = None

    # model save path
    backupDir = "backup"
    max_epochs = 2000
    save_interval = 10
    # e.g. 0,1,2,3
    gpus = [0]

    # multithreading
    num_workers = 2
    batch_size = 1

    # Solver params
    # adam or sgd
    solver = "adam"
    steps = [10000]
    scales = [0.1]
    learning_rate = 3e-4
    momentum = 0.9
    decay = 5e-4
    betas = (0.9, 0.98)

    num_classes = 5

