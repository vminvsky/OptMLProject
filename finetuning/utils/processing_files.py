def process_col_names(train_df,validation_df, test_df, cfg):
    if ("OneStopEnglish" in cfg.data_processing.file_dir) & (cfg.training_name == "train.csv"):
        train_df["label"] = train_df["id"]
        test_df["label"] = test_df["id"]
        validation_df["label"] = validation_df["id"]
    elif ("OneStopEnglish" in cfg.data_processing.file_dir) & (cfg.testing_name == "test.csv"):
        print("yay")
        test_df["label"] = test_df["id"]
        validation_df["label"] = validation_df["id"]
        train_df["label"] = train_df["label_name"]
    elif ("sarcasm" in cfg.data_processing.file_dir) & (cfg.testing_name == "test.csv") & (cfg.training_name == "train.csv"):
        test_df["label"] = test_df["label"]
        train_df["label"] = train_df["label"]
        # train_df = train_df.groupby("label").sample(400).reset_index(drop=True)
    elif ("sarcasm" in cfg.data_processing.file_dir) & (cfg.testing_name == "test.csv") & ((cfg.training_name == "train_synth.csv") | (cfg.training_name == "ChatGPTTrain.csv") | (cfg.training_name == "TrainGPT4.csv") | (cfg.training_name == "TrainChatGPT.csv") | (cfg.training_name == "sarcasm_GPT_4_iterative.csv") | (cfg.training_name == "sarcasm_chatgpt_iterative.csv")):
        test_df["label"] = test_df["label"]
        train_df["label"] = train_df["label_name"]
        validation_df["label"] = validation_df["label"]
        # train_df = train_df.groupby("label").sample(400).reset_index(drop=True)
    else:
        # test_df["label"] = test_df["label_name"]
        test_df["label"] = test_df["label"]
        train_df["label"] = train_df["label"]
        validation_df["label"] = validation_df["label_name"]
    return train_df, validation_df, test_df