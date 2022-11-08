import splitfolders

splitfolders.ratio("/home/acarlier/code/code_to_present_class_imbalanced_issue/Caltech_trap_DB", output="Caltech_trap_train_val_test", seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False)