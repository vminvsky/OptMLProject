import matplotlib.pyplot as plt


def plot(accuracy, train_label, test_label, save_file):
    fig, ax = plt.subplots()
    num_points = len(accuracy)
    
    ax.plot(range(num_points), accuracy, label="Prediction")
        
    ax.set_xlabel("Epoch")
    name = save_file.split("/")[-1].split(".")[0]
    ax.set_ylabel(name)
    ax.set_title(f"{train_label}, {test_label}")
    plt.savefig(save_file)
    plt.show()