from pre_processing import generate_datasets
from transformer import train_and_fit
from post_processing import post_process

if __name__ == "__main__":
    print("Step1: Load training and testing datasets")
    train_dataset, test_dataset = generate_datasets()

    print("Step2: Train and validate model")
    train_and_fit(train_dataset, test_dataset)

    print("Step3: Display in tensor dashboard")
    post_process()
