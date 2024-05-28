from datasets import load_dataset

# Load the dataset

# Load the dataset
dataset = load_dataset("silatus/1k_Website_Screenshots_and_Metadata")

# Save the dataset locally
dataset.save_to_disk("D://int1//dataset")

# Save images and metadata locally
