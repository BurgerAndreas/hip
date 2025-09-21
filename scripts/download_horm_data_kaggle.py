import kagglehub
import os

if __name__ == "__main__":
    # export KAGGLEHUB_CACHE=/path/to/your/preferred/directory

    # Download latest version
    path = kagglehub.dataset_download(
        "yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm",
    )

    print("Path to dataset files:", path)
