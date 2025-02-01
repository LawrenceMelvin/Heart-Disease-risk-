import kagglehub

# Download latest version
path = kagglehub.dataset_download("shriyashjagtap/heart-attack-risk-assessment-dataset")

print("Path to dataset files:", path)