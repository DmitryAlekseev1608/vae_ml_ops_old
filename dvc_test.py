from dvc.api import DVCFileSystem


url = "https://drive.google.com/drive/folders/1Lce2qugeTvVmcL_NzkLKfDQ0Y4h1nG6Z"
fs = DVCFileSystem(url, rev="main")
fs.get("data", "files", recursive=True)
