from pathlib import Path

import gdown


class GoogleDataDownloader:
    def __init__(self, file_url: str, file_name: str):
        self.data_folder = Path("data")
        self.file_path = self.data_folder / file_name
        self.data_folder.mkdir(exist_ok=True)
        self.download(file_url=file_url)

    def download(self, file_url: str):
        if not self.file_path.is_file():
            gdown.download(url=file_url,
                           output=str(self.file_path))
