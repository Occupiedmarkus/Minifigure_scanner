from google.cloud import storage
import os

class CloudStorage:
    def __init__(self):
        self.client = storage.Client()
        self.bucket_name = os.getenv('CLOUD_STORAGE_BUCKET')
        self.bucket = self.client.bucket(self.bucket_name)

    def upload_file(self, file_content, filename):
        blob = self.bucket.blob(filename)
        blob.upload_from_string(file_content)
        return blob.public_url
