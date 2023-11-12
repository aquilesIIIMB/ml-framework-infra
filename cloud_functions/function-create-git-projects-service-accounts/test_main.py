import unittest
from unittest.mock import patch, MagicMock
import main

class TestMainFunctions(unittest.TestCase):

    @patch('main.storage.Client')
    def test_read_file_from_bucket(self, mock_storage_client):
        mock_blob = MagicMock()
        mock_blob.download_as_text.return_value = 'sample text'
        mock_storage_client.return_value.bucket.return_value.blob.return_value = mock_blob

        result = main.read_file_from_bucket('bucket', 'blob')
        self.assertEqual(result, 'sample text')

    @patch('main.service_account.Credentials.from_service_account_info')
    @patch('main.iam.ServiceAccountServiceClient')
    def test_create_service_account(
        self, mock_iam_client, mock_service_account_credentials
    ):
        mock_iam_client.return_value.create_service_account.return_value = MagicMock(
            name='new_account'
        )

        result = main.create_service_account(
            'account', 'description', 'project_id', {}
        )
        self.assertIn('Service account created', result)

# Add more tests as needed

if __name__ == '__main__':
    unittest.main()