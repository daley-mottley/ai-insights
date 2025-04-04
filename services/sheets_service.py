import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from config import Config
import logging

class SheetsService:

    def __init__(self, credentials_json, sheet_name):
        self.sheet = None  # Initialize the sheet attribute
        self.folder_id = None  # Initialize the folder_id attribute

        if not Config.ENABLE_SHEETS_SERVICE:
            logging.info('Sheets service is disabled.')
            return

        self.logger = logging.getLogger(__name__)

        self.scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/documents"
        ]
        self.creds = ServiceAccountCredentials.from_json_keyfile_name(
            credentials_json, self.scope)
        self.client = gspread.authorize(self.creds)
        self.drive_service = build('drive', 'v3', credentials=self.creds)
        self.docs_service = build('docs', 'v1', credentials=self.creds)
        self.sheets_service = build('sheets', 'v4', credentials=self.creds)
        self.sheet = self._get_or_create_sheet(sheet_name)
        self.folder_id = self._get_or_create_folder(Config.GOOGLE_DRIVE_FOLDER_NAME)

    def _get_or_create_sheet(self, sheet_name):
        try:
            # Try to open the existing sheet by name
            sheet = self.client.open(sheet_name).sheet1
            self.logger.info(f'Using existing sheet: {sheet_name}')
            return sheet
        except gspread.SpreadsheetNotFound:
            self.logger.info(f'Sheet "{sheet_name}" not found. Creating a new sheet.')
            try:
                spreadsheet = {"properties": {"title": sheet_name}}
                spreadsheet = self.sheets_service.spreadsheets().create(
                    body=spreadsheet, fields="spreadsheetId"
                ).execute()
                sheet_id = spreadsheet.get("spreadsheetId")
                self.logger.info(f'Created new sheet: {sheet_name}, ID: {sheet_id}')
                return self.client.open_by_key(sheet_id).sheet1  # Open the newly created sheet
            except HttpError as e:
                self.logger.error(f'Error creating Google Sheet: {e}')
                raise

    def _get_or_create_folder(self, folder_name):
        try:
            response = self.drive_service.files().list(
                q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
                spaces='drive'
            ).execute()

            if len(response.get('files', [])) > 0:
                self.logger.info(f'Using existing folder: {folder_name}')
                return response['files'][0]['id']

            # Folder doesn't exist, create it
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = self.drive_service.files().create(body=file_metadata, fields='id').execute()
            self.logger.info(f'Created new folder: {folder_name}')
            return folder.get('id')

        except HttpError as error:
            self.logger.error(f'An error occurred while accessing Google Drive: {error}')
            raise

    def save_pdf_to_drive(self, file_path, file_name):
        if not Config.ENABLE_SHEETS_SERVICE or not self.folder_id:
            logging.info('Sheets service is disabled or folder ID is missing. Skipping PDF saving to Drive.')
            return None

        try:
            file_metadata = {
                'name': file_name,
                'parents': [self.folder_id],
                'mimeType': 'application/pdf'
            }
            media = MediaFileUpload(file_path, mimetype='application/pdf')
            file = self.drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            file_id = file.get('id')
            pdf_url = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"

            # Make the PDF publicly accessible
            self.drive_service.permissions().create(
                fileId=file_id,
                body={
                    'type': 'anyone',
                    'role': 'reader'
                }
            ).execute()

            self.logger.info(f'PDF saved to Google Drive: {pdf_url}')
            return pdf_url
        except Exception as e:
            self.logger.error(f'Error saving PDF to Google Drive: {e}')
            return None

    def create_google_doc(self, report_id, content):
        if not Config.ENABLE_SHEETS_SERVICE:
            logging.info('Sheets service is disabled. Skipping Google Doc creation.')
            return None

        self.logger.debug('Creating Google Doc for the report')
        try:
            # Create the Google Doc
            doc_title = f"AI Insights Report - {report_id}"
            body = {'title': doc_title}
            doc = self.docs_service.documents().create(body=body).execute()

            # Insert content into the document
            requests = [{
                'insertText': {
                    'location': {
                        'index': 1
                    },
                    'text': content
                }
            }]
            self.docs_service.documents().batchUpdate(
                documentId=doc.get('documentId'), body={'requests': requests}
            ).execute()

            # Make the document publicly accessible
            self.drive_service.permissions().create(
                fileId=doc.get('documentId'),
                body={
                    'type': 'anyone',
                    'role': 'reader'
                }
            ).execute()

            doc_url = f"https://docs.google.com/document/d/{doc.get('documentId')}/edit"
            self.logger.info(f'Document created successfully: {doc_url}')
            return doc_url
        except Exception as e:
            self.logger.error(f'Error creating Google Doc: {e}')
            return None

    def write_data(self, data: dict):
        if not Config.ENABLE_SHEETS_SERVICE:
            logging.info('Sheets service is disabled. Skipping data write to Google Sheets.')
            return

        try:
            # Write to Google Sheets
            if self.sheet and self.sheet.id:
                range_name = "A1"
                values = [list(data.values())]
                body = {"values": values}

                result = self.sheets_service.spreadsheets().values().append(
                    spreadsheetId=self.sheet.id,
                    range=range_name,
                    valueInputOption="USER_ENTERED",
                    insertDataOption="INSERT_ROWS",
                    body=body
                ).execute()

                self.logger.info(f"{result.get('updates').get('updatedCells')} cells updated in Google Sheets.")

        except HttpError as error:
            self.logger.error(f'An error occurred while writing data to Google Sheets: {error}')
        except Exception as e:
            self.logger.error(f'Error writing data: {e}')
