    path('get_recording_file/', GetRecordingFileView.as_view(), name='get_recording_file'),


# views.py
import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse

class GetRecordingFileView(APIView):
    def post(self, request):
        file_path = request.data.get("file_path")
        if not file_path:
            return Response({"error": "file_path is required"}, status=status.HTTP_400_BAD_REQUEST)

        external_url = "http://192.168.0.181:3003/get_record_file"
        try:
            # Forward the file_path to the external server
            external_response = requests.post(external_url, json={"file_path": file_path}, stream=True)

            # If it's an audio file, return the binary content
            if external_response.status_code == 200:
                content_type = external_response.headers.get('Content-Type', 'application/octet-stream')
                return HttpResponse(
                    external_response.content,
                    content_type=content_type,
                    headers={
                        'Content-Disposition': f'attachment; filename="{file_path.split("/")[-1]}"'
                    }
                )
            else:
                return Response({
                    "error": "Failed to fetch file",
                    "details": external_response.text
                }, status=external_response.status_code)

        except requests.exceptions.RequestException as e:
            return Response({"error": str(e)}, status=status.HTTP_502_BAD_GATEWAY)
