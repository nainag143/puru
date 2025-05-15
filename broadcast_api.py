    path('call_broadcast/', CallBroadcastView.as_view(), name='call_broadcast'),



# views.py
import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class CallBroadcastView(APIView):
    def post(self, request):
        payload = request.data

        # Validate required keys
        required_keys = ["agent_id", "user_id", "group_id", "phone_numbers"]
        if not all(key in payload for key in required_keys):
            return Response({"error": "Missing required fields."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Forward the request to the external API
            external_url = "http://192.168.0.181:3003/originate"
            response = requests.post(external_url, json=payload)

            return Response({
                "forwarded_payload": payload,
                "external_response": response.json()
            }, status=response.status_code)

        except requests.exceptions.RequestException as e:
            return Response({"error": str(e)}, status=status.HTTP_502_BAD_GATEWAY)
