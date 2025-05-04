import os
import uuid
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage

@csrf_exempt
def upload_any_file(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get('file')

        if not uploaded_file:
            return JsonResponse({"error": "No file provided"}, status=400)

        # Create a unique filename
        ext = os.path.splitext(uploaded_file.name)[1]
        filename = f"{uuid.uuid4()}{ext}"
        filepath = default_storage.save(f"uploads/{filename}", uploaded_file)
        full_path = default_storage.path(filepath)

        return JsonResponse({"file_path": full_path}, status=201)
