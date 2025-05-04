path('api/upload-file/', views.upload_any_file),




api calling


curl -X POST http://localhost:8000/api/upload-file/ \
  -F "file=@/path/to/your/file.pdf"
