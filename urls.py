path('j-v1/upload-file/', views.upload_any_file),
path('j-v1/all_models/', views.get_all_models_view, name = 'get_all_models_api'),
path('j-v1/model_parameters/', views.update_or_create_model_parameters_view, name = 'update_or_create_model_param_api'),



api calling


# Upload Files

curl --location 'http://1msg.1point1.in:3001/api/auth/j-v1/upload-file/' \
--header 'Content-Type: application/json' \
--form 'file=@"/C:/Users/puru.koli/Downloads/Introduction to Machine Learning.pdf"'

# All Models
curl --location 'http://1msg.1point1.in:3001/api/auth/j-v1/all_models/' \
--header 'Content-Type: application/json'

# Model Parameters
curl --location 'http://1msg.1point1.in:3001/api/auth/j-v1/model_parameters/?model_id=1' \
--header 'Content-Type: application/json'
