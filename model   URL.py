path('api/models/', views.get_all_models),
path('api/models/<int:model_id>/parameters/', views.get_model_parameters_by_id),
