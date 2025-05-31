curl -X POST http://127.0.0.1:8020/api/meeting/bot/response/j-v1/test-meet-bot/ \
     -H "Content-Type: application/json" \
     -d '{
           "question": "Hello Jessica!",
           "channel_id": "1001",
           "phonenumber": "9876543210",
           "uuid": "123e4567-e89b-12d3-a456-426614174000",
           "call_disconnect": false
         }'
