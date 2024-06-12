import pika

# Địa chỉ IP của máy Windows chạy RabbitMQ
rabbitmq_host = '192.168.1.1'

rabbitmq_username = 'guest'
rabbitmq_password = 'guest'

parameters = pika.ConnectionParameters(host=rabbitmq_host)

# Kết nối tới RabbitMQ server
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

# Khai báo một queue (nếu queue chưa tồn tại, nó sẽ được tạo)
channel.queue_declare(queue='test_queue')

# Gửi một tin nhắn
message = 'Hello from Jetson!'
channel.basic_publish(exchange='',
                      routing_key='test_queue',
                      body=message)
print(f" [x] Sent '{message}'")

# Đóng kết nối
connection.close()