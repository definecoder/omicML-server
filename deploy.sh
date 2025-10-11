sudo  yum install git
sudo yum install docker
sudo systemctl start docker
sudo systemctl enable docker
git clone https://github.com/definecoder/PLAGL1-server.git
cd PLAGL1-server/
sudo docker rm -f plgl-server && sudo docker build -t wonderful_rubin . && sudo docker run --name plgl-server -p 8000:8000 -d wonderful_rubin:latest



sudo yum install -y nginx
sudo systemctl start nginx
sudo systemctl enable nginx
sudo systemctl status nginx
sudo nano /etc/nginx/conf.d/backend.omicml.org.conf

server {
    listen 80;
    server_name backend.omicml.org;

    # Allow large file uploads
    client_max_body_size 1G;

    # Increase client timeouts
    client_body_timeout 3600s;    # 1 hour
    client_header_timeout 3600s;  # 1 hour
    send_timeout 3600s;           # 1 hour

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Increase proxy timeouts for long-running requests
        proxy_connect_timeout 3600s;
        proxy_send_timeout 3600s;
        proxy_read_timeout 3600s;

        # Increase buffer sizes for large responses
        proxy_buffer_size 256k;
        proxy_buffers 8 512k;
        proxy_busy_buffers_size 512k;
    }
}


sudo nginx -t
sudo systemctl reload nginx

sudo dnf install -y certbot python3-certbot-nginx
sudo certbot --nginx -d backend.omicml.org


sudo docker exec 29f7958ab530 R -e 'if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager"); BiocManager::install("sva", ask=FALSE)'

