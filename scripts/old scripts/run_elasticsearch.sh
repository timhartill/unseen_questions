# run ElasticSearch
# kill with: pkill -F pid 
# alternatively find the pid in system monitor and run: kill -15 pid

cd /home/thar011/elasticsearch/elasticsearch-7.16.2

./bin/elasticsearch -d -p pid

#curl -X GET "localhost:9200/?pretty"

