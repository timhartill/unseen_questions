#
# kill with: pkill -F pid  Note: pid will be in ... elasticsearch-6.7.0/bin/elasticsearch
# alternatively find the pid in system monitor and run: kill -15 pid

cd elasticsearch-6.7.0
#bin/elasticsearch 2>&1 >/dev/null &
bin/elasticsearch -d -p pid
while ! curl -I localhost:9200 2>/dev/null;
do
  sleep 2;
done

#curl -X GET "localhost:9200/?pretty"
