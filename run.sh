cd /app/attention/src
make
python3 attention.py 
/app/attention/bin/attention /app/attention/data/input.bin /app/attention/data/query.bin /app/attention/data/key.bin /app/attention/data/value.bin /app/attention/data/out_weight.bin /app/attention/data/out_bias.bin /app/attention/data/output.bin 1024 8 64 1024 1