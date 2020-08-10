python main.py \
--arch mobilenet_v2 \
--dist-url 'tcp://127.0.0.1:1234' \
--dist-backend 'nccl' \
--multiprocessing-distributed \
--world-size 1 \
--rank 0 \
--workers 16 \
--batch-size  164 \
--multiprocessing-distributed \
/workspace/downloads/ILSVRC2012


python main.py \
--arch mobilenet_v2 \
--gpu 1 \
--workers 14 \
--batch-size  256 \
--pretrained \
--evaluate \
/workspace/downloads/ILSVRC2012
# * Acc@1 71.878 Acc@5 90.286