#path = "output/run2/finetune_valid.log"
#path = "output/pruned_attention2/finetune_valid.log"
#path = "output/golden/finetune_valid.log"
#path = "output/prune_only_attention/finetune_valid.log"
#path = "output/pruned_attention_SSL/finetune_valid.log"
#path = "output/pruned_attention_SSL_flat/finetune_valid.log"
path = "output/csp128/finetune_valid.log"

first = True
top1 = 0
fp = open(path, "r")
for line in fp:
    if first:
        first = False
        continue
    line = line.split(',')
    #print(line)
    num = float(line[3])
    if num > top1:
        top1 = num
fp.close()
print("Top1 = {}".format(top1))
