#path = "output/run1/valid.log"
path = "output/pruned_attention/valid.log"

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
